"""
Pitcher True Performance Pipeline
==================================
End-to-end pipeline for evaluating pitcher true performance using:
  - Stage 1: PitchValueNet — pitch-level xRV model (luck removed)
  - Stage 2: Exponential decay aggregation (recency weighting)
  - Stage 3: Empirical Bayes shrinkage (uncertainty quantification)

Usage:
    python pipeline.py --mode train --season 2023
    python pipeline.py --mode score --season 2025
    python pipeline.py --mode full

Requirements:
    pip install pybaseball torch scikit-learn pandas numpy scipy
"""

import os
import pickle
import argparse
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from sklearn.preprocessing import RobustScaler, LabelEncoder
from pybaseball import statcast, batting_stats, playerid_reverse_lookup, pitching_stats
from scipy.stats import spearmanr

# ── Constants ──────────────────────────────────────────────────────────────────
ARTIFACT_DIR = "artifacts"
DATA_DIR     = "data"
LAM          = 0.046   # 14-day half-life for exponential decay

STUFF_COLS   = ["release_speed", "pfx_x", "pfx_z", "release_spin_rate", "release_extension"]
COMMAND_COLS = ["plate_x", "plate_z", "zone"]
CONT_COLS    = STUFF_COLS + COMMAND_COLS + ["pitch_number", "xwOBA"]
FEATURES     = CONT_COLS + ["pitch_type_enc", "prev_pitch_type_enc"]
TARGET       = "delta_run_exp"

# Best config from grid search
MODEL_CONFIG = dict(emb_dim=4, hidden_size=32, dropout=0.0)
TRAIN_CONFIG = dict(lr=1e-3, batch_size=512, epochs=10)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITION
# ══════════════════════════════════════════════════════════════════════════════

class PitchValueNet(nn.Module):
    """
    Shallow MLP with separate embedding tables for pitch_type and prev_pitch_type.

    Separate embedding tables capture positional role differences:
    a fastball thrown AS the current pitch differs from a fastball thrown BEFORE
    the current pitch (deception/sequencing signal).

    Architecture:
        [pitch_type_enc]      -> Embedding(n_pitch_types, emb_dim) -+
        [prev_pitch_type_enc] -> Embedding(n_pitch_types, emb_dim) -+-> concat -> Linear -> ReLU -> Linear(1)
        [continuous features] ------------------------------------------+
    """
    def __init__(self, n_pitch_types: int, emb_dim: int, n_continuous: int,
                 hidden_size: int, dropout: float):
        super().__init__()
        self.emb_current  = nn.Embedding(n_pitch_types, emb_dim)
        self.emb_previous = nn.Embedding(n_pitch_types, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2 + n_continuous, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x_cont, x_pitch_type, x_prev_pitch_type):
        x = torch.cat([
            self.emb_current(x_pitch_type),
            self.emb_previous(x_prev_pitch_type),
            x_cont
        ], dim=1)
        return self.mlp(x)


# ══════════════════════════════════════════════════════════════════════════════
# DATA PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def build_date_ranges(season: int) -> list:
    """Monthly chunks April-September, capped at yesterday for current season."""
    yesterday  = datetime.date.today() - datetime.timedelta(days=1)
    season_end = datetime.date(season, 9, 30)
    end_date   = min(yesterday, season_end)
    month_starts = [datetime.date(season, m, 1) for m in range(4, 10)]
    ranges = []
    for i, start in enumerate(month_starts):
        if start > end_date:
            break
        month_end = (month_starts[i + 1] - datetime.timedelta(days=1)
                     if i + 1 < len(month_starts) else season_end)
        ranges.append((start.strftime("%Y-%m-%d"), min(month_end, end_date).strftime("%Y-%m-%d")))
    return ranges


def pull_statcast(season: int, verbose: bool = True) -> pd.DataFrame:
    """Pull full season Statcast data in monthly chunks."""
    chunks = []
    for start, end in build_date_ranges(season):
        if verbose:
            print(f"  Pulling {start} -> {end}...")
        chunks.append(statcast(start_dt=start, end_dt=end))
    return pd.concat(chunks, ignore_index=True)


def build_features(df: pd.DataFrame, season: int,
                   scaler: RobustScaler = None,
                   le: LabelEncoder = None,
                   fit: bool = False) -> tuple:
    """
    Full feature engineering pipeline.

    Steps:
      1. Join batter seasonal xwOBA (league-avg impute for missing)
      2. Create prev_pitch_type lag feature (START for first pitch of each PA)
      3. Label encode pitch types (consistent encoding across current + prev)
      4. Classify SP vs RP by median pitches per appearance (threshold: 50)
      5. Drop rows with missing Statcast physics (<1% of pitches, MCAR)
      6. RobustScaler on continuous features (fit only on training data)

    Args:
        df     : raw Statcast DataFrame
        season : season year for FanGraphs xwOBA pull
        scaler : fitted RobustScaler (pass None if fit=True)
        le     : fitted LabelEncoder (pass None if fit=True)
        fit    : if True, fit scaler and le on this data (training only)

    Returns:
        df, scaler, le
    """
    print("  Building features...")

    # xwOBA join
    fg = batting_stats(season, qual=50)
    fg.columns = [c.strip() for c in fg.columns]
    xwoba_col = next((c for c in fg.columns if "woba" in c.lower() and "x" in c.lower()), "xwOBA")
    lookup = playerid_reverse_lookup(df["batter"].unique())[["key_mlbam", "key_fangraphs"]]
    lookup = lookup.rename(columns={"key_mlbam": "batter", "key_fangraphs": "IDfg"})
    df = df.merge(lookup, on="batter", how="left")
    df = df.merge(fg[["IDfg", xwoba_col]].rename(columns={xwoba_col: "xwOBA"}),
                  on="IDfg", how="left")
    df["xwOBA"] = df["xwOBA"].fillna(df["xwOBA"].mean())

    # prev_pitch_type lag feature
    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"])
    df["prev_pitch_type"] = (
        df.groupby(["game_pk", "at_bat_number"])["pitch_type"]
        .shift(1).fillna("START")
    )

    # label encoding
    all_types = pd.concat([df["pitch_type"], df["prev_pitch_type"]]).dropna().unique()
    if fit:
        le = LabelEncoder()
        le.fit(all_types)
    else:
        new_types = [t for t in all_types if t not in le.classes_]
        if new_types:
            print(f"  New pitch types: {new_types} -- appending to encoder")
            le.classes_ = np.concatenate([le.classes_, new_types])
    df["pitch_type_enc"]      = le.transform(df["pitch_type"].fillna("START"))
    df["prev_pitch_type_enc"] = le.transform(df["prev_pitch_type"])

    # SP/RP classification: median pitches per appearance >= 50 -> SP
    appearance = (
        df.groupby(["pitcher", "game_pk"])["pitch_number"].max().reset_index()
        .rename(columns={"pitch_number": "pitches_in_game"})
    )
    role_df = (
        appearance.groupby("pitcher")["pitches_in_game"].median().reset_index()
        .rename(columns={"pitches_in_game": "median_pitches"})
    )
    role_df["role"] = role_df["median_pitches"].apply(lambda x: "SP" if x >= 50 else "RP")
    df = df.merge(role_df[["pitcher", "role"]], on="pitcher", how="left")

    # drop missing physics (MCAR, <1%)
    before = len(df)
    df = df.dropna(subset=FEATURES + [TARGET])
    print(f"  Dropped {before - len(df):,} rows with missing physics ({(before-len(df))/before:.1%})")

    # scaling -- fit only on training data
    if fit:
        scaler = RobustScaler()
        scaler.fit(df[CONT_COLS])
        print("  Scaler fitted.")

    return df, scaler, le


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: TRAIN PitchValueNet
# ══════════════════════════════════════════════════════════════════════════════

def train_model(df: pd.DataFrame, scaler: RobustScaler,
                le: LabelEncoder) -> PitchValueNet:
    """
    Train PitchValueNet on pitch-level delta_run_exp.

    Time-series split (no leakage):
      Train: April - July | Test: August - September

    Design decisions:
      - MSELoss: delta_run_exp is unbounded continuous target
      - Label encoding + embeddings: compact, embedding-ready
      - Separate embedding tables: positional role differences for sequencing
      - Shallow MLP: depth doesn't help tabular data (confirmed by grid search)
      - RobustScaler: handles Statcast outliers (erroneous readings)
    """
    print("\n[Stage 1] Training PitchValueNet...")
    df["game_date"] = pd.to_datetime(df["game_date"])
    train = df[df["game_date"] < "2023-08-01"]
    test  = df[df["game_date"] >= "2023-08-01"]
    print(f"  Train: {len(train):,} | Test: {len(test):,}")

    def to_tensors(split):
        return (
            torch.tensor(scaler.transform(split[CONT_COLS]), dtype=torch.float32),
            torch.tensor(split["pitch_type_enc"].values, dtype=torch.long),
            torch.tensor(split["prev_pitch_type_enc"].values, dtype=torch.long),
            torch.tensor(split[TARGET].values, dtype=torch.float32),
        )

    Xc_tr, Xp_tr, Xpr_tr, y_tr = to_tensors(train)
    Xc_te, Xp_te, Xpr_te, y_te = to_tensors(test)

    train_loader = DataLoader(TensorDataset(Xc_tr, Xp_tr, Xpr_tr, y_tr),
                              batch_size=TRAIN_CONFIG["batch_size"], shuffle=True)
    test_loader  = DataLoader(TensorDataset(Xc_te, Xp_te, Xpr_te, y_te),
                              batch_size=TRAIN_CONFIG["batch_size"], shuffle=False)

    model     = PitchValueNet(n_pitch_types=len(le.classes_),
                              n_continuous=len(CONT_COLS), **MODEL_CONFIG)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=TRAIN_CONFIG["lr"])

    for epoch in range(TRAIN_CONFIG["epochs"]):
        model.train()
        tr_loss = 0
        for Xc, Xp, Xpr, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(Xc, Xp, Xpr).squeeze(), y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(y)
        tr_loss /= len(train)

        model.eval()
        va_loss = 0
        with torch.no_grad():
            for Xc, Xp, Xpr, y in test_loader:
                loss = criterion(model(Xc, Xp, Xpr).squeeze(), y)
                va_loss += loss.item() * len(y)
        va_loss /= len(test)
        print(f"  Epoch {epoch+1:02d}: train={tr_loss:.5f}  val={va_loss:.5f}")

    model.eval()
    return model


def score_pitches(df: pd.DataFrame, model: PitchValueNet,
                  scaler: RobustScaler) -> pd.DataFrame:
    """Generate pitch-level xRV predictions. Lower = better for pitcher."""
    Xc = torch.tensor(scaler.transform(df[CONT_COLS]), dtype=torch.float32)
    Xp = torch.tensor(df["pitch_type_enc"].values, dtype=torch.long)
    Xpr= torch.tensor(df["prev_pitch_type_enc"].values, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        preds = model(Xc, Xp, Xpr).squeeze().numpy()
    df = df.copy()
    df["pred_xrv"] = preds
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: EXPONENTIAL DECAY AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

def pitcher_score(df: pd.DataFrame, pitcher_id: int,
                  as_of_date, lam: float = LAM) -> float:
    """
    Exponentially weighted mean xRV for a pitcher as of a given date.
    Lower = better. Decay: w_t = exp(-lambda * days_ago).
    lambda=0.046 -> 14-day half-life (responsive for weekly fantasy decisions).
    """
    sub = df[(df["pitcher"] == pitcher_id) &
             (df["game_date"] <= pd.Timestamp(as_of_date))].copy()
    if len(sub) == 0:
        return np.nan
    T = pd.Timestamp(as_of_date)
    sub["days_ago"] = (T - sub["game_date"]).dt.days
    sub["weight"]   = np.exp(-lam * sub["days_ago"])
    return (sub["weight"] * sub["pred_xrv"]).sum() / sub["weight"].sum()


def build_leaderboard(df: pd.DataFrame, as_of_date,
                      role: str = "SP", min_pitches: int = 100,
                      lam: float = LAM) -> pd.DataFrame:
    """
    Stage 2: rank pitchers by exponentially weighted mean xRV.
    Ascending sort -- lower weighted_xrv = better pitcher.
    """
    sub = df[(df["game_date"] <= pd.Timestamp(as_of_date)) & (df["role"] == role)]
    eligible = (
        sub.groupby("pitcher")["pred_xrv"].count().reset_index()
        .rename(columns={"pred_xrv": "pitch_count"})
    )
    eligible = eligible[eligible["pitch_count"] >= min_pitches].copy()
    eligible["weighted_xrv"] = eligible["pitcher"].apply(
        lambda pid: pitcher_score(df, pid, as_of_date, lam)
    )
    return eligible.sort_values("weighted_xrv").reset_index(drop=True)


def pitcher_trajectory(df: pd.DataFrame, pitcher_id: int,
                       dates, lam: float = LAM) -> pd.DataFrame:
    """
    Weekly trajectory of weighted xRV with EWM smoothing (span=4).
    Two smoothing levers:
      - lam: controls score responsiveness (14-day half-life default)
      - EWM span=4: smooths visualization without affecting scores
    """
    records = []
    for date in dates:
        if len(df[(df["pitcher"] == pitcher_id) & (df["game_date"] <= date)]) >= 100:
            records.append({"date": date,
                            "weighted_xrv": pitcher_score(df, pitcher_id, date, lam)})
    traj = pd.DataFrame(records)
    if len(traj) > 1:
        traj["smoothed_xrv"] = traj["weighted_xrv"].ewm(span=4).mean()
    return traj


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3: EMPIRICAL BAYES SHRINKAGE
# ══════════════════════════════════════════════════════════════════════════════

def compute_empirical_bayes(df: pd.DataFrame, as_of_date,
                            role: str = "SP",
                            min_pitches: int = 100) -> pd.DataFrame:
    """
    Stage 3: Empirical Bayes (Normal-Normal conjugate) shrinkage.

    Model:
        mu_i  ~ N(mu_league, sigma_league^2)    pitcher true skill prior
        xRV_j ~ N(mu_i, sigma_noise^2)          appearance-level likelihood

    Closed-form posterior:
        posterior_var_i  = 1 / (1/sigma_league^2 + n_i/sigma_noise^2)
        posterior_mean_i = posterior_var_i * (mu_league/sigma_league^2
                           + n_i * mean_xrv_i / sigma_noise^2)

    Key properties:
        - Low n_i  -> heavy shrinkage toward mu_league (cup-of-coffee pitchers)
        - High n_i -> posterior_mean stays close to observed mean
        - posterior_std decreases as n_i increases (uncertainty estimate)

    Why Empirical Bayes over full hierarchical SVI:
        - AutoNormal mean-field fails for hierarchical models (sigma_league blows
          up to 0.37 regardless of prior -- likelihood overwhelms prior with 5000+
          observations and ~250 pitchers)
        - Non-centered parameterization improves convergence speed but doesn't fix
          mean-field's structural limitation (sigma_league and z_i are correlated
          in the true posterior; mean-field assumes independence)
        - Empirical Bayes is closed-form, instantaneous, and gives correct shrinkage
        - Clean separation of concerns: recency in Stage 2, shrinkage in Stage 3
    """
    sub = df[(df["game_date"] <= pd.Timestamp(as_of_date)) & (df["role"] == role)]

    # appearance-level aggregation
    appearance_df = (
        sub.groupby(["pitcher", "game_pk"])
        .agg(mean_xrv=("pred_xrv", "mean"))
        .reset_index()
    )

    # filter eligible pitchers
    pitch_counts = (
        sub.groupby("pitcher")["pred_xrv"].count().reset_index()
        .rename(columns={"pred_xrv": "pitch_count"})
    )
    eligible = pitch_counts[pitch_counts["pitch_count"] >= min_pitches]["pitcher"]
    appearance_df = appearance_df[appearance_df["pitcher"].isin(eligible)]

    if len(appearance_df) == 0:
        return pd.DataFrame()

    # estimate empirical hyperparameters from data
    mu_league    = appearance_df["mean_xrv"].mean()
    sigma_league = appearance_df.groupby("pitcher")["mean_xrv"].mean().std()
    sigma_noise  = appearance_df["mean_xrv"].std()

    print(f"  mu_league={mu_league:.5f}  sigma_league={sigma_league:.5f}  sigma_noise={sigma_noise:.5f}")

    # pitcher-level stats
    pitcher_stats = (
        appearance_df.groupby("pitcher")
        .agg(mean_xrv=("mean_xrv", "mean"), n_appearances=("mean_xrv", "count"))
        .reset_index()
        .merge(pitch_counts, on="pitcher", how="left")
    )

    # closed-form Normal-Normal posterior
    pitcher_stats["posterior_var"]  = 1 / (
        1 / sigma_league**2 + pitcher_stats["n_appearances"] / sigma_noise**2
    )
    pitcher_stats["posterior_mean"] = pitcher_stats["posterior_var"] * (
        mu_league / sigma_league**2
        + pitcher_stats["n_appearances"] * pitcher_stats["mean_xrv"] / sigma_noise**2
    )
    pitcher_stats["posterior_std"] = np.sqrt(pitcher_stats["posterior_var"])

    return (
        pitcher_stats[["pitcher", "pitch_count", "mean_xrv",
                        "n_appearances", "posterior_mean", "posterior_std"]]
        .sort_values("posterior_mean")
        .reset_index(drop=True)
    )


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate(leaderboard: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Spearman rank correlation vs FanGraphs Stuff+ and SIERA.
    Expected: Stuff+ negative r, SIERA positive r.
    """
    print("\n[Validation] Spearman vs FanGraphs...")
    try:
        ps = pitching_stats(season, qual=50)
        ps.columns = [c.strip() for c in ps.columns]
        lookup = (
            playerid_reverse_lookup(leaderboard["pitcher"].values)
            [["key_mlbam", "key_fangraphs"]]
            .rename(columns={"key_mlbam": "pitcher", "key_fangraphs": "IDfg"})
        )
        merged = leaderboard.merge(lookup, on="pitcher", how="left")
        cols   = ["IDfg"] + [c for c in ["Stuff+", "SIERA"] if c in ps.columns]
        merged = merged.merge(ps[cols], on="IDfg", how="left")

        results = []
        for metric in ["Stuff+", "SIERA"]:
            if metric not in merged.columns:
                continue
            clean = merged.dropna(subset=[metric, "weighted_xrv"])
            r, p = spearmanr(clean["weighted_xrv"], clean[metric])
            results.append({"Metric": metric, "Spearman r": round(r, 3),
                             "p-value": round(p, 4), "n": len(clean)})
            print(f"  {metric}: r={r:.3f}  p={p:.4f}  n={len(clean)}")
        return pd.DataFrame(results)
    except Exception as e:
        print(f"  Validation failed: {e}")
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# ARTIFACT I/O
# ══════════════════════════════════════════════════════════════════════════════

def save_artifacts(model: PitchValueNet, scaler: RobustScaler, le: LabelEncoder):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ARTIFACT_DIR, "model.pt"))
    pickle.dump(scaler, open(os.path.join(ARTIFACT_DIR, "scaler.pkl"), "wb"))
    pickle.dump(le,     open(os.path.join(ARTIFACT_DIR, "le.pkl"), "wb"))
    print(f"  Artifacts saved to {ARTIFACT_DIR}/")


def load_artifacts() -> tuple:
    le     = pickle.load(open(os.path.join(ARTIFACT_DIR, "le.pkl"), "rb"))
    scaler = pickle.load(open(os.path.join(ARTIFACT_DIR, "scaler.pkl"), "rb"))
    model  = PitchValueNet(n_pitch_types=len(le.classes_),
                           n_continuous=len(CONT_COLS), **MODEL_CONFIG)
    model.load_state_dict(torch.load(os.path.join(ARTIFACT_DIR, "model.pt"),
                                     map_location="cpu"))
    model.eval()
    print("  Artifacts loaded.")
    return model, scaler, le


def save_cache(df: pd.DataFrame, season: int):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"cache_{season}.parquet")
    df[["pitcher", "game_pk", "game_date", "role", "pred_xrv"]].to_parquet(path, index=False)
    print(f"  Cache saved to {path}")


def load_cache(season: int) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"cache_{season}.parquet")
    df = pd.read_parquet(path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    print(f"  Cache loaded: {len(df):,} pitches")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

def run_train(season: int = 2023):
    """Train pipeline: pull -> features -> train -> score -> validate -> save."""
    print(f"\n{'='*60}\nTRAINING PIPELINE -- Season {season}\n{'='*60}")

    print("\n[Data] Pulling Statcast...")
    raw = pull_statcast(season)
    print(f"  {len(raw):,} raw pitches")

    print("\n[Features] Engineering (fit=True)...")
    df, scaler, le = build_features(raw, season, fit=True)

    model = train_model(df, scaler, le)

    print("\n[Scoring] Pitch-level xRV...")
    df = score_pitches(df, model, scaler)

    as_of = f"{season}-09-30"
    stage2 = build_leaderboard(df, as_of)
    validate(stage2, season)

    print("\n[Save]")
    save_artifacts(model, scaler, le)
    save_cache(df, season)

    print(f"\n{'='*60}\nTraining complete.\n{'='*60}")
    return model, scaler, le, df


def run_score(season: int = None):
    """Score pipeline: load artifacts -> pull -> features -> score -> leaderboards -> save."""
    if season is None:
        season = datetime.date.today().year
        if datetime.date.today() < datetime.date(season, 4, 1):
            season -= 1

    print(f"\n{'='*60}\nSCORING PIPELINE -- Season {season}\n{'='*60}")

    print("\n[Load] Artifacts...")
    model, scaler, le = load_artifacts()

    print(f"\n[Data] Pulling {season} Statcast...")
    raw = pull_statcast(season)
    print(f"  {len(raw):,} raw pitches")

    print("\n[Features] Engineering (fit=False)...")
    df, scaler, le = build_features(raw, season, scaler=scaler, le=le, fit=False)

    print("\n[Stage 1] Scoring pitches...")
    df = score_pitches(df, model, scaler)

    as_of = datetime.date.today() - datetime.timedelta(days=1)

    print(f"\n[Stage 2] Leaderboard as of {as_of}...")
    stage2 = build_leaderboard(df, as_of)
    print(f"  {len(stage2)} eligible SPs")

    print(f"\n[Stage 3] Empirical Bayes shrinkage...")
    stage3 = compute_empirical_bayes(df, as_of)
    print("\n  Top 10 SPs:")
    print(stage3.head(10)[["pitcher", "pitch_count", "mean_xrv",
                            "posterior_mean", "posterior_std"]].to_string(index=False))

    print("\n[Save] Cache...")
    save_cache(df, season)

    print(f"\n{'='*60}\nScoring complete.\n{'='*60}")
    return df, stage2, stage3


def run_full(train_season: int = 2023, score_season: int = None):
    """Full pipeline: train then score."""
    run_train(train_season)
    return run_score(score_season)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pitcher True Performance Pipeline")
    parser.add_argument("--mode",   choices=["train", "score", "full"], default="score")
    parser.add_argument("--season", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "train":
        run_train(args.season or 2023)
    elif args.mode == "score":
        run_score(args.season)
    elif args.mode == "full":
        run_full(train_season=args.season or 2023)