"""
Pitcher True Performance Dashboard
-----------------------------------
Streamlit app for in-season fantasy pitcher evaluation.
Tracks stuff quality, command, and sequencing with luck removed.

Usage:
    streamlit run app.py

Requirements:
    pip install streamlit pybaseball torch scikit-learn pandas numpy scipy matplotlib
"""

import os
import pickle
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import torch
import torch.nn as nn
from pybaseball import statcast, batting_stats, playerid_reverse_lookup, playerid_lookup, pitching_stats
from sklearn.preprocessing import RobustScaler, LabelEncoder

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pitcher xRV Dashboard",
    page_icon="⚾",
    layout="wide",
)

# ── Constants ──────────────────────────────────────────────────────────────────
ARTIFACT_DIR    = "artifacts"
DATA_DIR        = "data"

def _current_season():
    today = datetime.date.today()
    return today.year if today >= datetime.date(today.year, 3, 20) else today.year - 1

CACHE_PATH      = os.path.join(DATA_DIR, f"cache_{_current_season()}.parquet")
MODEL_PATH      = os.path.join(ARTIFACT_DIR, "model.pt")
SCALER_PATH     = os.path.join(ARTIFACT_DIR, "scaler.pkl")
LE_PATH         = os.path.join(ARTIFACT_DIR, "le.pkl")
CACHE_MAX_DAYS  = 7
TRAIN_SEASON    = 2023
CURRENT_SEASON  = _current_season()
LAM             = 0.046

STUFF_COLS   = ["release_speed", "pfx_x", "pfx_z", "release_spin_rate", "release_extension"]
COMMAND_COLS = ["plate_x", "plate_z", "zone"]
CONT_COLS    = STUFF_COLS + COMMAND_COLS + ["pitch_number", "xwOBA"]
FEATURES     = CONT_COLS + ["pitch_type_enc", "prev_pitch_type_enc"]
TARGET       = "delta_run_exp"


def build_date_ranges(season: int) -> list:
    """Monthly chunks March 20 through yesterday (or Sep 30 for past seasons)."""
    yesterday  = datetime.date.today() - datetime.timedelta(days=1)
    season_end = datetime.date(season, 9, 30)
    end_date   = min(yesterday, season_end)
    month_starts = ([datetime.date(season, 3, 20)] +
                    [datetime.date(season, m, 1) for m in range(4, 10)])
    ranges = []
    for i, start in enumerate(month_starts):
        if start > end_date:
            break
        month_end = month_starts[i + 1] - datetime.timedelta(days=1) if i + 1 < len(month_starts) else season_end
        chunk_end = min(month_end, end_date)
        ranges.append((start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
    return ranges


def adaptive_min_pitches(as_of_date) -> int:
    """Scales linearly from 20 pitches (March 20) to 100 pitches (60 days later)."""
    season_start = pd.Timestamp(f"{pd.Timestamp(as_of_date).year}-03-20")
    days_in      = max(0, (pd.Timestamp(as_of_date) - season_start).days)
    return int(np.clip(20 + (100 - 20) * days_in / 60, 20, 100))


# ── Model definition ───────────────────────────────────────────────────────────
class PitchValueNet(nn.Module):
    def __init__(self, n_pitch_types, emb_dim, n_continuous, hidden_size, dropout):
        super().__init__()
        self.emb_current  = nn.Embedding(n_pitch_types, emb_dim)
        self.emb_previous = nn.Embedding(n_pitch_types, emb_dim)
        mlp_input_dim = emb_dim * 2 + n_continuous
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x_cont, x_pitch_type, x_prev_pitch_type):
        x = torch.cat([self.emb_current(x_pitch_type),
                       self.emb_previous(x_prev_pitch_type),
                       x_cont], dim=1)
        return self.mlp(x)


# ── Artifact loading ───────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    le     = pickle.load(open(LE_PATH, "rb"))
    scaler = pickle.load(open(SCALER_PATH, "rb"))
    model  = PitchValueNet(
        n_pitch_types=len(le.classes_),
        emb_dim=4,
        n_continuous=len(CONT_COLS),
        hidden_size=32,
        dropout=0.0,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model, scaler, le


# ── Data pipeline ──────────────────────────────────────────────────────────────
def pull_statcast():
    date_ranges = build_date_ranges(CURRENT_SEASON)
    chunks = []
    progress = st.progress(0, text="Pulling Statcast data...")
    for i, (start, end) in enumerate(date_ranges):
        progress.progress((i + 1) / len(date_ranges), text=f"Pulling {start} → {end}")
        chunks.append(statcast(start_dt=start, end_dt=end))
    progress.empty()
    return pd.concat(chunks, ignore_index=True)


def build_features(df, scaler, le):
    # xwOBA join — cached to avoid repeated FanGraphs requests + 403s
    fg_cache = os.path.join(DATA_DIR, f"fg_batters_{CURRENT_SEASON}.parquet")
    fg = None
    if os.path.exists(fg_cache):
        fg = pd.read_parquet(fg_cache)
    else:
        import time
        for attempt in range(3):
            try:
                fg = batting_stats(CURRENT_SEASON, qual=50)
                fg.columns = [c.strip() for c in fg.columns]
                os.makedirs(DATA_DIR, exist_ok=True)
                fg.to_parquet(fg_cache, index=False)
                break
            except Exception as e:
                if attempt < 2:
                    print(f"  FanGraphs request failed ({e}), retrying in 30s...")
                    time.sleep(30)
                else:
                    print("  FanGraphs unavailable — using league-avg xwOBA imputation")

    if fg is not None:
        fg.columns = [c.strip() for c in fg.columns]
        xwoba_col = next((c for c in fg.columns if "woba" in c.lower() and "x" in c.lower()), "xwOBA")
        lookup = playerid_reverse_lookup(df["batter"].unique())[["key_mlbam", "key_fangraphs"]]
        lookup = lookup.rename(columns={"key_mlbam": "batter", "key_fangraphs": "IDfg"})
        df = df.merge(lookup, on="batter", how="left")
        df = df.merge(fg[["IDfg", xwoba_col]].rename(columns={xwoba_col: "xwOBA"}),
                      on="IDfg", how="left")
    else:
        df["xwOBA"] = np.nan
    df["xwOBA"] = df["xwOBA"].fillna(df["xwOBA"].mean() if df["xwOBA"].notna().any() else 0.320)

    # prev_pitch_type
    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"])
    df["prev_pitch_type"] = (
        df.groupby(["game_pk", "at_bat_number"])["pitch_type"].shift(1).fillna("START")
    )

    # label encode
    all_types = pd.concat([df["pitch_type"], df["prev_pitch_type"]]).dropna().unique()
    new_types  = [t for t in all_types if t not in le.classes_]
    if new_types:
        le.classes_ = np.concatenate([le.classes_, new_types])
    df["pitch_type_enc"]      = le.transform(df["pitch_type"].fillna("START"))
    df["prev_pitch_type_enc"] = le.transform(df["prev_pitch_type"])

    # drop missing physics
    df = df.dropna(subset=FEATURES + [TARGET])

    # SP/RP classification — use max pitches early in season (< 3 appearances)
    appearance = (
        df.groupby(["pitcher", "game_pk"])["pitch_number"].max().reset_index()
        .rename(columns={"pitch_number": "pitches_in_game"})
    )
    role_df = (
        appearance.groupby("pitcher")["pitches_in_game"]
        .agg(max_pitches="max", n_appearances="count")
        .reset_index()
    )
    role_df["rep_pitches"] = np.where(
        role_df["n_appearances"] < 3,
        role_df["max_pitches"],
        appearance.groupby("pitcher")["pitches_in_game"].median().values
    )
    role_df["role"] = role_df["rep_pitches"].apply(lambda x: "SP" if x >= 50 else "RP")
    df = df.merge(role_df[["pitcher", "role"]], on="pitcher", how="left")

    return df


def score_pitches(df, model, scaler, le):
    X_cont  = torch.tensor(scaler.transform(df[CONT_COLS]), dtype=torch.float32)
    X_pitch = torch.tensor(df["pitch_type_enc"].values, dtype=torch.long)
    X_prev  = torch.tensor(df["prev_pitch_type_enc"].values, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        preds = model(X_cont, X_pitch, X_prev).squeeze().numpy()
    df = df.copy()
    df["pred_xrv"] = preds
    return df


def is_cache_fresh():
    if not os.path.exists(CACHE_PATH):
        return False
    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(CACHE_PATH))
    return (datetime.datetime.now() - mtime).days < CACHE_MAX_DAYS


def load_or_refresh_cache(model, scaler, le, force=False):
    if not force and is_cache_fresh():
        return pd.read_parquet(CACHE_PATH)
    with st.spinner("Pulling and scoring pitch data — this takes a few minutes..."):
        raw = pull_statcast()
        df  = build_features(raw, scaler, le)
        df  = score_pitches(df, model, scaler, le)
        os.makedirs(DATA_DIR, exist_ok=True)
        cols = ["pitcher", "game_pk", "game_date", "role", "pred_xrv", "events"]
        df[cols].to_parquet(CACHE_PATH, index=False)
    return df[cols]


# ── Scoring utilities ──────────────────────────────────────────────────────────
def pitcher_score(df, pitcher_id, as_of_date, lam=LAM):
    sub = df[(df["pitcher"] == pitcher_id) & (df["game_date"] <= as_of_date)].copy()
    if len(sub) == 0:
        return np.nan
    T = pd.Timestamp(as_of_date)
    sub["days_ago"] = (T - sub["game_date"]).dt.days
    sub["weight"]   = np.exp(-lam * sub["days_ago"])
    return (sub["weight"] * sub["pred_xrv"]).sum() / sub["weight"].sum()


def pitcher_trajectory(df, pitcher_id, dates, lam=LAM):
    records = []
    for date in dates:
        if len(df[(df["pitcher"] == pitcher_id) & (df["game_date"] <= date)]) >= 100:
            records.append({"date": date,
                            "weighted_xrv": pitcher_score(df, pitcher_id, date, lam)})
    traj = pd.DataFrame(records)
    if len(traj) >= 4:
        traj["smoothed_xrv"] = traj["weighted_xrv"].ewm(span=4).mean()
    return traj


def compute_empirical_bayes(df, as_of_date, role="SP", min_pitches=100, lam=LAM):
    sub = df[(df["game_date"] <= pd.Timestamp(as_of_date)) & (df["role"] == role)]
    appearance_df = (
        sub.groupby(["pitcher", "game_pk"])
        .agg(mean_xrv=("pred_xrv", "mean"))
        .reset_index()
    )
    pitch_counts = (
        sub.groupby("pitcher")["pred_xrv"].count()
        .reset_index().rename(columns={"pred_xrv": "pitch_count"})
    )
    eligible = pitch_counts[pitch_counts["pitch_count"] >= min_pitches]["pitcher"]
    appearance_df = appearance_df[appearance_df["pitcher"].isin(eligible)]
    if len(appearance_df) == 0:
        return pd.DataFrame()

    mu_league    = appearance_df["mean_xrv"].mean()
    sigma_league = appearance_df.groupby("pitcher")["mean_xrv"].mean().std()
    sigma_noise  = appearance_df["mean_xrv"].std()

    pitcher_stats = (
        appearance_df.groupby("pitcher")
        .agg(mean_xrv=("mean_xrv", "mean"), n_appearances=("mean_xrv", "count"))
        .reset_index()
    )
    pitcher_stats = pitcher_stats.merge(pitch_counts, on="pitcher", how="left")
    pitcher_stats["posterior_var"]  = 1 / (
        1 / sigma_league**2 + pitcher_stats["n_appearances"] / sigma_noise**2
    )
    pitcher_stats["posterior_mean"] = pitcher_stats["posterior_var"] * (
        mu_league / sigma_league**2
        + pitcher_stats["n_appearances"] * pitcher_stats["mean_xrv"] / sigma_noise**2
    )
    pitcher_stats["posterior_std"] = np.sqrt(pitcher_stats["posterior_var"])
    return pitcher_stats[["pitcher", "pitch_count", "mean_xrv", "posterior_mean", "posterior_std"]]


def compute_kbb(df, as_of_date, role="SP", min_pa=20):
    """K%, BB%, K%-BB% from Statcast events column."""
    if "events" not in df.columns:
        return pd.DataFrame()
    sub = df[
        (df["game_date"] <= pd.Timestamp(as_of_date)) &
        (df["role"] == role) &
        (df["events"].notna()) &
        (df["events"] != "")
    ].copy()
    if len(sub) == 0:
        return pd.DataFrame()

    sub["is_k"]  = sub["events"].isin({"strikeout", "strikeout_double_play"})
    sub["is_bb"] = sub["events"].isin({"walk", "intent_walk"})
    stats = (
        sub.groupby("pitcher")
        .agg(pa=("events", "count"), k=("is_k", "sum"), bb=("is_bb", "sum"))
        .reset_index()
    )
    stats = stats[stats["pa"] >= min_pa]
    stats["k_pct"]   = (stats["k"]  / stats["pa"]).round(3)
    stats["bb_pct"]  = (stats["bb"] / stats["pa"]).round(3)
    stats["kbb_pct"] = (stats["k_pct"] - stats["bb_pct"]).round(3)
    return stats[["pitcher", "pa", "k_pct", "bb_pct", "kbb_pct"]]


MANUAL_ID_TO_NAME = {
    700413: ("Tyler",   "Uberstine"),
    690155: ("Matt",    "Pushard"),
    700712: ("Walbert", "Urena"),
    680933: ("Joander", "Suarez"),
}


def add_names(leaderboard):
    ids = leaderboard["pitcher"].values
    try:
        lookup = playerid_reverse_lookup(ids)[["key_mlbam", "name_first", "name_last"]]
        lookup = lookup.rename(columns={"key_mlbam": "pitcher"})
        leaderboard = leaderboard.merge(lookup, on="pitcher", how="left")
    except Exception:
        leaderboard["name_first"] = ""
        leaderboard["name_last"]  = leaderboard["pitcher"].astype(str)

    missing = leaderboard["name_first"].isna() | (leaderboard["name_first"] == "")
    for pid, (first, last) in MANUAL_ID_TO_NAME.items():
        mask = missing & (leaderboard["pitcher"] == pid)
        leaderboard.loc[mask, "name_first"] = first
        leaderboard.loc[mask, "name_last"]  = last
    return leaderboard


def add_validation_metrics(leaderboard):
    """Join Stuff+ and SIERA from FanGraphs (cached)."""
    try:
        fg_pitch_cache = os.path.join(DATA_DIR, f"fg_pitchers_{CURRENT_SEASON}.parquet")
        if os.path.exists(fg_pitch_cache):
            ps = pd.read_parquet(fg_pitch_cache)
        else:
            import time
            ps = None
            for attempt in range(3):
                try:
                    ps = pitching_stats(CURRENT_SEASON, qual=1)
                    ps.columns = [c.strip() for c in ps.columns]
                    os.makedirs(DATA_DIR, exist_ok=True)
                    ps.to_parquet(fg_pitch_cache, index=False)
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(15)
                    else:
                        print(f"  FanGraphs pitching stats unavailable: {e}")
            if ps is None:
                return leaderboard

        ps.columns = [c.strip() for c in ps.columns]
        lookup = playerid_reverse_lookup(leaderboard["pitcher"].values)[["key_mlbam", "key_fangraphs"]]
        lookup = lookup.rename(columns={"key_mlbam": "pitcher", "key_fangraphs": "IDfg"})
        merged = leaderboard.merge(lookup, on="pitcher", how="left")
        cols   = ["IDfg"] + [c for c in ["Stuff+", "SIERA"] if c in ps.columns]
        merged = merged.merge(ps[cols], on="IDfg", how="left")
        return merged
    except Exception as e:
        print(f"  add_validation_metrics failed: {e}")
        return leaderboard


# ── App layout ─────────────────────────────────────────────────────────────────
def main():
    st.title("⚾ Pitcher True Performance Dashboard")
    st.caption("Luck-removed xRV model — stuff, command, sequencing, recency-weighted")

    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, LE_PATH]):
        st.error("Model artifacts not found. Save model.pt, scaler.pkl, le.pkl to artifacts/")
        return

    model, scaler, le = load_artifacts()

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        role        = st.selectbox("Role", ["SP", "RP"], index=0)
        lam         = st.slider("Decay λ (half-life)", 0.02, 0.10, LAM, step=0.005,
                                help="λ=0.046 → 14-day half-life")
        alpha       = st.slider("Risk α", 0.0, 1.0, 0.5, step=0.1,
                                help="0=posterior mean only | 1=full Sharpe ratio")
        sort_by     = st.selectbox("Sort leaderboard by",
                                   ["Combined Rank (xRV + K-BB%)",
                                    "Risk-Adj xRV only",
                                    "K-BB% only"],
                                   index=0)
        as_of_date  = st.date_input("As-of date", value=datetime.date.today())
        as_of_date  = pd.Timestamp(as_of_date)
        adaptive    = adaptive_min_pitches(as_of_date)
        min_pitches = st.slider("Min pitches", 10, 300, adaptive, step=10,
                                help=f"Auto-set to {adaptive} based on days into season")

        st.divider()
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.session_state["force_refresh"] = True

        cache_status = "✅ Fresh" if is_cache_fresh() else "⚠️ Stale / Missing"
        st.caption(f"Cache: {cache_status} — Season {CURRENT_SEASON}")

    # ── Load data ──────────────────────────────────────────────────────────────
    force = st.session_state.pop("force_refresh", False)

    @st.cache_data(show_spinner=False)
    def get_data(force_flag):
        return load_or_refresh_cache(model, scaler, le, force=force_flag)

    df = get_data(force)
    df["game_date"] = pd.to_datetime(df["game_date"])

    tab1, tab2 = st.tabs(["📋 Leaderboard", "📈 Trajectory"])

    # ── Tab 1: Leaderboard ─────────────────────────────────────────────────────
    with tab1:
        st.subheader(f"Top {role}s — as of {as_of_date}")

        @st.cache_data(show_spinner=False)
        def build_eb(role_, min_pitches_, lam_, as_of_date_str):
            as_of = pd.Timestamp(as_of_date_str)
            eb_   = compute_empirical_bayes(df, as_of, role=role_,
                                            min_pitches=min_pitches_, lam=lam_)
            if eb_.empty:
                return eb_
            eb_ = add_names(eb_)
            eb_ = add_validation_metrics(eb_)
            kbb_ = compute_kbb(df, as_of, role=role_)
            if not kbb_.empty:
                eb_ = eb_.merge(kbb_[["pitcher", "k_pct", "bb_pct", "kbb_pct"]],
                                on="pitcher", how="left")
            # K%/BB%/K%-BB% as percentage columns — computed inside cache so they persist
            if "k_pct" in eb_.columns:
                eb_["K%"]     = (eb_["k_pct"]  * 100).round(1)
                eb_["BB%"]    = (eb_["bb_pct"] * 100).round(1)
                eb_["K%-BB%"] = (eb_["kbb_pct"] * 100).round(1)
            # combined rank: avg of xRV rank and K-BB% rank
            eb_["xrv_rank"] = eb_["posterior_mean"].rank(ascending=True)
            if "K%-BB%" in eb_.columns:
                eb_["kbb_rank"]      = eb_["K%-BB%"].rank(ascending=False, na_option="bottom")
                eb_["combined_rank"] = ((eb_["xrv_rank"] + eb_["kbb_rank"]) / 2).round(1)
            else:
                eb_["combined_rank"] = eb_["xrv_rank"]
            return eb_

        with st.spinner("Building leaderboard..."):
            # diagnostic info
            total_pitches  = len(df)
            role_pitches   = len(df[(df["game_date"] <= as_of_date) & (df["role"] == role)])
            n_eligible     = len(df[(df["game_date"] <= as_of_date) & (df["role"] == role)]
                                 .groupby("pitcher")["pred_xrv"].count()
                                 .pipe(lambda s: s[s >= min_pitches]))
            st.info(
                f"Cache: {total_pitches:,} pitches | "
                f"Role={role}: {role_pitches:,} | "
                f"Eligible (≥{min_pitches}): {n_eligible} | "
                f"Date range: {df['game_date'].min().date()} → {df['game_date'].max().date()} | "
                f"Roles: {df['role'].value_counts().to_dict()}"
            )

            eb = build_eb(role, min_pitches, lam, str(as_of_date))
            if eb.empty:
                st.warning("Not enough data for current filters.")
            else:
                # risk-adjusted score and sort
                eb["risk_adj_score"] = eb["posterior_mean"] / (eb["posterior_std"] ** alpha)
                if sort_by == "Combined Rank (xRV + K-BB%)" and "combined_rank" in eb.columns:
                    eb = eb.sort_values("combined_rank", ascending=True).reset_index(drop=True)
                elif sort_by == "K-BB% only" and "K%-BB%" in eb.columns:
                    eb = eb.sort_values("K%-BB%", ascending=False).reset_index(drop=True)
                else:
                    eb = eb.sort_values("risk_adj_score", ascending=True).reset_index(drop=True)

                eb["Name"] = eb["name_first"].str.title() + " " + eb["name_last"].str.title()
                eb["Rank"] = range(1, len(eb) + 1)
                eb["Confidence"] = eb["posterior_std"].apply(
                    lambda s: "🟢 High" if s < 0.002 else ("🟡 Medium" if s < 0.003 else "🔴 Low")
                )

                # pitcher filter
                all_names = eb["Name"].tolist()
                selected  = st.multiselect(
                    "Filter to specific pitchers (leave empty to show all)",
                    options=all_names, default=[],
                    placeholder="Type a name to search...",
                )
                display_eb = eb[eb["Name"].isin(selected)] if selected else eb

                display_cols = ["Rank", "Name", "combined_rank", "risk_adj_score",
                                "posterior_mean", "mean_xrv", "pitch_count",
                                "posterior_std", "Confidence"]
                rename_map   = {
                    "combined_rank":  "Combined ↓",
                    "risk_adj_score": f"Risk-Adj xRV (α={alpha}) ↓",
                    "posterior_mean": "Posterior xRV",
                    "mean_xrv":       "Raw xRV",
                    "pitch_count":    "Pitches",
                    "posterior_std":  "Uncertainty",
                }
                for col in ["Stuff+", "SIERA", "K%", "BB%", "K%-BB%"]:
                    if col in eb.columns:
                        display_cols.append(col)

                st.dataframe(
                    display_eb[display_cols].rename(columns=rename_map).set_index("Rank"),
                    use_container_width=True,
                    height=600,
                )
                st.caption(
                    f"Combined = avg(xRV rank, K-BB% rank). "
                    f"α={alpha}: 0=ignore uncertainty, 1=Sharpe ratio. "
                    "🟢 High / 🟡 Medium / 🔴 Low = confidence from sample size."
                )

                # ── Posterior distribution plot ────────────────────────────
                st.subheader("Posterior xRV Distribution")
                st.caption("Wider = less confident. Lower = better.")
                plot_names = selected if selected else display_eb["Name"].head(10).tolist()
                plot_df    = eb[eb["Name"].isin(plot_names)].copy()

                if not plot_df.empty:
                    fig2, ax2 = plt.subplots(figsize=(10, 4))
                    fig2.patch.set_facecolor("#0e1117")
                    ax2.set_facecolor("#0e1117")
                    x = np.linspace(
                        plot_df["posterior_mean"].min() - 4 * plot_df["posterior_std"].max(),
                        plot_df["posterior_mean"].max() + 4 * plot_df["posterior_std"].max(),
                        500
                    )
                    cmap   = plt.cm.get_cmap("tab10", len(plot_df))
                    colors = [cmap(i) for i in range(len(plot_df))]
                    for (_, row), color in zip(plot_df.iterrows(), colors):
                        mu, sigma = row["posterior_mean"], row["posterior_std"]
                        pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
                        ax2.plot(x, pdf, color=color, linewidth=2, label=row["Name"])
                        ax2.fill_between(x, pdf, alpha=0.08, color=color)
                        ax2.axvline(mu, color=color, linewidth=1, linestyle="--", alpha=0.5)
                    ax2.axvline(0, color="white", linewidth=0.5, linestyle=":", alpha=0.3)
                    ax2.set_xlabel("Posterior xRV (lower = better)", color="white")
                    ax2.set_ylabel("Density", color="white")
                    ax2.tick_params(colors="white")
                    ax2.spines[:].set_color("#333")
                    ax2.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8,
                               loc="upper right", ncol=2)
                    plt.tight_layout()
                    st.pyplot(fig2)

    # ── Tab 2: Trajectory ──────────────────────────────────────────────────────
    with tab2:
        st.subheader("Pitcher Trajectory Comparison")

        def search_pitchers(last_name):
            if not last_name.strip():
                return []
            try:
                result = playerid_lookup(last_name.strip())
                if len(result) == 0:
                    return []
                options = []
                for _, row in result.iterrows():
                    if pd.notna(row.get("key_mlbam")) and pd.notna(row.get("name_first")):
                        full = f"{str(row['name_first']).title()} {str(row['name_last']).title()}"
                        options.append((int(row["key_mlbam"]), full))
                return options
            except Exception:
                return []

        col1, col2 = st.columns(2)
        with col1:
            p1_last = st.text_input("Pitcher 1 last name", value="senga")
            p1_options = search_pitchers(p1_last)
            if p1_options:
                p1_choice = st.selectbox("Select Pitcher 1",
                                         options=[n for _, n in p1_options], key="p1_select")
                p1_id    = next(pid for pid, n in p1_options if n == p1_choice)
                p1_label = p1_choice
            else:
                st.warning(f"No results for '{p1_last}'")
                p1_id, p1_label = None, p1_last

        with col2:
            p2_last = st.text_input("Pitcher 2 last name", value="elder")
            p2_options = search_pitchers(p2_last)
            if p2_options:
                p2_choice = st.selectbox("Select Pitcher 2",
                                         options=[n for _, n in p2_options], key="p2_select")
                p2_id    = next(pid for pid, n in p2_options if n == p2_choice)
                p2_label = p2_choice
            else:
                st.warning(f"No results for '{p2_last}'")
                p2_id, p2_label = None, p2_last

        if st.button("Plot Trajectories"):
            dates = pd.date_range(
                start=f"{CURRENT_SEASON}-04-01",
                end=(datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
                freq="W"
            )
            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#0e1117")
            colors = ["#4fc3f7", "#ff8a65"]
            for pid, label, color in [(p1_id, p1_label, colors[0]),
                                       (p2_id, p2_label, colors[1])]:
                if pid is None:
                    st.warning(f"Could not find pitcher: {label}")
                    continue
                traj = pitcher_trajectory(df, pid, dates, lam=lam)
                if traj.empty:
                    st.warning(f"Not enough data for {label}")
                    continue
                ax.plot(traj["date"], traj["weighted_xrv"], alpha=0.3, color=color, linewidth=1)
                if "smoothed_xrv" in traj.columns:
                    ax.plot(traj["date"], traj["smoothed_xrv"],
                            color=color, linewidth=2.5, label=label)
                else:
                    ax.plot(traj["date"], traj["weighted_xrv"],
                            color=color, linewidth=2.5, label=f"{label} (raw)")
            ax.axhline(0, linestyle="--", color="#555", alpha=0.6)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            ax.set_ylabel("Weighted xRV (lower = better)", color="white")
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#333")
            ax.legend(facecolor="#1a1a2e", labelcolor="white")
            plt.tight_layout()
            st.pyplot(fig)
            st.caption("Solid = EWM smoothed (span=4 weeks). Faint = raw weekly score.")


if __name__ == "__main__":
    main()