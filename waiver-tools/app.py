"""
Net Power Score Dashboard — 2026
Streamlit app that packages the net_power_score notebook pipeline and shows
NPS rankings alongside current-season xwOBA, wOBA, HR, K%, and BB%.
"""
import os, warnings, datetime, time, functools
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
import pyro.infer.autoguide as autoguide
from pyro.optim import Adam
from functools import partial
from sklearn.preprocessing import StandardScaler
import polars as pl
from pybaseballstats.statcast import pitch_by_pitch_data
from pybaseballstats.utils.retrosheet_utils import _get_people_data

# ── Paths ─────────────────────────────────────────────────────────────────────
WAIVER_DIR  = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR   = os.path.join(WAIVER_DIR, 'data')
OUTPUT_DIR  = os.path.join(WAIVER_DIR, 'outputs')
FULL_CACHE  = os.path.join(OUTPUT_DIR, 'nps_2026_full.parquet')
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_YEARS      = [2021, 2022, 2023, 2024]
TEST_YEAR        = 2025
PRED_YEAR        = 2026
CURRENT_YEAR     = PRED_YEAR
MIN_PA_TRAIN     = 100
MIN_PA_PRED      = 20
SHRINK_N0        = 50
PRIOR_DECAY      = 0.5
W_HR             = 4.94
W_BB             = 1.23
W_K              = 0.50
RISK_ALPHA       = 0.5
N_SAMPLES        = 2000
SVI_STEPS        = 3000
INCREMENTAL_LOOKBACK_DAYS = 3

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# ── Feature definitions ───────────────────────────────────────────────────────
SWING_DESCS = frozenset([
    'swinging_strike', 'swinging_strike_blocked',
    'foul', 'foul_tip', 'foul_bunt', 'missed_bunt',
    'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score',
])
WHIFF_DESCS = frozenset(['swinging_strike', 'swinging_strike_blocked', 'missed_bunt'])
OUT_ZONE    = frozenset([11, 12, 13, 14])
FAIR_DESCS  = frozenset(['hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'])
K_EVENTS    = frozenset(['strikeout', 'strikeout_double_play'])
BB_EVENTS   = frozenset(['walk', 'intent_walk'])
PA_EVENTS   = frozenset([
    'single', 'double', 'triple', 'home_run',
    'walk', 'intent_walk', 'hit_by_pitch',
    'strikeout', 'strikeout_double_play',
    'field_out', 'grounded_into_double_play', 'double_play',
    'force_out', 'fielders_choice', 'fielders_choice_out',
    'field_error', 'sac_fly', 'sac_bunt',
    'sac_fly_double_play', 'triple_play', 'catcher_interf',
])
FEAT_DENOM = {
    'whiff_pct':      'swings',
    'chase_pct':      'out_zone',
    'hard_hit_pct':   'in_play',
    'sweet_spot_pct': 'in_play',
}
FEAT_K  = ['whiff_pct', 'chase_pct']
FEAT_BB = ['chase_pct']
FEAT_HR = ['hard_hit_pct', 'sweet_spot_pct']

MANUAL_ID_TO_NAME = {
    608070: "José Ramírez",   608324: "Alex Bregman",
    621493: "Taylor Ward",    624413: "Pete Alonso",
    665742: "Juan Soto",      666182: "Bo Bichette",
    686948: "Drake Baldwin",  691777: "Max Muncy",
    691781: "Brady House",    695578: "James Wood",
    695657: "Colson Montgomery", 695734: "Daylen Lile",
    701350: "Roman Anthony",  701398: "Sal Stewart",
    802139: "JJ Wetherholt", 805300: "Jakob Marsee",
    805808: "Kevin McGonigle", 808959: "Munetaka Murakami",
    679529: "Spencer Torkelson", 800050: "Chase DeLauter",
    683357: 'Owen Caissie',
}

# ── Data helpers ──────────────────────────────────────────────────────────────
def _fetch_range(yr: int, start: str, end: str) -> pd.DataFrame:
    start_d = datetime.date.fromisoformat(start)
    end_d   = datetime.date.fromisoformat(end)
    month_starts = []
    d = start_d.replace(day=1)
    while d <= end_d:
        month_starts.append(d)
        d = (d.replace(day=28) + datetime.timedelta(days=4)).replace(day=1)
    chunks = []
    for ms in month_starts:
        next_m = (ms.replace(day=28) + datetime.timedelta(days=4)).replace(day=1)
        me = next_m - datetime.timedelta(days=1)
        chunk_start = max(ms, start_d).isoformat()
        chunk_end   = min(me, end_d).isoformat()
        for attempt in range(1, 4):
            try:
                result = pitch_by_pitch_data(start_date=chunk_start, end_date=chunk_end, force_collect=True)
                if result is not None:
                    df = result.to_pandas()
                    for col in df.select_dtypes('object').columns:
                        non_null = df[col].dropna()
                        if len(non_null) > 0:
                            n_num = pd.to_numeric(non_null, errors='coerce').notna().sum()
                            if n_num / len(non_null) >= 0.5:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                    chunks.append(df)
                break
            except Exception as e:
                if attempt < 3:
                    time.sleep(10 * attempt)
                else:
                    pass
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def _fetch_year(yr: int) -> pd.DataFrame:
    cache = os.path.join(CACHE_DIR, f'statcast_{yr}.parquet')
    is_current = (yr == CURRENT_YEAR)
    today = datetime.date.today()
    if not is_current:
        if os.path.exists(cache):
            return pd.read_parquet(cache)
        df = _fetch_range(yr, f'{yr}-03-01', f'{yr}-10-31')
        df.to_parquet(cache, index=False)
        return df
    if os.path.exists(cache):
        age_s = (datetime.datetime.now() -
                 datetime.datetime.fromtimestamp(os.path.getmtime(cache))).total_seconds()
        if age_s < 86400:
            return pd.read_parquet(cache)
        existing  = pd.read_parquet(cache)
        last_date = pd.to_datetime(existing['game_date']).dt.date.max()
        fetch_from = last_date - datetime.timedelta(days=INCREMENTAL_LOOKBACK_DAYS)
        delta = _fetch_range(yr, fetch_from.isoformat(), today.isoformat())
        if not delta.empty:
            keep = pd.to_datetime(existing['game_date']).dt.date < fetch_from
            df = pd.concat([existing[keep], delta], ignore_index=True)
        else:
            df = existing
    else:
        df = _fetch_range(yr, f'{yr}-03-01', today.isoformat())
    df.to_parquet(cache, index=False)
    return df


def compute_season_features(raw: pd.DataFrame, yr: int, min_pa: int) -> pd.DataFrame:
    raw = raw.copy()
    desc   = raw['description'].fillna('').astype(str)
    zone   = pd.to_numeric(raw.get('zone',         pd.Series(np.nan, index=raw.index)), errors='coerce')
    events = raw['events'].fillna('').astype(str)
    lv     = pd.to_numeric(raw.get('launch_speed', pd.Series(np.nan, index=raw.index)), errors='coerce')
    la     = pd.to_numeric(raw.get('launch_angle', pd.Series(np.nan, index=raw.index)), errors='coerce')
    raw['_swing']      = desc.isin(SWING_DESCS)
    raw['_whiff']      = desc.isin(WHIFF_DESCS)
    raw['_out_zone']   = zone.isin(OUT_ZONE)
    raw['_chase']      = raw['_swing'] & raw['_out_zone']
    raw['_in_play']    = desc.isin(FAIR_DESCS)
    raw['_hard_hit']   = raw['_in_play'] & (lv >= 95)
    raw['_sweet_spot'] = raw['_in_play'] & (la >= 8) & (la <= 32)
    raw['_pa']         = events.isin(PA_EVENTS)
    raw['_k']          = events.isin(K_EVENTS)
    raw['_bb']         = events.isin(BB_EVENTS)
    raw['_hr']         = events == 'home_run'
    g = raw.groupby('batter')
    stats = pd.DataFrame({
        'season':       yr,
        'swings':       g['_swing'].sum(),
        'whiffs':       g['_whiff'].sum(),
        'out_zone':     g['_out_zone'].sum(),
        'chases':       g['_chase'].sum(),
        'in_play':      g['_in_play'].sum(),
        'hard_hits':    g['_hard_hit'].sum(),
        'sweet_spots':  g['_sweet_spot'].sum(),
        'PA':           g['_pa'].sum(),
        'K':            g['_k'].sum(),
        'BB':           g['_bb'].sum(),
        'HR':           g['_hr'].sum(),
    }).reset_index()
    stats = stats[stats['PA'] >= min_pa].copy()
    stats['whiff_pct']      = stats['whiffs']      / stats['swings'].clip(lower=1)
    stats['chase_pct']      = stats['chases']      / stats['out_zone'].clip(lower=1)
    stats['hard_hit_pct']   = stats['hard_hits']   / stats['in_play'].clip(lower=1)
    stats['sweet_spot_pct'] = stats['sweet_spots'] / stats['in_play'].clip(lower=1)
    stats['k_per_pa']       = stats['K']  / stats['PA']
    stats['bb_per_pa']      = stats['BB'] / stats['PA']
    stats['hr_per_pa']      = stats['HR'] / stats['PA']
    return stats


@functools.lru_cache(maxsize=1)
def _id_map() -> dict:
    people = _get_people_data()
    result = people.filter(pl.col('key_mlbam').is_not_null()).select(['key_mlbam', 'name_first', 'name_last'])
    id_map = {}
    for row in result.iter_rows(named=True):
        name = f"{row['name_first'] or ''} {row['name_last'] or ''}".strip().title()
        id_map[int(row['key_mlbam'])] = name
    id_map.update(MANUAL_ID_TO_NAME)
    return id_map


def build_dataset(years: list, min_pa: int) -> pd.DataFrame:
    frames = []
    for yr in years:
        raw = _fetch_year(yr)
        if raw.empty:
            continue
        stats = compute_season_features(raw, yr, min_pa=min_pa)
        frames.append(stats)
    df = pd.concat(frames, ignore_index=True)
    id_map = _id_map()
    df['name'] = df['batter'].astype(int).map(id_map).fillna(df['batter'].astype(str))
    return df


def compute_player_priors(train_df: pd.DataFrame, decay: float = PRIOR_DECAY) -> dict:
    max_yr = train_df['season'].max()
    df = train_df.copy()
    df['_decay_w'] = decay ** (max_yr - df['season'])
    priors = {}
    for batter, grp in df.groupby('batter'):
        player_p = {}
        for feat, denom_col in FEAT_DENOM.items():
            w = grp['_decay_w'] * grp[denom_col]
            total_w = w.sum()
            player_p[feat] = float((w * grp[feat]).sum() / total_w) if total_w > 0 else None
        priors[int(batter)] = player_p
    return priors


def shrink_pred_features(pred_df: pd.DataFrame, league_means: dict,
                         player_priors: dict = None, n0: int = SHRINK_N0) -> pd.DataFrame:
    df = pred_df.copy()
    for feat, denom_col in FEAT_DENOM.items():
        n = df[denom_col].clip(lower=0)
        if player_priors:
            prior = (
                df['batter'].astype(int)
                .map(lambda bid, f=feat: (player_priors.get(bid) or {}).get(f))
                .fillna(league_means[feat])
                .astype(float)
            )
        else:
            prior = league_means[feat]
        df[feat] = (n * df[feat] + n0 * prior) / (n + n0)
    return df


# ── Bayesian model ────────────────────────────────────────────────────────────
def beta_reg_model(X, y=None, name=''):
    n_feat = X.shape[1]
    alpha  = pyro.sample(f'{name}_alpha', dist.Normal(0., 1.))
    beta   = pyro.sample(f'{name}_beta',
                         dist.Normal(torch.zeros(n_feat, device=device),
                                     torch.ones(n_feat, device=device)).to_event(1))
    phi    = pyro.sample(f'{name}_phi',  dist.LogNormal(2., 1.))
    mu     = torch.sigmoid(alpha + X @ beta)
    conc1  = (mu * phi).clamp(min=1e-4)
    conc0  = ((1. - mu) * phi).clamp(min=1e-4)
    with pyro.plate(f'{name}_plate', X.shape[0]):
        pyro.sample(f'{name}_obs', dist.Beta(conc1, conc0), obs=y)
    return mu


def run_svi(model_fn, guide, X_trn, y_trn, num_steps=SVI_STEPS, lr=0.01):
    svi    = SVI(model_fn, guide, Adam({'lr': lr}), loss=Trace_ELBO())
    for _ in range(num_steps):
        svi.step(X_trn, y_trn)


def posterior_mu_samples(model_fn, guide, X, name, n_samples=N_SAMPLES):
    guide_predictive = Predictive(
        guide, num_samples=n_samples,
        return_sites=[f'{name}_alpha', f'{name}_beta']
    )
    with torch.no_grad():
        samples = guide_predictive(X)
    alpha_s  = samples[f'{name}_alpha'].to(X.device)
    beta_s   = samples[f'{name}_beta'].to(X.device)
    mu_logit = alpha_s[:, None] + (beta_s @ X.T)
    return torch.sigmoid(mu_logit).cpu().numpy()


def X_only(df, scaler, feat_cols):
    return torch.tensor(scaler.transform(df[feat_cols].values), dtype=torch.float32).to(device)


def to_tensors(df, scaler, feat_cols, target_col):
    EPS = 1e-6
    X = torch.tensor(scaler.transform(df[feat_cols].values), dtype=torch.float32).to(device)
    y = torch.tensor(df[target_col].values, dtype=torch.float32).to(device)
    return X, y.clamp(EPS, 1 - EPS)


# ── Current-season stats (xwOBA, wOBA, HR count, K%, BB%) ─────────────────────
@st.cache_data(show_spinner=False)
def load_current_season_stats() -> pd.DataFrame:
    """Load 2026 statcast cache → compute player-level xwOBA, wOBA, HR, K%, BB%."""
    raw = _fetch_year(PRED_YEAR)
    if raw.empty:
        return pd.DataFrame()

    events = raw['events'].fillna('').astype(str)
    raw['_pa']      = events.isin(PA_EVENTS)
    raw['_k']       = events.isin(K_EVENTS)
    raw['_bb']      = events.isin(BB_EVENTS)
    raw['_hr']      = events == 'home_run'

    woba_val  = pd.to_numeric(raw.get('woba_value',                    pd.Series(np.nan, index=raw.index)), errors='coerce')
    woba_den  = pd.to_numeric(raw.get('woba_denom',                    pd.Series(np.nan, index=raw.index)), errors='coerce')
    xwoba_val = pd.to_numeric(raw.get('estimated_woba_using_speedangle', pd.Series(np.nan, index=raw.index)), errors='coerce')

    raw['_woba_val']  = woba_val
    raw['_woba_den']  = woba_den
    raw['_xwoba_val'] = xwoba_val

    g = raw.groupby('batter')

    pa_counts = g['_pa'].sum()
    k_counts  = g['_k'].sum()
    bb_counts = g['_bb'].sum()
    hr_counts = g['_hr'].sum()

    # wOBA = sum(woba_value) / sum(woba_denom) on PA rows
    pa_mask    = raw['_pa']
    woba_num   = raw.loc[pa_mask].groupby('batter')['_woba_val'].sum()
    woba_denom = raw.loc[pa_mask].groupby('batter')['_woba_den'].sum()
    xwoba_num  = raw.loc[pa_mask].groupby('batter')['_xwoba_val'].sum()

    stats = pd.DataFrame({
        'PA_actual': pa_counts,
        'HR':        hr_counts,
        'K':         k_counts,
        'BB':        bb_counts,
    })
    stats['woba_num']  = woba_num
    stats['woba_den']  = woba_denom
    stats['xwoba_num'] = xwoba_num
    stats['wOBA']  = (stats['woba_num']  / stats['woba_den'].replace(0, np.nan)).round(3)
    stats['xwOBA'] = (stats['xwoba_num'] / stats['woba_den'].replace(0, np.nan)).round(3)
    stats['K_pct'] = (stats['K'] / stats['PA_actual'].replace(0, np.nan)).round(3)
    stats['BB_pct'] = (stats['BB'] / stats['PA_actual'].replace(0, np.nan)).round(3)
    stats = stats.reset_index()

    id_map = _id_map()
    stats['name'] = stats['batter'].astype(int).map(id_map).fillna(stats['batter'].astype(str))

    return stats[['batter', 'name', 'PA_actual', 'HR', 'K_pct', 'BB_pct', 'wOBA', 'xwOBA']]


# ── Full model pipeline ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def run_model_pipeline():
    """
    Train the Bayesian models and return (results_df, nps_std_array).
    Cached across Streamlit reruns so training only happens once per session.
    If a full-cache parquet exists from a previous run, load it directly.
    """
    if os.path.exists(FULL_CACHE):
        df = pd.read_parquet(FULL_CACHE)
        nps_std = df['nps_std'].values
        return df, nps_std

    pyro.set_rng_seed(42)
    pyro.clear_param_store()

    train_df = build_dataset(TRAIN_YEARS, min_pa=MIN_PA_TRAIN)
    pred_df  = build_dataset([PRED_YEAR],  min_pa=MIN_PA_PRED)

    LEAGUE_MEANS  = {feat: train_df[feat].mean() for feat in FEAT_DENOM}
    PLAYER_PRIORS = compute_player_priors(train_df, decay=PRIOR_DECAY)
    pred_df_shrunk = shrink_pred_features(pred_df, LEAGUE_MEANS, PLAYER_PRIORS, n0=SHRINK_N0)

    scaler_k  = StandardScaler().fit(train_df[FEAT_K])
    scaler_bb = StandardScaler().fit(train_df[FEAT_BB])
    scaler_hr = StandardScaler().fit(train_df[FEAT_HR])

    EPS = 1e-6
    X_k_trn  = torch.tensor(scaler_k.transform(train_df[FEAT_K].values),   dtype=torch.float32).to(device)
    y_k_trn  = torch.tensor(train_df['k_per_pa'].values,  dtype=torch.float32).to(device).clamp(EPS, 1-EPS)
    X_bb_trn = torch.tensor(scaler_bb.transform(train_df[FEAT_BB].values),  dtype=torch.float32).to(device)
    y_bb_trn = torch.tensor(train_df['bb_per_pa'].values, dtype=torch.float32).to(device).clamp(EPS, 1-EPS)
    X_hr_trn = torch.tensor(scaler_hr.transform(train_df[FEAT_HR].values),  dtype=torch.float32).to(device)
    y_hr_trn = torch.tensor(train_df['hr_per_pa'].values, dtype=torch.float32).to(device).clamp(EPS, 1-EPS)

    X_k_pred  = X_only(pred_df_shrunk, scaler_k,  FEAT_K)
    X_bb_pred = X_only(pred_df_shrunk, scaler_bb, FEAT_BB)
    X_hr_pred = X_only(pred_df_shrunk, scaler_hr, FEAT_HR)

    model_k  = partial(beta_reg_model, name='k')
    model_bb = partial(beta_reg_model, name='bb')
    model_hr = partial(beta_reg_model, name='hr')
    guide_k  = autoguide.AutoNormal(model_k)
    guide_bb = autoguide.AutoNormal(model_bb)
    guide_hr = autoguide.AutoNormal(model_hr)

    run_svi(model_k,  guide_k,  X_k_trn,  y_k_trn)
    run_svi(model_bb, guide_bb, X_bb_trn, y_bb_trn)
    run_svi(model_hr, guide_hr, X_hr_trn, y_hr_trn)

    mu_k_pred  = posterior_mu_samples(model_k,  guide_k,  X_k_pred,  'k')
    mu_bb_pred = posterior_mu_samples(model_bb, guide_bb, X_bb_pred, 'bb')
    mu_hr_pred = posterior_mu_samples(model_hr, guide_hr, X_hr_pred, 'hr')

    nps_samples = mu_hr_pred * W_HR + mu_bb_pred * W_BB - mu_k_pred * W_K
    nps_mean    = nps_samples.mean(axis=0)
    nps_std     = nps_samples.std(axis=0)

    results = pred_df_shrunk[['batter', 'name', 'PA',
                               'whiff_pct', 'chase_pct', 'hard_hit_pct', 'sweet_spot_pct']].copy()
    results['pred_k_per_pa']  = mu_k_pred.mean(axis=0)
    results['pred_bb_per_pa'] = mu_bb_pred.mean(axis=0)
    results['pred_hr_per_pa'] = mu_hr_pred.mean(axis=0)
    results['nps_mean']       = nps_mean
    results['nps_std']        = nps_std
    results['nps_risk_adj']   = nps_mean - RISK_ALPHA * nps_std
    results = results.sort_values('nps_risk_adj', ascending=False).reset_index(drop=True)
    results.index += 1

    results.to_parquet(FULL_CACHE, index=True)
    return results, nps_std


# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title='Net Power Score 2026', layout='wide')
st.title('⚾ Net Power Score Dashboard — 2026')
st.caption(
    'Bayesian beta-regression NPS predictions (K%, BB%, HR rate) '
    'merged with current-season statcast stats.'
)

# Sidebar controls
with st.sidebar:
    st.header('Controls')
    risk_alpha = st.slider(
        'Risk Alpha (NPS risk adjustment)',
        min_value=0.0, max_value=1.5, value=RISK_ALPHA, step=0.05,
        help='0 = pure expected NPS, 0.5 = moderate risk penalty (default), 1.0 = conservative'
    )
    min_pa_filter = st.number_input('Min PA (current season)', min_value=0, max_value=300, value=20, step=5)
    st.markdown('---')
    if st.button('Clear model cache & retrain'):
        if os.path.exists(FULL_CACHE):
            os.remove(FULL_CACHE)
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

# Load data
with st.spinner('Loading model (first run trains Bayesian models — this takes ~1-2 min)...'):
    results_base, nps_std_arr = run_model_pipeline()

with st.spinner('Loading current-season stats...'):
    season_stats = load_current_season_stats()

# Re-apply risk alpha from slider (no retrain needed)
results = results_base.copy()
results['nps_risk_adj'] = results['nps_mean'] - risk_alpha * results['nps_std']
results = results.sort_values('nps_risk_adj', ascending=False).reset_index(drop=True)
results.index += 1

# Merge in current-season stats
merged = results.merge(
    season_stats[['batter', 'PA_actual', 'HR', 'K_pct', 'BB_pct', 'wOBA', 'xwOBA']],
    on='batter', how='left'
)

# Apply PA filter on actual current-season PA
if min_pa_filter > 0:
    merged = merged[merged['PA_actual'].fillna(0) >= min_pa_filter]

# Player multi-select
all_names = sorted(merged['name'].dropna().unique())
selected = st.multiselect(
    'Filter by player (leave blank to show all)',
    options=all_names,
    default=[],
    placeholder='Search for players...'
)
display_df = merged[merged['name'].isin(selected)] if selected else merged

# Column display config
display_cols = [
    'name', 'PA', 'PA_actual',
    'xwOBA', 'wOBA', 'HR',
    'K_pct', 'BB_pct',
    'whiff_pct', 'chase_pct', 'hard_hit_pct', 'sweet_spot_pct',
    'pred_k_per_pa', 'pred_bb_per_pa', 'pred_hr_per_pa',
    'nps_mean', 'nps_risk_adj',
]
display_df = display_df[[c for c in display_cols if c in display_df.columns]].copy()
display_df.index = range(1, len(display_df) + 1)

# Cast every numeric column to plain float so Streamlit can format them.
_all_num_cols = ['PA', 'PA_actual', 'HR',
                 'xwOBA', 'wOBA', 'K_pct', 'BB_pct',
                 'whiff_pct', 'chase_pct', 'hard_hit_pct', 'sweet_spot_pct',
                 'pred_k_per_pa', 'pred_bb_per_pa', 'pred_hr_per_pa',
                 'nps_mean', 'nps_risk_adj']
for col in _all_num_cols:
    if col in display_df.columns:
        display_df[col] = pd.to_numeric(display_df[col], errors='coerce').astype(float)

# Multiply rate columns by 100 so sprintf '%.1f%%' renders e.g. "23.4%"
_pct_cols = ['K_pct', 'BB_pct', 'whiff_pct', 'chase_pct', 'hard_hit_pct', 'sweet_spot_pct']
for col in _pct_cols:
    if col in display_df.columns:
        display_df[col] = display_df[col] * 100

col_cfg = {
    'name':           st.column_config.TextColumn('Player'),
    'PA':             st.column_config.NumberColumn('PA (model)',   format='%.0f'),
    'PA_actual':      st.column_config.NumberColumn('PA (season)',  format='%.0f'),
    'xwOBA':          st.column_config.NumberColumn('xwOBA',        format='%.3f'),
    'wOBA':           st.column_config.NumberColumn('wOBA',         format='%.3f'),
    'HR':             st.column_config.NumberColumn('HR',           format='%.0f'),
    'K_pct':          st.column_config.NumberColumn('K%',           format='%.1f%%'),
    'BB_pct':         st.column_config.NumberColumn('BB%',          format='%.1f%%'),
    'whiff_pct':      st.column_config.NumberColumn('Whiff%',       format='%.1f%%'),
    'chase_pct':      st.column_config.NumberColumn('Chase%',       format='%.1f%%'),
    'hard_hit_pct':   st.column_config.NumberColumn('HardHit%',     format='%.1f%%'),
    'sweet_spot_pct': st.column_config.NumberColumn('SweetSpot%',   format='%.1f%%'),
    'pred_k_per_pa':  st.column_config.NumberColumn('Pred K/PA',    format='%.3f'),
    'pred_bb_per_pa': st.column_config.NumberColumn('Pred BB/PA',   format='%.3f'),
    'pred_hr_per_pa': st.column_config.NumberColumn('Pred HR/PA',   format='%.3f'),
    'nps_mean':       st.column_config.NumberColumn('NPS Mean',     format='%.4f'),
    'nps_risk_adj':   st.column_config.NumberColumn('NPS Risk-Adj', format='%.4f'),
}

st.subheader(f'Rankings — {len(display_df)} players shown')
st.dataframe(display_df, column_config=col_cfg, use_container_width=True, height=600)

# Summary metrics for selected players
if selected and len(display_df) > 0:
    st.markdown('---')
    st.subheader('Selected Player Summary')
    cols = st.columns(min(len(display_df), 4))
    for i, (_, row) in enumerate(display_df.iterrows()):
        with cols[i % len(cols)]:
            st.metric(row['name'], f"NPS {row['nps_risk_adj']:.4f}")
            st.caption(
                f"xwOBA {row.get('xwOBA', 'N/A'):.3f}  |  wOBA {row.get('wOBA', 'N/A'):.3f}  |  "
                f"HR {int(row['HR']) if pd.notna(row.get('HR')) else 'N/A'}  |  "
                f"K% {row['K_pct']:.1f}%  |  BB% {row['BB_pct']:.1f}%"
                if pd.notna(row.get('xwOBA')) else ''
            )
