import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from data_pull import pull_statcast_data
from prior_estimation import compute_weights, compute_player_stats, estimate_population_params
from posterior_update import compute_posteriors
from constants import DEFAULT_LAMBDA, LOOKBACK_YEARS


@st.cache_data
def load_data(years):
    return pull_statcast_data(years)


def compute_risk_adjusted_score(df, alpha):
    return df['posterior_mean'] / (df['posterior_sd'] ** alpha)


def plot_posterior_distributions(posteriors, selected_batters):
    selected_df = posteriors[posteriors['batter'].isin(selected_batters)]
    z = norm.ppf(0.95)
    x_min = (selected_df['posterior_mean'] - z * selected_df['posterior_sd']).min()
    x_max = (selected_df['posterior_mean'] + z * selected_df['posterior_sd']).max()
    x = np.linspace(x_min, x_max, 200)

    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in selected_df.iterrows():
        ax.plot(x, norm.pdf(x, loc=row['posterior_mean'], scale=row['posterior_sd']),
                label=row['player_name'])
    ax.set_xlabel('xwOBA')
    ax.set_ylabel('Density')
    ax.set_title('Posterior Distributions — Selected Players')
    ax.legend()
    st.pyplot(fig)


def plot_shrinkage_evolution(player_df, player_name, params, lam):
    mu_0 = params['mu_0']
    tau_sq = params['tau_sq']
    sigma_sq = params['sigma_sq']

    df = player_df.sort_values('game_date').reset_index(drop=True)
    df['weight'] = np.exp(-lam * df['days_ago'])

    milestones = range(10, len(df) + 1, 10)
    records = []

    for n in milestones:
        subset = df.iloc[:n]
        w = subset['weight']
        x = subset['estimated_woba_using_speedangle']
        n_eff = w.sum()
        xwoba_mean = np.average(x, weights=w)
        lambda_i = tau_sq / (tau_sq + sigma_sq / n_eff)
        posterior_mean = lambda_i * xwoba_mean + (1 - lambda_i) * mu_0
        posterior_var = 1 / (1 / tau_sq + n_eff / sigma_sq)
        posterior_sd = np.sqrt(posterior_var)

        records.append({
            'pa': n,
            'raw_xwoba': x.mean(),
            'posterior_mean': posterior_mean,
            'ci_low': posterior_mean - 1.645 * posterior_sd,
            'ci_high': posterior_mean + 1.645 * posterior_sd
        })

    evo = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(evo['pa'], evo['raw_xwoba'], label='Raw xwOBA', alpha=0.7)
    ax.plot(evo['pa'], evo['posterior_mean'], label='Posterior Mean')
    ax.fill_between(evo['pa'], evo['ci_low'], evo['ci_high'], alpha=0.3, label='90% CI')
    ax.axhline(y=mu_0, linestyle='--', color='gray', label='Population Mean')
    ax.set_xlabel('Plate Appearances')
    ax.set_ylabel('xwOBA')
    ax.set_title(f'Shrinkage Evolution — {player_name}')
    ax.legend()
    st.pyplot(fig)


def main():
    st.title("🔵 Hitter xwOBA Monitor")

    # --- Sidebar controls ---
    lam = st.sidebar.slider("Decay λ", 0.001, 0.05, DEFAULT_LAMBDA, step=0.001)
    alpha = st.sidebar.slider("Risk Alpha", 0.0, 1.0, 0.5, step=0.05)
    min_pa = st.sidebar.slider("Min Effective PA", 0, 100, 10, step=5)

    # refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    # --- Data pipeline ---
    raw = load_data(LOOKBACK_YEARS)
    weighted = compute_weights(raw, lam)
    player_stats = compute_player_stats(weighted)
    params = estimate_population_params(player_stats)
    posteriors = compute_posteriors(player_stats, params)
    posteriors['risk_adjusted_score'] = compute_risk_adjusted_score(posteriors, alpha)

    # --- PA / xwOBA / wOBA stats ---
    pa_stats = raw.groupby('batter').agg(
        PA=('estimated_woba_using_speedangle', 'count'),
        xwOBA=('estimated_woba_using_speedangle', 'mean'),
        wOBA=('woba_value', lambda x: x.dropna().mean())
    ).reset_index()
    posteriors = posteriors.merge(pa_stats, on='batter', how='left')

    # --- Player selector (unique by batter ID) ---
    posteriors['display_label'] = (
        posteriors['player_name'] + ' (' + posteriors['batter'].astype(str) + ')'
    )
    selected_labels = st.multiselect(
        "Select Players for Comparison",
        options=sorted(posteriors['display_label'].tolist())
    )
    selected_batters = [
        int(label.split('(')[-1].rstrip(')'))
        for label in selected_labels
    ]

    # --- Ranked table ---
    display_df = posteriors[posteriors['n_eff'] >= min_pa][[
        'batter', 'player_name', 'PA', 'xwOBA', 'wOBA',
        'posterior_mean', 'posterior_sd',
        'ci_low', 'ci_high', 'risk_adjusted_score'
    ]].sort_values('risk_adjusted_score', ascending=False)

    if selected_batters:
        display_df = display_df[display_df['batter'].isin(selected_batters)]

    st.dataframe(display_df.drop(columns=['batter']))

    # --- Plots ---
    if selected_batters:
        plot_posterior_distributions(posteriors, selected_batters)
        for batter_id in selected_batters:
            player_df = raw[raw['batter'] == batter_id]
            player_name = posteriors[posteriors['batter'] == batter_id]['player_name'].values[0]
            plot_shrinkage_evolution(player_df, player_name, params, lam)


if __name__ == "__main__":
    main()