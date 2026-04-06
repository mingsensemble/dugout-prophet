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


def plot_posterior_distributions(df, selected_players):
    selected_df = df[df['player_name'].isin(selected_players)]
    z = norm.ppf(0.95)
    x_min = (selected_df['posterior_mean'] - z * selected_df['posterior_sd']).min()
    x_max = (selected_df['posterior_mean'] + z * selected_df['posterior_sd']).max()
    x = np.linspace(x_min, x_max, 200)

    fig, ax = plt.subplots(figsize=(10, 6))
    for player in selected_players:
        row = selected_df[selected_df['player_name'] == player]
        if not row.empty:
            ax.plot(x, norm.pdf(x, loc=row['posterior_mean'].values[0],
                                scale=row['posterior_sd'].values[0]), label=player)
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
    # temporarily add this as the very first line of main():
    st.cache_data.clear()
    st.title("🔵 Hitter xwOBA Monitor")

    lam = st.sidebar.slider("Decay λ", 0.001, 0.05, DEFAULT_LAMBDA, step=0.001)
    alpha = st.sidebar.slider("Risk Alpha", 0.0, 1.0, 0.5, step=0.05)
    min_pa = st.sidebar.slider("Min Effective PA", 0, 100, 10, step=5)

    raw = load_data(LOOKBACK_YEARS)
    # st.write(f"weight sample: {raw['weight'].describe() if 'weight' in raw.columns else 'weight column NOT in raw'}")
    # st.write(f"raw rows: {len(raw)}")
    # st.write(f"days_ago range: {raw['days_ago'].min()} to {raw['days_ago'].max()}")
    # st.write(f"days_ago range: {raw['days_ago'].min()} to {raw['days_ago'].max()}")
    # st.write(f"LOOKBACK_YEARS being used: {LOOKBACK_YEARS}")

    weighted = compute_weights(raw, lam)
    # st.write(f"weighted rows: {len(weighted)}")
    # st.write(f"columns in weighted: {weighted.columns.tolist()}")
    # st.write(f"weight stats: {weighted['weight'].describe()}")
    # st.write(f"unique batters in weighted: {weighted['batter'].nunique()}")
    # st.write(f"unique player_names in weighted: {weighted['player_name'].nunique()}")
    # st.write(f"sample batter values: {weighted['batter'].head(10).tolist()}")
    # st.write(f"sample player_name values: {weighted['player_name'].head(10).tolist()}")
    # st.write(f"unique batters: {weighted['batter'].nunique()}")
    # st.write(f"unique player_names: {weighted['player_name'].nunique()}")
    # st.write(f"player_name nulls: {weighted['player_name'].isnull().sum()}")
    # st.write(f"sample player_names: {weighted['player_name'].head(10).tolist()}")

    player_stats = compute_player_stats(weighted)
    # st.write(f"player_stats rows: {len(player_stats)}")
    # st.write(player_stats.head())
    # st.write(f"n_eff stats: {player_stats['n_eff'].describe()}")
    # st.write(f"players with n_eff >= 50: {(player_stats['n_eff'] >= 50).sum()}")

    params = estimate_population_params(player_stats)
    # st.write(f"params: {params}")


    posteriors = compute_posteriors(player_stats, params)
    # st.write(f"posteriors rows: {len(posteriors)}")
    # st.write(posteriors.head())

    posteriors['risk_adjusted_score'] = compute_risk_adjusted_score(posteriors, alpha)

    selected_players = st.multiselect(
        "Select players for comparison",
        options=sorted(posteriors['player_name'].tolist())
    )
    # compute PA count and actual wOBA per player
    pa_stats = raw.groupby('batter').agg(
        PA=('estimated_woba_using_speedangle', 'count'),
        xwOBA=('estimated_woba_using_speedangle', 'mean'),
        wOBA=('woba_value', 'mean')
    ).reset_index()
    posteriors = posteriors.merge(pa_stats, on='batter', how='left')

    display_df = posteriors[posteriors['n_eff'] >= min_pa][
        ['player_name', 'PA', 'xwOBA', 'wOBA', 'n_eff', 'xwoba_weighted_mean',
         'posterior_mean', 'posterior_sd', 'ci_low', 'ci_high', 'risk_adjusted_score']
    ].sort_values('risk_adjusted_score', ascending=False)

    st.dataframe(display_df)

    if selected_players:
        plot_posterior_distributions(posteriors, selected_players)
        for player in selected_players:
            player_df = raw[raw['player_name'] == player]
            plot_shrinkage_evolution(player_df, player, params, lam)


if __name__ == "__main__":
    main()