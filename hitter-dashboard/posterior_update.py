import numpy as np
import pandas as pd
from scipy.stats import norm
from constants import MIN_N_EFF_POSTERIOR


def compute_posteriors(player_stats: pd.DataFrame, params: dict) -> pd.DataFrame:
    mu_0 = params['mu_0']
    tau_sq = params['tau_sq']
    sigma_sq = params['sigma_sq']

    df = player_stats[player_stats['n_eff'] >= MIN_N_EFF_POSTERIOR].copy()

    df['lambda_i'] = tau_sq / (tau_sq + sigma_sq / df['n_eff'])
    df['posterior_mean'] = df['lambda_i'] * df['xwoba_weighted_mean'] + (1 - df['lambda_i']) * mu_0
    df['posterior_var'] = 1 / (1 / tau_sq + df['n_eff'] / sigma_sq)
    df['posterior_sd'] = np.sqrt(df['posterior_var'])
    df['ci_low'] = norm.ppf(0.05, loc=df['posterior_mean'], scale=df['posterior_sd'])
    df['ci_high'] = norm.ppf(0.95, loc=df['posterior_mean'], scale=df['posterior_sd'])

    return df[['batter', 'player_name', 'n_eff', 'xwoba_weighted_mean',
               'posterior_mean', 'posterior_sd', 'ci_low', 'ci_high']]