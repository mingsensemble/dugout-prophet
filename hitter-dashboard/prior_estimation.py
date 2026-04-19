import numpy as np
import pandas as pd
from constants import MIN_N_EFF_PRIOR, MIN_N_EFF_POSTERIOR


def compute_weights(df: pd.DataFrame, lam: float) -> pd.DataFrame:
    df = df.copy()
    df['weight'] = np.exp(-lam * df['days_ago'])
    return df


def compute_player_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['wx'] = df['weight'] * df['estimated_woba_using_speedangle']
    
    agg = df.groupby(['batter', 'player_name']).agg(
        sum_w=('weight', 'sum'),
        sum_wx=('wx', 'sum')
    ).reset_index()
    
    agg['xwoba_weighted_mean'] = agg['sum_wx'] / agg['sum_w']
    agg['n_eff'] = agg['sum_w']
    
    # weighted variance requires a second pass
    df = df.merge(agg[['batter', 'xwoba_weighted_mean']], on='batter')
    df['wresid2'] = df['weight'] * (df['estimated_woba_using_speedangle'] - df['xwoba_weighted_mean'])**2
    
    var_agg = df.groupby(['batter', 'player_name']).agg(
        sum_wresid2=('wresid2', 'sum')
    ).reset_index()
    
    result = agg.merge(var_agg, on=['batter', 'player_name'])
    result['xwoba_weighted_var'] = result['sum_wresid2'] / result['sum_w']
    
    return result[['batter', 'player_name', 'n_eff', 'xwoba_weighted_mean', 'xwoba_weighted_var']]


def estimate_population_params(player_stats: pd.DataFrame) -> dict:
    est = player_stats[player_stats['n_eff'] >= MIN_N_EFF_PRIOR].copy()

    mu_0 = np.average(est['xwoba_weighted_mean'], weights=est['n_eff'])
    sigma_sq = np.average(est['xwoba_weighted_var'], weights=est['n_eff'])

    raw_var = np.average(
        (est['xwoba_weighted_mean'] - mu_0) ** 2,
        weights=est['n_eff']
    )
    # correction = (sigma_sq / est['n_eff']).mean()
    # correction = sigma_sq * len(est) / est['n_eff'].sum()
    # correction = sigma_sq / est['n_eff']


    # tau_sq = max(raw_var - correction, 1e-6)
    tau_sq = max(raw_var, 1e-4)

    # print(f"raw_var: {raw_var:.6f}")
    # print(f"sigma_sq: {sigma_sq:.6f}")
    # print(f"tau_sq: {tau_sq:.6f}")
    # # print(f"correction: {correction:.6f}") 
    # print(f"est n_eff mean: {est['n_eff'].mean():.2f}")
    # print(f"est rows: {len(est)}")

    return {'mu_0': mu_0, 'tau_sq': tau_sq, 'sigma_sq': sigma_sq}