"""
model.py — Exponential decay weighting + empirical Bayes posteriors + simulation.
"""

from __future__ import annotations

import numpy as np

from .config import N_SIM, PRIORS


# ---------------------------------------------------------------------------
# Prior scaling
# ---------------------------------------------------------------------------

def effective_prior_strength(base_strength: float, current_week: int) -> float:
    """
    Scale prior strength by season progress.

    Week 1:  50% of base — weak prior, let early data speak.
    Week 13+: 100% of base — full prior, stabilize late-season noise.

    Linear ramp: scale = min(1.0, 0.5 + current_week / 26)
    """
    scale = min(1.0, 0.5 + (current_week / 26))
    return base_strength * scale


# ---------------------------------------------------------------------------
# Decay
# ---------------------------------------------------------------------------

def decay_weights(weeks: list[int], current_week: int, lambda_: float) -> np.ndarray:
    """
    Compute exponential decay weights.

    w_t = exp(-lambda_ * (current_week - t))

    More recent starts receive higher weight. All weights are positive.
    """
    w = np.exp(-lambda_ * (current_week - np.array(weeks, dtype=float)))
    return w


def compute_effective_n(weights: np.ndarray) -> float:
    """
    Precision-weighted effective sample size.

        n_eff = (sum(w))^2 / sum(w^2)

    Always <= raw N. Propagates the information cost of decay into EB shrinkage.
    """
    sum_w  = weights.sum()
    sum_w2 = (weights ** 2).sum()
    if sum_w2 == 0:
        return 0.0
    return float(sum_w ** 2 / sum_w2)


# ---------------------------------------------------------------------------
# Posteriors
# ---------------------------------------------------------------------------

def compute_posterior(
    observations: np.ndarray,
    weeks: list[int],
    current_week: int,
    prior: dict,
) -> tuple[float, float, float, float]:
    """
    Beta-conjugate EB posterior for rate metrics (BB%, K%, HR/FB, GB%).

    Returns (alpha_post, beta_post, n_eff, shrinkage).

    Starts with fewer batters faced or NaN rate values are skipped via valid_mask.
    shrinkage = strength / (strength + n_eff): 1.0 = full prior, 0.0 = full data.
    Empty observations returns the prior pseudo-counts with n_eff=0, shrinkage=1.0.
    """
    obs_arr    = np.asarray(observations, dtype=float)
    week_arr   = np.asarray(weeks, dtype=int)
    valid_mask = ~np.isnan(obs_arr)

    obs_clean   = obs_arr[valid_mask]
    weeks_clean = week_arr[valid_mask].tolist()

    strength = effective_prior_strength(prior["strength"], current_week)
    mean     = prior["mean"]

    alpha_0 = mean * strength
    beta_0  = (1.0 - mean) * strength

    if len(obs_clean) == 0:
        return alpha_0, beta_0, 0.0, 1.0

    w         = decay_weights(weeks_clean, current_week, prior["lambda_"])
    n_eff     = compute_effective_n(w)
    shrinkage = strength / (strength + n_eff)
    mu_obs    = float(np.average(obs_clean, weights=w))

    x_eff      = mu_obs * n_eff
    alpha_post = alpha_0 + x_eff
    beta_post  = beta_0  + (n_eff - x_eff)

    return float(alpha_post), float(beta_post), float(n_eff), float(shrinkage)


def normal_posterior_from_starts(
    xwoba_means: np.ndarray,
    xwoba_ns: np.ndarray,
    weeks: list[int],
    current_week: int,
    prior: dict,
) -> tuple[float, float, float, float]:
    """
    Normal-conjugate EB posterior for xwOBA using per-start PA counts as denominator.

    Parameters
    ----------
    xwoba_means : per-start xwOBA means (NaN for starts with no batted-ball events)
    xwoba_ns    : per-start count of batted-ball events with a Statcast xwOBA value
    weeks       : week number for each start (same length as xwoba_means)
    current_week: current season week
    prior       : xwOBA prior dict; must contain "mean", "sigma", "strength", "lambda_"

    Returns (mu_post, sigma_post, n_eff_starts, shrinkage).

    Precision model (obs_sigma_pa cancels from shrinkage and mu_post):
        denominator = sum(w * xwoba_n)        [PA-weighted effective count]
        shrinkage   = strength / (strength + denominator)
        mu_post     = (strength * mean + denominator * mu_obs) / (strength + denominator)
        sigma_post  = sigma / sqrt(1 + denominator / strength)

    n_eff_starts is the decay-weighted start count (returned for display/debug, not
    used in the normal posterior math — PA count drives precision here).
    """
    means_arr = np.asarray(xwoba_means, dtype=float)
    ns_arr    = np.asarray(xwoba_ns,    dtype=float)
    week_arr  = np.asarray(weeks,       dtype=int)

    # Exclude starts with no batted-ball xwOBA values.
    valid_mask  = ~np.isnan(means_arr)
    means_clean = means_arr[valid_mask]
    ns_clean    = ns_arr[valid_mask]
    weeks_clean = week_arr[valid_mask].tolist()

    strength = effective_prior_strength(prior["strength"], current_week)
    mean     = prior["mean"]
    sigma    = prior["sigma"]

    if len(means_clean) == 0:
        return mean, sigma, 0.0, 1.0

    w     = decay_weights(weeks_clean, current_week, prior["lambda_"])
    n_eff = compute_effective_n(w)   # starts-equivalent; returned for display

    wn          = w * ns_clean       # decay × PA count per start
    denominator = float(wn.sum())
    mu_obs      = float(np.dot(wn, means_clean) / denominator) if denominator > 0 else mean

    shrinkage  = strength / (strength + denominator)
    mu_post    = (strength * mean + denominator * mu_obs) / (strength + denominator)
    sigma_post = sigma / np.sqrt(1.0 + denominator / strength)

    return float(mu_post), float(sigma_post), float(n_eff), float(shrinkage)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_posteriors(
    posteriors: dict[str, tuple[str, float, float]],
    n_sim: int = N_SIM,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """
    Draw n_sim samples from each metric's posterior distribution.

    posteriors: {metric: (dist_type, param1, param2)}

    Returns {metric: np.ndarray shape (n_sim,)} plus derived "kbb" key.

    K-BB% is always derived as k_draws - bb_draws to preserve their
    mechanical dependency — never sampled independently.
    """
    rng = np.random.default_rng(seed)
    draws: dict[str, np.ndarray] = {}

    for metric, (dist_type, p1, p2) in posteriors.items():
        if dist_type == "beta":
            samples = rng.beta(p1, p2, size=n_sim)
            # Clip to avoid boundary NaN propagation.
            samples = np.clip(samples, 0.001, 0.999)
        else:  # normal
            samples = rng.normal(p1, p2, size=n_sim)
        draws[metric] = samples

    # K-BB% is derived, not independently sampled.
    if "k_pct" in draws and "bb_pct" in draws:
        draws["kbb"] = draws["k_pct"] - draws["bb_pct"]

    return draws
