"""
tagger.py — Tag logic and tag probability distribution output.
"""

from __future__ import annotations

import numpy as np

from .config import CB_PENALTY, TAG_VALUE, THRESHOLDS


# ---------------------------------------------------------------------------
# Single-draw deterministic tag
# ---------------------------------------------------------------------------

def apply_tag(
    bb: float,
    k_pct: float,
    hr_fb: float,
    xwoba: float,
    gb: float,
) -> str:
    """
    Apply deterministic tag to a single draw.

    Priority order (top evaluated first — Cherry Bomb is a hard disqualifier):

    1. Cherry Bomb:  bb >= 0.09 OR kbb < 0.08
    2. Ace:          xwoba <= 0.290 AND kbb >= 0.15 AND bb < 0.09 AND hr_fb <= 0.12
    3. Volatile Ace: kbb >= 0.15 AND (xwoba > 0.290 OR hr_fb > 0.15)
    4. Workhorse:    xwoba <= 0.320 AND kbb >= 0.08 AND hr_fb <= 0.15
    5. Cherry Bomb:  fallthrough
    """
    kbb = k_pct - bb

    cb  = THRESHOLDS["cherry_bomb"]
    ace = THRESHOLDS["ace"]
    ap  = THRESHOLDS["ace_potential"]
    wh  = THRESHOLDS["workhorse"]

    # 1. Cherry Bomb — hard disqualifier checked first.
    # 1. Cherry Bomb — requires 2 of 3 conditions (any single condition is too noisy)
    cb_violations = sum([
        bb    >= cb["bb_min"],
        kbb   <  cb["kbb_max"],
        hr_fb >= cb["hrfb_min"],
    ])
    if cb_violations >= 2:
        return "Cherry Bomb"
    # if (
    #     (bb >= cb["bb_min"] or kbb <= cb["kbb_max"]) 
    #     and hr_fb >= cb['hrfb_min'] 
    # ):
    #     return "Cherry Bomb"

    # 2. Ace
    if (
        xwoba <= ace["xwoba_max"]
        and kbb  >= ace["kbb_min"]
        and bb   <=  ace["bb_max"]
        and hr_fb <= ace['hrfb_max']
    ):
        return "Ace"

    # 3. Volatile Ace — elite K-BB% but fails ace contact/control gates
    if (
        kbb >= ap["kbb_min"]
        and bb <= ap["bb_max"]
        and (hr_fb <= ap['hrfb_max'] or gb >= ap['gb_min'])

    ):
        return "Ace Potential"

    # 4. Workhorse
    if (
        xwoba <= wh["xwoba_max"]
        and kbb  >= wh["kbb_min"]
        and hr_fb <= wh["hrfb_max"]
        and bb <= wh['bb_max']
    ):
        return "Workhorse"

    # 5. Fallthrough — fails all positive gates
    return "Toby" #"Cherry Bomb"


# ---------------------------------------------------------------------------
# Vectorized tag distribution over simulation draws
# ---------------------------------------------------------------------------

def tag_distribution(draws_dict: dict[str, np.ndarray]) -> dict:
    """
    Apply apply_tag across all N_SIM draws and return probability distribution.

    Returns
    -------
    {
        "tag_probs": {tag_name: float},
        "modal_tag": str,
        "ev_score":  float,
    }
    """
    bb_arr    = draws_dict["bb_pct"]
    k_arr     = draws_dict["k_pct"]
    hr_fb_arr = draws_dict["hr_fb"]
    xwoba_arr = draws_dict["xwoba"]
    gb_arr    = draws_dict["gb_pct"]

    n = len(bb_arr)
    counts: dict[str, int] = {tag: 0 for tag in TAG_VALUE}

    for i in range(n):
        tag = apply_tag(
            bb=bb_arr[i],
            k_pct=k_arr[i],
            hr_fb=hr_fb_arr[i],
            xwoba=xwoba_arr[i],
            gb=gb_arr[i],
        )
        counts[tag] += 1

    tag_probs = {tag: counts[tag] / n for tag in TAG_VALUE}
    modal_tag = max(tag_probs, key=tag_probs.__getitem__)

    return {
        "tag_probs": tag_probs,
        "modal_tag": modal_tag,
        "ev_score":  expected_value(tag_probs),
    }


# ---------------------------------------------------------------------------
# Expected value
# ---------------------------------------------------------------------------

def expected_value(tag_probs: dict[str, float]) -> float:
    """
    Compute expected GS value from tag probability distribution.

    Formula:
        EV = sum(P(tag) * TAG_VALUE[tag]) - CB_PENALTY * P(Cherry Bomb)

    The CB_PENALTY term makes Cherry Bomb probability actively subtract from EV
    rather than contributing a small positive weight. Net CB contribution is
    TAG_VALUE["Cherry Bomb"] - CB_PENALTY = 0.1 - 0.30 = -0.20 per unit of
    probability, which meaningfully separates pitchers by start risk.
    """
    base_ev    = sum(tag_probs[tag] * TAG_VALUE[tag] for tag in TAG_VALUE)
    cb_penalty = CB_PENALTY * tag_probs.get("Cherry Bomb", 0.0)
    return base_ev - cb_penalty


# ---------------------------------------------------------------------------
# Confidence flag
# ---------------------------------------------------------------------------

def confidence_flag(mean_shrinkage: float) -> str:
    """
    Return confidence tier based on mean shrinkage across metrics.

    Thresholds are calibrated for the current prior strength regime (base 15–40,
    scaled to ~10–26 at week 4). With these strengths the achievable shrinkage
    range at 3–7 starts is roughly 0.73–0.88, not 0.0–1.0.

    > 0.83  → "LOW"   (≤ ~3 effective starts; prior-dominated; append † to tag)
    0.73–0.83 → "MED"  (~4–6 effective starts)
    < 0.73  → "HIGH"  (7+ effective starts; in-season data leading)
    """
    if mean_shrinkage > 0.83:
        return "LOW"
    elif mean_shrinkage >= 0.73:
        return "MED"
    else:
        return "HIGH"


# ---------------------------------------------------------------------------
# Metric summary
# ---------------------------------------------------------------------------

def metric_summary(draws: np.ndarray) -> dict[str, float]:
    """Return mean, p10, p90 for a metric's draw array."""
    return {
        "mean": float(np.mean(draws)),
        "p10":  float(np.percentile(draws, 10)),
        "p90":  float(np.percentile(draws, 90)),
    }
