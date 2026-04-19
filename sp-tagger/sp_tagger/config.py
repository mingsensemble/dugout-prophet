"""
config.py — Roster, priors, thresholds, and season parameters for SP Tagger.
"""

# ---------------------------------------------------------------------------
# Roster
# MLBAM IDs are hardcoded — do not use playerid_lookup at runtime (name
# matching fails on accented characters).
# ---------------------------------------------------------------------------

ROSTER = {
    "Rasmussen": {
        "mlbam_id": 656876,
        'team': "TBR",
        "roster_type": "on_roster",
        "prior_overrides": {
            "bb_pct": {"mean": 0.063, "strength": 20},  
            "k_pct":  {"mean": 0.234, "strength": 20},  
            "hr_fb":  {"mean": 0.100, "strength": 15},  
            "gb_pct": {"mean": 0.486, "strength": 30},  
        },
    },
    "Glasnow": {
        "mlbam_id": 607192,
        'team': 'LAD',
        "roster_type": "on_roster",
        "prior_overrides": {
            "bb_pct": {"mean": 0.093, "strength": 20},
            "k_pct":  {"mean": 0.309, "strength": 20},  
            "hr_fb":  {"mean": 0.149, "strength": 15},  
            "gb_pct": {"mean": 0.469, "strength": 30},  
        }
    },
    "Gray": {
        "mlbam_id": 543243,
        "team": "BOS",
        "roster_type": "on_roster",
        "prior_overrides": {
            "bb_pct": {"mean": 0.078, "strength": 20},  
            "k_pct":  {"mean": 0.241, "strength": 20},  
            "hr_fb":  {"mean": 0.121, "strength": 15},  
            "gb_pct": {"mean": 0.498, "strength": 30}, 
        }
    },
    "McClanahan": {
        "mlbam_id": 663372,
        "team": "TBR",
        "roster_type": "on_roster",
        # Pre-TJS 2022–23 baseline is more informative than a missing 2025 season.
        "prior_overrides": {
            "bb_pct": {"mean": 0.075, "strength": 15},  
            "k_pct":  {"mean": 0.278, "strength": 15},  
            "hr_fb":  {"mean": 0.142, "strength": 15},  
            "gb_pct": {"mean": 0.466, "strength": 15},  
        }
    },
    "Bradish": {
        "mlbam_id": 680694,
        "team": "BAL",
        "roster_type": "on_roster",
        "prior_overrides": {
            "bb_pct": {"mean": 0.081, "strength": 15},  
            "k_pct":  {"mean": 0.259, "strength": 15},  
            "hr_fb":  {"mean": 0.116, "strength": 15},  
            "gb_pct": {"mean": 0.474, "strength": 15}, 
        }
    },

    "Eovaldi": {
        "mlbam_id": 543135,
        "team": "TEX",
        "roster_type": "on_roster",
        "prior_overrides": {
            "bb_pct": {"mean": 0.066, "strength": 20},  
            "k_pct":  {"mean": 0.210, "strength": 20}, 
            "hr_fb":  {"mean": 0.115, "strength": 15},  
            "gb_pct": {"mean": 0.470, "strength": 30},  
        }        
    },
    "Perez": {
        "mlbam_id": 691587,
        "team": "MIA",
        "roster_type": "on_roster",
        "prior_overrides": {
            "bb_pct": {"mean": 0.087, "strength": 15},  
            "k_pct":  {"mean": 0.274, "strength": 15},  
            "hr_fb":  {"mean": 0.111, "strength": 15},  
            "gb_pct": {"mean": 0.289, "strength": 15}, 
        }     
   },
# monitoring
    "Early": {
        "mlbam_id": 813349,
        "team": "BOS",
        "roster_type": "monitoring",
    },
    "Nelson": {
        "mlbam_id": 669194,
        "team": "AZ",
        "roster_type": "monitoring",
        "prior_overrides": {
            "bb_pct": {"mean": 0.067, "strength": 20},  
            "k_pct":  {"mean": 0.192, "strength": 20},  
            "hr_fb":  {"mean": 0.105, "strength": 15},  
            "gb_pct": {"mean": 0.386, "strength": 30}, 
        }  
    },
    "Abel": {
        "mlbam_id": 690953,
        "team": "MIN",
        "roster_type": "monitoring",
    },
    "Detmers": {
        "mlbam_id": 672282,
        "team": "LAA",
        "roster_type": "monitoring",
    },
    "Roupp": {
        "mlbam_id": 694738,
        "team": "SFG",
        "roster_type": "monitoring",
    },    

# benchmark players for calibration and comparison
    # "Skubal": {
    #     "mlbam_id": 669373,
    #     "team": "DET",
    #     "prior_overrides": {
    #         "bb_pct": {"mean": 0.056, "strength": 20},  
    #         "k_pct":  {"mean": 0.289, "strength": 20},  
    #         "hr_fb":  {"mean": 0.120, "strength": 15},  
    #         "gb_pct": {"mean": 0.428, "strength": 30}, 
    #     }
    # },
    # "Alcantara": {
    #     "mlbam_id": 645261,
    #     "team": "PHI",
    #     "prior_overrides": {
    #         "bb_pct": {"mean": 0.074, "strength": 20},  
    #         "k_pct":  {"mean": 0.210, "strength": 20},  
    #         "hr_fb":  {"mean": 0.115, "strength": 15},  
    #         "gb_pct": {"mean": 0.497, "strength": 20}, 
    #     }
    # },
    # typical cherry bomb profile — high walk rate, low k%, high hr/fb, low gb%
    # "Luzardo": {
    #     "mlbam_id": 666200,
    #     "team": "PHI",
    #     "prior_overrides": {
    #         "bb_pct": {"mean": 0.081, "strength": 20},  
    #         "k_pct":  {"mean": 0.268, "strength": 20},  
    #         "hr_fb":  {"mean": 0.125, "strength": 15},  
    #         "gb_pct": {"mean": 0.410, "strength": 30}, 
    #     }
    # },
    # "Pivetta": {
    #     "mlbam_id": 601713,
    #     "team": "SDP",
    #     "prior_overrides": {
    #         "bb_pct": {"mean": 0.084, "strength": 20},  
    #         "k_pct":  {"mean": 0.263, "strength": 20},  
    #         "hr_fb":  {"mean": 0.143, "strength": 15},  
    #         "gb_pct": {"mean": 0.388, "strength": 20}, 
    #     }
    # },
}
# updated using career data
# for pitchers with fewer season, the strength is 15; otherwise 20
# hr/fb is always hard to predict; hence strength  = 15 for all

# ---------------------------------------------------------------------------
# Prior Configs
# dist: "beta" or "normal"
# beta metrics: no obs noise field — posterior uses sufficient stats directly.
# normal (xwoba): sigma = prior std dev. Per-PA observation noise is derived
#   dynamically as obs_sigma_pa = sigma * sqrt(strength_scaled), so obs_sigma
#   cancels in the shrinkage formula (= strength / (strength + sum(w*n_pa))).
# lambda_: exponential decay rate — higher = more recency weight
# strength: effective prior N — controls shrinkage aggressiveness
# ---------------------------------------------------------------------------
PRIORS: dict[str, dict] = {
    "bb_pct":  dict(mean=0.084, strength=20,  lambda_=0.30, dist="beta"),
    "k_pct":   dict(mean=0.225, strength=25,  lambda_=0.20, dist="beta"),
    "hr_fb":   dict(mean=0.118, strength=15,  lambda_=0.30, dist="beta"),
    "gb_pct":  dict(mean=0.418, strength=40,  lambda_=0.10, dist="beta"),
    "xwoba":   dict(mean=0.310, strength=30,  lambda_=0.15, dist="normal",
                    sigma=0.040),  # prior std dev; obs noise derived from PA counts
}
# updated league priors

# ---------------------------------------------------------------------------
# Tag Thresholds
# ---------------------------------------------------------------------------
# original setting
# THRESHOLDS = {
#     "cherry_bomb":  dict(bb_min=0.09,  kbb_max=0.08),
#     "ace":          dict(xwoba_max=0.290, kbb_min=0.15, bb_max=0.09, hrfb_max=0.12),
#     "volatile_ace": dict(kbb_min=0.15),   # triggered when ace gates fail
#     "workhorse":    dict(xwoba_max=0.320, kbb_min=0.08, hrfb_max=0.15),
# }


THRESHOLDS = {
    "cherry_bomb":  dict(bb_min=0.098,  kbb_max=0.090, hrfb_min = 0.155),  
    # "cherry_bomb":  dict(bb_min=0.077,  kbb_max=0.121, hrfb_min = 0.108),  
    # more lenient than original to reflect 2025 league-wide environment — higher walk rate, higher hr/fb, and lower gb% across the board
    "ace":          dict(xwoba_max=0.306, kbb_min=0.143, bb_max=0.077, hrfb_max = 0.115),
    # top 30 in all categories
    # ace: hard to hit, k-bb rewarding, don't walk much; hr/fb is noisy and less predictive, so not included in ace gates
    "ace_potential": dict(kbb_min=0.143, bb_max=0.077, hrfb_max = 0.115, gb_min = 0.413),   
    # triggered when ace gates fail
    # ace potential: ace but still elite k-bb
    # top 30 k-bb and top 30 bb% and (top 30 hr fb or top 30 gb)
    # "workhorse":    dict(
    #     xwoba_max=0.339, 
    #     kbb_min=0.079, 
    #     bb_max = 0.098,
    #     hrfb_max=0.156,
    # ), # top 50 SP in these categories in 2025

    "workhorse": dict(
        xwoba_max=0.350,    # was 0.339 — 60th percentile xwOBA against
        kbb_min=0.065,      # was 0.079 — 60th percentile K-BB%
        bb_max=0.105,       # was 0.098 — 60th percentile BB%
        hrfb_max=0.175,     # was 0.156 — 60th percentile HR/FB
    )
}

TAG_VALUE: dict[str, float] = {
    "Ace": 1.0,
    "Ace Potential": 0.9,
    "Workhorse": 0.5,
    "Toby": 0.3,
    "Cherry Bomb": 0.1,
}

# Additional penalty subtracted from EV for each unit of Cherry Bomb probability.
# Net CB contribution = TAG_VALUE["Cherry Bomb"] - CB_PENALTY = 0.1 - 0.30 = -0.20.
# Keeps CB out of TAG_VALUE so the table still displays positive tag weights,
# while making EV actively penalise high CB probability.
CB_PENALTY: float = 0.30

# ---------------------------------------------------------------------------
# Season Config
# ---------------------------------------------------------------------------
SEASON       = 2026
SEASON_START = "2026-03-25"
N_SIM        = 10000
MIN_BF       = 6       # exclude starts with fewer batters faced (injury exits)
