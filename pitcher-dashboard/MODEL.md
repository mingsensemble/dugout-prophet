# Pitcher True Performance Model — Training Summary

## Overview

Three-stage pipeline that produces a **luck-adjusted, recency-weighted, uncertainty-quantified** pitcher ranking from raw Statcast data.

| Stage | Name | Purpose |
|-------|------|---------|
| 1 | PitchValueNet | Predict pitch-level expected run value (xRV), removing sequencing luck |
| 2 | Exponential Decay Aggregation | Weight recent pitches more heavily for weekly fantasy decisions |
| 3 | Empirical Bayes Shrinkage | Shrink small-sample pitchers toward the league mean |

---

## Stage 1: PitchValueNet

### Architecture

A shallow MLP with separate embedding tables for pitch type and previous pitch type.

```
[pitch_type_enc]      → Embedding(n_pitch_types, 4) ─┐
[prev_pitch_type_enc] → Embedding(n_pitch_types, 4) ─┤→ concat → Linear(hidden=32) → ReLU → Linear(1)
[continuous features] ─────────────────────────────────┘
```

Separate embedding tables capture positional role: a fastball thrown *as* the current pitch carries different deception/sequencing signal than a fastball thrown *before* it.

### Features

**Continuous (RobustScaler-normalized):**

| Group | Columns |
|-------|---------|
| Stuff | `release_speed`, `pfx_x`, `pfx_z`, `release_spin_rate`, `release_extension` |
| Command | `plate_x`, `plate_z`, `zone` |
| Context | `pitch_number`, `xwOBA` (batter seasonal, FanGraphs, league-avg imputed) |

**Categorical (label-encoded → embeddings):**
- `pitch_type_enc` — current pitch type
- `prev_pitch_type_enc` — previous pitch type in the PA (`START` for first pitch)

### Target

`delta_run_exp` — pitch-level change in run expectancy (Statcast). Lower = better for the pitcher.

### Training Data

- **Source:** Full 2023 Statcast season via pybaseball
- **Train/test split:** April–July (train) | August–September (test) — time-series split, no leakage
- **Preprocessing:**
  - `prev_pitch_type` lag feature built within `(game_pk, at_bat_number)` groups
  - Rows with missing Statcast physics dropped (MCAR, <1%)
  - RobustScaler fit on training data only, applied to test and scoring data

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `emb_dim` | 4 | Embedding size for pitch type tables |
| `hidden_size` | 32 | Single hidden layer; depth didn't help (grid search) |
| `dropout` | 0.0 | No dropout needed at this scale |
| `lr` | 1e-3 | Adam optimizer |
| `batch_size` | 512 | |
| `epochs` | 10 | |
| Loss | MSELoss | Unbounded continuous target |

### SP/RP Classification

Pitchers are classified using pitches-per-appearance:
- **< 3 appearances:** use `max` pitches in any game (avoids median instability early in season)
- **≥ 3 appearances:** use `median` pitches per game
- **Threshold:** ≥ 50 pitches → SP; < 50 → RP

---

## Stage 2: Exponential Decay Aggregation

Weighted mean xRV as of a given date:

```
w_t = exp(−λ · days_ago)
weighted_xrv = Σ(w_t · xRV_t) / Σ(w_t)
```

- **λ = 0.046** → 14-day half-life (tuned for weekly fantasy roster decisions)
- **Eligibility:** ≥ 100 pitches in the scoring window
- **Sort:** ascending — lower weighted_xrv = better pitcher

---

## Stage 3: Empirical Bayes Shrinkage

Normal-Normal conjugate model applied at the appearance level.

**Model:**
```
μᵢ  ~ N(μ_league, σ_league²)    — pitcher true skill prior
xRV_j ~ N(μᵢ, σ_noise²)         — appearance-level likelihood
```

**Closed-form posterior:**
```
posterior_var_i  = 1 / (1/σ_league² + nᵢ/σ_noise²)
posterior_mean_i = posterior_var_i × (μ_league/σ_league² + nᵢ × mean_xrv_i/σ_noise²)
posterior_std_i  = √posterior_var_i
```

**Hyperparameters estimated from data (empirical):**
- `μ_league` — mean of per-pitcher appearance-level means
- `σ_league` — std of per-pitcher means (between-pitcher spread)
- `σ_noise`  — std of all appearance-level means (within-pitcher noise)

**Effect:**
- Low nᵢ (cup-of-coffee pitchers) → heavy shrinkage toward league average
- High nᵢ → posterior mean stays close to observed mean
- `posterior_std` provides a calibrated uncertainty estimate

**Why Empirical Bayes over full hierarchical VI:** AutoNormal mean-field fails here — the likelihood overwhelms the prior with 5000+ observations and ~250 pitchers, causing `σ_league` to collapse regardless of the prior. Empirical Bayes is closed-form, instantaneous, and produces correct shrinkage.

---

## Validation

Spearman rank correlation of `weighted_xrv` vs FanGraphs end-of-season metrics:

| Metric | Expected direction |
|--------|--------------------|
| Stuff+ | Negative r (better stuff → lower xRV) |
| SIERA  | Positive r (worse ERA estimator → higher xRV) |

---

## Artifacts

| File | Contents |
|------|----------|
| `artifacts/model.pt` | PitchValueNet state dict |
| `artifacts/scaler.pkl` | Fitted RobustScaler |
| `artifacts/le.pkl` | Fitted LabelEncoder (pitch types) |
| `data/cache_{season}.parquet` | Scored pitches: `pitcher, game_pk, game_date, role, pred_xrv` |
| `data/fg_batters_{season}.parquet` | FanGraphs batter xwOBA cache |
