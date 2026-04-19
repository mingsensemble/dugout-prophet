"""
ingest.py — Pybaseball pulls and per-start aggregation.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pybaseball

from .config import MIN_BF, SEASON, SEASON_START

# Enable pybaseball cache at import time to avoid redundant Savant hits.
pybaseball.cache.enable()

# Savant column for xwOBA — verify this hasn't changed after schema updates.
XWOBA_COL = "estimated_woba_using_speedangle"

DATA_DIR = Path(__file__).parent / "data" / "starts"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# All columns that must be present for the cache to be considered valid.
# If a cached CSV is missing any of these (e.g. was written before xwoba_n was added),
# load_or_fetch_starts will discard it and re-fetch from season start.
REQUIRED_CACHE_COLUMNS = {
    "game_date", "week", "bf",
    "k_pct", "bb_pct", "gb_pct", "hr_fb",
    "xwoba", "xwoba_n",
}


def _assert_xwoba_col(df: pd.DataFrame) -> None:
    assert XWOBA_COL in df.columns, (
        f"Expected Savant column '{XWOBA_COL}' not found. "
        f"Available columns: {list(df.columns)}"
    )


def fetch_pitcher_statcast(
    mlbam_id: int, start_date: str, end_date: str
) -> pd.DataFrame:
    """Pull pitch-level Statcast data for a single pitcher."""
    from pybaseball import statcast_pitcher
    df = statcast_pitcher(start_date, end_date, player_id=mlbam_id)
    if df is not None and not df.empty:
        _assert_xwoba_col(df)
    return df if df is not None else pd.DataFrame()


def _aggregate_starts(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pitch-level rows → per-start metrics grouped by game_pk."""
    season_start = pd.Timestamp(SEASON_START)

    def _agg_game(grp: pd.DataFrame) -> pd.Series:
        game_date = pd.Timestamp(grp["game_date"].iloc[0])
        week = math.floor((game_date - season_start).days / 7) + 1

        bf = grp["at_bat_number"].nunique()

        events = grp["events"].dropna()
        k_count  = (events == "strikeout").sum()
        bb_count = (events == "walk").sum()
        hr_count = (events == "home_run").sum()

        bb_types = grp["bb_type"].dropna()
        gb_count = (bb_types == "ground_ball").sum()
        fb_count = (bb_types == "fly_ball").sum()
        total_bbe = bb_types.shape[0]

        k_pct  = k_count / bf  if bf > 0 else np.nan
        bb_pct = bb_count / bf if bf > 0 else np.nan
        gb_pct = gb_count / total_bbe if total_bbe > 0 else np.nan

        # HR/FB: NaN when no fly balls — zero-denominator is not a zero observation.
        hr_fb = hr_count / fb_count if fb_count > 0 else np.nan

        # xwOBA: mean over batted-ball events that have a value.
        # Store the count separately so the posterior can use PA count as the
        # proper precision denominator rather than treating each start as 1 obs.
        xwoba_vals = grp[XWOBA_COL].dropna()
        xwoba   = xwoba_vals.mean() if not xwoba_vals.empty else np.nan
        xwoba_n = int(len(xwoba_vals))  # number of batted-ball events with a value

        return pd.Series({
            "game_date": game_date,
            "week":      week,
            "bf":        bf,
            "k_pct":     k_pct,
            "bb_pct":    bb_pct,
            "gb_pct":    gb_pct,
            "hr_fb":     hr_fb,
            "xwoba":     xwoba,
            "xwoba_n":   xwoba_n,
        })

    result = (
        df.groupby("game_pk", sort=False)
          .apply(_agg_game)
          .reset_index(drop=True)
          .sort_values("game_date")
    )
    return result


def load_or_fetch_starts(
    name: str,
    mlbam_id: int,
    end_date: str,
) -> pd.DataFrame:
    """
    Return per-start DataFrame for a pitcher, using cached CSV when available.

    Cache lives at data/starts/{name}_{season}.csv.
    On subsequent runs, only fetches new starts since the last cached game_date.
    """
    cache_path = DATA_DIR / f"{name}_{SEASON}.csv"
    season_start = SEASON_START

    existing: pd.DataFrame = pd.DataFrame()

    if cache_path.exists():
        existing = pd.read_csv(cache_path, parse_dates=["game_date"])
        # If the cache is missing any required column it was written by an older
        # version of the pipeline — discard and re-fetch from season start.
        if not REQUIRED_CACHE_COLUMNS.issubset(existing.columns):
            existing = pd.DataFrame()
        if existing.empty:
            fetch_start = season_start
        else:
            last_date = existing["game_date"].max()
            next_day  = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            if next_day > end_date:
                # Cache is fully up-to-date.
                return existing
            fetch_start = next_day
    else:
        fetch_start = season_start

    raw = fetch_pitcher_statcast(mlbam_id, fetch_start, end_date)
    if raw.empty:
        return existing

    new_starts = _aggregate_starts(raw)

    combined = (
        pd.concat([existing, new_starts], ignore_index=True)
          .drop_duplicates(subset=["game_date"])
          .sort_values("game_date")
          .reset_index(drop=True)
    )

    combined.to_csv(cache_path, index=False)
    return combined


def filter_starts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply season and MIN_BF filters.

    Excludes starts before SEASON_START and starts with bf < MIN_BF
    (injury exits / ejections produce garbage rate stats).
    """
    if df.empty:
        return df
    df = df[df["game_date"] >= pd.Timestamp(SEASON_START)].copy()
    df = df[df["bf"] >= MIN_BF].copy()
    return df.reset_index(drop=True)
