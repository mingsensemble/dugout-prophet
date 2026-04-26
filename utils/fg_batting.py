"""
fg_batting.py — Fetch FanGraphs team batting leaderboards split by pitcher handedness and venue.

Fetched splits: vs_lhp, vs_rhp, home, away
Derived cross-splits: home_vs_lhp, home_vs_rhp, away_vs_lhp, away_vs_rhp

Usage:
    python fg_batting.py                    # print merged + cross-split table
    python fg_batting.py --season 2026      # explicit season
    python fg_batting.py --out batting.csv  # write full CSV
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv
load_dotenv()
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FG_API = "https://www.fangraphs.com/api/leaders/major-league/data"

SPLITS: dict[str, int] = {
    "vs_lhp": 13,   # split vs left-handed pitchers
    "vs_rhp": 14,   # split vs right-handed pitchers
    "home":   15,   # home games
    "away":   16,   # away games
}

# Team-level columns to keep
KEEP_COLS = ["Team", "TG", "PA", "K%", "BB%", "xwOBA", "wOBA"]

ID_COLS = ["Team"]

_BASE_PARAMS: dict[str, str | int] = {
    "pos":      "all",
    "stats":    "bat",
    "lg":       "all",
    "qual":     "0",
    "ind":      "0",
    "team":     "0,ts",
    "rost":     "",
    "filter":   "",
    "players":  "0",
    "sortcol":  "9",
    "sortdir":  "default",
    "type":     "8",
    "pageitems": 2_000_000_000,
    "pagenum":  1,
}

# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def _session(cookie: str | None = None) -> requests.Session:
    """Build a requests Session, optionally injecting a FanGraphs auth cookie.

    Pass a raw cookie string (e.g. "__cflb=<val>; fg_user=<val>") or set the
    FG_COOKIE environment variable — useful in CI where interactive login isn't
    possible.
    """
    s = requests.Session()
    raw = cookie or os.environ.get("FG_COOKIE")
    if raw:
        for part in raw.split(";"):
            part = part.strip()
            if "=" in part:
                name, _, val = part.partition("=")
                s.cookies.set(name.strip(), val.strip(), domain="www.fangraphs.com")
    return s


def fetch_split(month: int, season: int = 2026, cookie: str | None = None) -> pd.DataFrame:
    """Return raw DataFrame for one FanGraphs time split."""
    params = {**_BASE_PARAMS, "month": month, "season": season, "season1": season}
    resp = _session(cookie).get(FG_API, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    return pd.DataFrame(payload["data"])


def fetch_splits(season: int = 2026, cookie: str | None = None) -> dict[str, pd.DataFrame]:
    """Fetch all defined splits and return as {split_name: DataFrame}."""
    return {name: fetch_split(month, season, cookie) for name, month in SPLITS.items()}

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------

def _extract_team(val: str) -> str:
    """Strip FanGraphs HTML anchor and return bare team abbreviation."""
    m = re.search(r">([^<]+)</a>", str(val))
    return m.group(1) if m else val


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Select, strip HTML from Team, and type-cast fantasy-relevant columns."""
    available = [c for c in KEEP_COLS if c in df.columns]
    out = df[available].copy()
    if "Team" in out.columns:
        out["Team"] = out["Team"].apply(_extract_team)
    for pct_col in ("BB%", "K%"):
        if pct_col in out.columns:
            out[pct_col] = pd.to_numeric(out[pct_col], errors="coerce")
    return out.reset_index(drop=True)


def merge_splits(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join splits side-by-side on Team; stat columns get a _vs_lhp/_vs_rhp suffix."""
    merged: Optional[pd.DataFrame] = None
    for split_name, df in frames.items():
        cleaned = clean(df)
        stat_cols = [c for c in cleaned.columns if c not in ID_COLS]
        renamed = cleaned.rename(columns={c: f"{c}_{split_name}" for c in stat_cols})
        if merged is None:
            merged = renamed
        else:
            merged = merged.merge(renamed, on=ID_COLS, how="outer")
    return merged  # type: ignore[return-value]

# ---------------------------------------------------------------------------
# Cross-split estimation
# ---------------------------------------------------------------------------

RATE_STATS = ["wOBA", "K%", "BB%"]
CROSS_SHRINKAGE = 200  # pseudo-PA toward 0.5 home fraction; tune as needed


def compute_pa_cross_splits(df: pd.DataFrame, shrinkage: int = CROSS_SHRINKAGE) -> pd.DataFrame:
    """
    Estimate PA for home/away x vs_lhp/vs_rhp using proportionality.

    Assumes P(home | vs_lhp) ≈ P(home). Shrinkage regularizes the home
    fraction toward 0.5 — useful early in the season when PA_total is small.
    """
    out = df[["Team"]].copy()
    pa_total = df["PA_vs_lhp"] + df["PA_vs_rhp"]
    alpha = pa_total / (pa_total + shrinkage)
    home_frac = alpha * (df["PA_home"] / pa_total) + (1 - alpha) * 0.5
    away_frac = 1 - home_frac
    out["PA_home_vs_lhp"] = (df["PA_vs_lhp"] * home_frac).round(1)
    out["PA_away_vs_lhp"] = (df["PA_vs_lhp"] * away_frac).round(1)
    out["PA_home_vs_rhp"] = (df["PA_vs_rhp"] * home_frac).round(1)
    out["PA_away_vs_rhp"] = (df["PA_vs_rhp"] * away_frac).round(1)
    return out


def compute_rate_cross_splits(df: pd.DataFrame, stat: str) -> pd.DataFrame:
    """
    Estimate cross-split rate stats via additive main-effects decomposition:

        rate_home_vs_lhp ≈ rate_home + rate_vs_lhp - rate_total

    Assumes venue and pitcher handedness effects are additive (no interaction).
    """
    out = df[["Team"]].copy()
    pa_total = df["PA_vs_lhp"] + df["PA_vs_rhp"]
    rate_total = (
        df["PA_vs_lhp"] * df[f"{stat}_vs_lhp"]
        + df["PA_vs_rhp"] * df[f"{stat}_vs_rhp"]
    ) / pa_total
    for venue in ("home", "away"):
        for hand in ("lhp", "rhp"):
            out[f"{stat}_{venue}_vs_{hand}"] = (
                df[f"{stat}_{venue}"] + df[f"{stat}_vs_{hand}"] - rate_total
            )
    return out


def build_full_table(merged: pd.DataFrame, shrinkage: int = CROSS_SHRINKAGE) -> pd.DataFrame:
    """Append all cross-split PA and rate columns to the merged base table."""
    parts = [
        merged,
        compute_pa_cross_splits(merged, shrinkage).drop(columns=["Team"]),
    ]
    for stat in RATE_STATS:
        if f"{stat}_vs_lhp" in merged.columns:
            parts.append(compute_rate_cross_splits(merged, stat).drop(columns=["Team"]))
    return pd.concat(parts, axis=1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch FanGraphs batter leaderboards.")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--out", type=str, default=None, help="CSV output path")
    parser.add_argument("--shrinkage", type=int, default=CROSS_SHRINKAGE,
                        help="Pseudo-PA shrinkage for cross-split PA estimates")
    parser.add_argument("--cookie", type=str, default=None,
                        help='FanGraphs session cookie string, e.g. "__cflb=X; fg_user=Y". '
                             "Falls back to FG_COOKIE env var if unset.")
    args = parser.parse_args(argv)

    frames = fetch_splits(season=args.season, cookie=args.cookie)
    merged = merge_splits(frames)
    full = build_full_table(merged, shrinkage=args.shrinkage)

    if args.out:
        full.to_csv(args.out, index=False)
        print(f"Wrote {len(full)} rows × {len(full.columns)} columns to {args.out}")
    else:
        print(full.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
