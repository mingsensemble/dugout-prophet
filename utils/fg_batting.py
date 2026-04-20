"""
fg_batting.py — Fetch FanGraphs team batting leaderboards split by pitcher handedness (vs LHP / vs RHP).

Usage:
    python fg_batting.py                    # prints both splits
    python fg_batting.py --season 2026      # explicit season
    python fg_batting.py --out batting.csv  # write merged CSV
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Optional

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FG_API = "https://www.fangraphs.com/api/leaders/major-league/data"

SPLITS: dict[str, int] = {
    "vs_lhp": 13,   # split vs left-handed pitchers
    "vs_rhp": 14,   # split vs right-handed pitchers
}

# Team-level columns to keep
KEEP_COLS = ["Team", "TG", "K%", "BB%", "xwOBA", "wOBA"]

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

def fetch_split(month: int, season: int = 2026) -> pd.DataFrame:
    """Return raw DataFrame for one FanGraphs time split."""
    params = {**_BASE_PARAMS, "month": month, "season": season, "season1": season}
    resp = requests.get(FG_API, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    return pd.DataFrame(payload["data"])


def fetch_splits(season: int = 2026) -> dict[str, pd.DataFrame]:
    """Fetch all defined splits and return as {split_name: DataFrame}."""
    return {name: fetch_split(month, season) for name, month in SPLITS.items()}

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
# CLI
# ---------------------------------------------------------------------------

def _print_split(name: str, df: pd.DataFrame) -> None:
    print(f"\n=== {name.upper()} ===")
    print(clean(df).to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch FanGraphs batter leaderboards.")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--out", type=str, default=None, help="CSV output path")
    parser.add_argument("--merge", action="store_true", help="Print merged side-by-side view")
    args = parser.parse_args(argv)

    frames = fetch_splits(season=args.season)

    if args.out:
        merged = merge_splits(frames)
        merged.to_csv(args.out, index=False)
        print(f"Wrote {len(merged)} rows to {args.out}")
    elif args.merge:
        print(merge_splits(frames).to_string(index=False))
    else:
        for split_name, df in frames.items():
            _print_split(split_name, df)


if __name__ == "__main__":
    main(sys.argv[1:])
