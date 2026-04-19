"""
pipeline.py — Orchestrates ingest → model → tagger → output.

Usage:
    python -m sp_tagger.pipeline --week 4 [--date 2026-04-14] [--verbose]
"""

from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

from .config import N_SIM, PRIORS, ROSTER, TAG_VALUE
from .ingest import filter_starts, load_or_fetch_starts
from .model import compute_posterior, normal_posterior_from_starts, simulate_posteriors
from .tagger import confidence_flag, metric_summary, tag_distribution

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

console = Console()

# Metrics that feed the model (order matters for posterior dict construction).
METRICS = ["bb_pct", "k_pct", "hr_fb", "gb_pct", "xwoba"]


# ---------------------------------------------------------------------------
# Per-pitcher processing
# ---------------------------------------------------------------------------

def process_pitcher(
    name: str,
    config: dict,
    current_week: int,
    end_date: str,
) -> dict:
    """
    Full pipeline for one pitcher.

    Returns a result dict with all columns needed for the output table and CSV.
    """
    mlbam_id  = config["mlbam_id"]
    team        = config.get("team", "")
    roster_type = config.get("roster_type", "")
    overrides   = config.get("prior_overrides", {})

    # --- Ingest ---
    starts = load_or_fetch_starts(name, mlbam_id, end_date)
    starts = filter_starts(starts)

    n_starts = len(starts)
    weeks    = starts["week"].tolist() if n_starts > 0 else []

    # Estimate IP: rough proxy (bf * 3/9 innings, not provided directly by Savant).
    # Use bf sum as a stand-in; label as "BF" in the table.
    total_bf = int(starts["bf"].sum()) if n_starts > 0 else 0

    # --- Posteriors ---
    posteriors_params: dict[str, tuple[str, float, float]] = {}
    shrinkages: list[float] = []
    n_effs: list[float] = []
    metric_details: dict[str, dict] = {}

    for metric in METRICS:
        prior = dict(PRIORS[metric])  # copy so overrides don't mutate global

        if metric in overrides:
            prior.update(overrides[metric])

        if metric == "xwoba":
            # xwOBA uses a normal posterior with per-start PA counts as the
            # precision denominator via normal_posterior_from_starts.
            xwoba_means = starts["xwoba"].values   if n_starts > 0 else np.array([])
            xwoba_ns    = starts["xwoba_n"].values if n_starts > 0 else np.array([])
            p1, p2, n_eff, shrinkage = normal_posterior_from_starts(
                xwoba_means=xwoba_means,
                xwoba_ns=xwoba_ns,
                weeks=weeks,
                current_week=current_week,
                prior=prior,
            )
        else:
            obs = starts[metric].values if n_starts > 0 else np.array([])
            p1, p2, n_eff, shrinkage = compute_posterior(
                observations=obs,
                weeks=weeks,
                current_week=current_week,
                prior=prior,
            )

        posteriors_params[metric] = (prior["dist"], p1, p2)
        shrinkages.append(shrinkage)
        n_effs.append(n_eff)
        metric_details[metric] = {"n_eff": n_eff, "shrinkage": shrinkage}

    # --- Simulation ---
    draws = simulate_posteriors(posteriors_params, n_sim=N_SIM)

    # --- Tagging ---
    tag_result = tag_distribution(draws)
    mean_shrinkage = float(np.mean(shrinkages))
    conf = confidence_flag(mean_shrinkage)

    # --- Metric summaries ---
    summaries: dict[str, dict] = {}
    for metric in METRICS:
        summaries[metric] = metric_summary(draws[metric])
    summaries["kbb"] = metric_summary(draws["kbb"])

    return {
        "name":           name,
        "team":           team,
        "roster_type":    roster_type,
        "overrides":      overrides,
        "n_starts":       n_starts,
        "total_bf":       total_bf,
        "modal_tag":      tag_result["modal_tag"],
        "ev_score":       tag_result["ev_score"],
        "tag_probs":      tag_result["tag_probs"],
        "confidence":     conf,
        "mean_shrinkage": mean_shrinkage,
        "n_eff_mean":     float(np.mean(n_effs)),
        "shrinkage_mean": mean_shrinkage,
        "summaries":      summaries,
        "metric_details": metric_details,  # per-metric n_eff + shrinkage for --debug
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _tag_dist_str(tag_probs: dict[str, float]) -> str:
    """Format tag distribution as 'Ace 71% | VA 22% | WH 6% | CB 1%'."""
    abbrevs = {
        "Ace": "Ace",
        "Volatile Ace": "VA",
        "Workhorse": "WH",
        "Cherry Bomb": "CB",
    }
    parts = sorted(tag_probs.items(), key=lambda x: -x[1])
    return " | ".join(
        f"{abbrevs[tag]} {pct:.0%}" for tag, pct in parts
    )


def print_debug(results: list[dict], week: int) -> None:
    """Print per-pitcher per-metric posterior details for diagnosis."""
    console.print(f"\n[bold]Debug — posterior details (Week {week})[/bold]")
    label_map = {
        "bb_pct": "bb_pct ",
        "k_pct":  "k_pct  ",
        "hr_fb":  "hr_fb  ",
        "gb_pct": "gb_pct ",
        "xwoba":  "xwoba  ",
    }
    for r in results:
        s  = r["summaries"]
        md = r["metric_details"]
        overrides_str = (
            "  overrides: " + ", ".join(
                f"{m}({'+'.join(f'{k}={v}' for k,v in ov.items())})"
                for m, ov in r["overrides"].items()
            )
            if r["overrides"] else "  no overrides"
        )
        console.print(
            f"  [bold]{r['name']}[/bold] posteriors"
            f" (shrinkage_mean={r['mean_shrinkage']:.3f}, conf={r['confidence']}, week {week})"
            f"\n  {overrides_str}"
        )
        for metric in METRICS:
            sm  = s[metric]
            det = md[metric]
            console.print(
                f"    {label_map[metric]}: "
                f"mean={sm['mean']:.3f} [p10={sm['p10']:.3f}, p90={sm['p90']:.3f}]"
                f"  n_eff={det['n_eff']:.2f}  shrinkage={det['shrinkage']:.3f}"
            )
        kbb = s["kbb"]
        console.print(
            f"    kbb    : "
            f"mean={kbb['mean']:.3f} [p10={kbb['p10']:.3f}, p90={kbb['p90']:.3f}]"
            f"  (derived)"
        )


def print_table(results: list[dict], week: int, run_date: str, verbose: bool) -> None:
    """Render the rich terminal table."""
    title = f"SP Tag Report — Week {week} ({run_date})"
    table = Table(title=title, show_lines=True)

    table.add_column("Pitcher",         style="bold")
    table.add_column("GS#",             justify="right")
    table.add_column("Tag Distribution")
    table.add_column("EV",              justify="right")
    table.add_column("Conf",            justify="center")
    table.add_column("BF",              justify="right")

    for rank, r in enumerate(results, start=1):
        conf       = r["confidence"]
        dagger     = " †" if conf == "LOW" else ""
        tag_str    = _tag_dist_str(r["tag_probs"]) + dagger
        ev_str     = f"{r['ev_score']:.2f}"
        bf_str     = str(r["total_bf"])
        table.add_row(
            r["name"],
            str(rank),
            tag_str,
            ev_str,
            conf,
            bf_str,
        )

    console.print(table)

    if any(r["confidence"] == "LOW" for r in results):
        console.print("† LOW confidence — prior-dominated", style="dim")

    if verbose:
        console.print()
        for r in results:
            s = r["summaries"]
            console.print(
                f"  [bold]{r['name']}[/bold]"
                f"  BB% {s['bb_pct']['mean']:.3f} [{s['bb_pct']['p10']:.3f}–{s['bb_pct']['p90']:.3f}]"
                f"  K-BB% {s['kbb']['mean']:.3f} [{s['kbb']['p10']:.3f}–{s['kbb']['p90']:.3f}]"
            )
            console.print(
                f"  {'':>{len(r['name'])}}"
                f"  HR/FB {s['hr_fb']['mean']:.3f} [{s['hr_fb']['p10']:.3f}–{s['hr_fb']['p90']:.3f}]"
                f"  GB% {s['gb_pct']['mean']:.3f} [{s['gb_pct']['p10']:.3f}–{s['gb_pct']['p90']:.3f}]"
                f"  xwOBA {s['xwoba']['mean']:.3f} [{s['xwoba']['p10']:.3f}–{s['xwoba']['p90']:.3f}]"
            )


def write_csv(results: list[dict], week: int) -> Path:
    """Write output/week_XX.csv with all columns."""
    out_path = OUTPUT_DIR / f"week_{week:02d}.csv"

    fieldnames = [
        "name", "team", "gs_priority", "modal_tag", "ev_score", "confidence",
        "p_ace", "p_volatile_ace", "p_workhorse", "p_cherry_bomb",
        "bb_mean", "bb_p10", "bb_p90",
        "k_mean", "k_p10", "k_p90",
        "kbb_mean", "kbb_p10", "kbb_p90",
        "hrfb_mean", "hrfb_p10", "hrfb_p90",
        "gb_mean", "gb_p10", "gb_p90",
        "xwoba_mean", "xwoba_p10", "xwoba_p90",
        "n_starts", "n_eff_mean", "shrinkage_mean",
    ]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rank, r in enumerate(results, start=1):
            tp = r["tag_probs"]
            s  = r["summaries"]
            writer.writerow({
                "name":           r["name"],
                "team":           r["team"],
                "gs_priority":    rank,
                "modal_tag":      r["modal_tag"],
                "ev_score":       round(r["ev_score"], 4),
                "confidence":     r["confidence"],
                "p_ace":          round(tp.get("Ace", 0), 4),
                "p_volatile_ace": round(tp.get("Volatile Ace", 0), 4),
                "p_workhorse":    round(tp.get("Workhorse", 0), 4),
                "p_cherry_bomb":  round(tp.get("Cherry Bomb", 0), 4),
                "bb_mean":        round(s["bb_pct"]["mean"], 4),
                "bb_p10":         round(s["bb_pct"]["p10"], 4),
                "bb_p90":         round(s["bb_pct"]["p90"], 4),
                "k_mean":         round(s["k_pct"]["mean"], 4),
                "k_p10":          round(s["k_pct"]["p10"], 4),
                "k_p90":          round(s["k_pct"]["p90"], 4),
                "kbb_mean":       round(s["kbb"]["mean"], 4),
                "kbb_p10":        round(s["kbb"]["p10"], 4),
                "kbb_p90":        round(s["kbb"]["p90"], 4),
                "hrfb_mean":      round(s["hr_fb"]["mean"], 4),
                "hrfb_p10":       round(s["hr_fb"]["p10"], 4),
                "hrfb_p90":       round(s["hr_fb"]["p90"], 4),
                "gb_mean":        round(s["gb_pct"]["mean"], 4),
                "gb_p10":         round(s["gb_pct"]["p10"], 4),
                "gb_p90":         round(s["gb_pct"]["p90"], 4),
                "xwoba_mean":     round(s["xwoba"]["mean"], 4),
                "xwoba_p10":      round(s["xwoba"]["p10"], 4),
                "xwoba_p90":      round(s["xwoba"]["p90"], 4),
                "n_starts":       r["n_starts"],
                "n_eff_mean":     round(r["n_eff_mean"], 3),
                "shrinkage_mean": round(r["shrinkage_mean"], 3),
            })

    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SP Tagger — weekly GS priority report")
    parser.add_argument("--week",    type=int, required=True, help="Current week number")
    parser.add_argument("--date",    type=str, default=date.today().isoformat(),
                        help="End date for Savant pull (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print P10/P90 metric intervals per pitcher")
    parser.add_argument("--debug",   action="store_true",
                        help="Print per-metric n_eff and shrinkage for every pitcher")
    args = parser.parse_args()

    console.print(f"[bold]Fetching data through {args.date} (week {args.week})…[/bold]")

    all_results: list[dict] = []

    for name, cfg in ROSTER.items():
        console.print(f"  Processing {name}…", end=" ")
        try:
            result = process_pitcher(name, cfg, args.week, args.date)
            all_results.append(result)
            console.print(
                f"[green]✓[/green] {result['n_starts']} starts, "
                f"modal={result['modal_tag']}, ev={result['ev_score']:.2f}"
            )
        except Exception as exc:
            console.print(f"[red]✗ {exc}[/red]")

    # Sort by EV score descending → GS priority.
    all_results.sort(key=lambda x: x["ev_score"], reverse=True)

    print_table(all_results, week=args.week, run_date=args.date, verbose=args.verbose)

    if args.debug:
        print_debug(all_results, week=args.week)

    out_path = write_csv(all_results, week=args.week)
    console.print(f"\n[dim]CSV written → {out_path}[/dim]")


if __name__ == "__main__":
    main()
