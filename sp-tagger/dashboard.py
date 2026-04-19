"""
dashboard.py — Streamlit SP Tag Report dashboard.

Usage:
    streamlit run dashboard.py
"""

from __future__ import annotations

import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from sp_tagger.config import CB_PENALTY, ROSTER, SEASON_START, TAG_VALUE
from sp_tagger.pipeline import METRICS, process_pitcher

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TAG_COLORS = {
    "Ace":          "#22c55e",
    "Ace Potential": "#f59e0b",
    "Workhorse":    "#3b82f6",
    "Cherry Bomb":  "#ef4444",
    "Toby": "#c1c1c1",
}

TAG_ABBREV = {
    "Ace":          "Ace",
    "Ace Potential": "AP",
    "Workhorse":    "WH",
    "Cherry Bomb":  "CB",
    "Toby": "TB"
}

ROSTER_TYPE_LABEL = {
    "on_roster":  "Roster",
    "monitoring": "Monitor",
    "benchmark":  "Bench",
}

CONF_COLORS = {
    "HIGH": "#22c55e",
    "MED":  "#f59e0b",
    "LOW":  "#ef4444",
}

METRIC_LABELS = {
    "bb_pct": "BB%",
    "k_pct":  "K%",
    "hr_fb":  "HR/FB",
    "gb_pct": "GB%",
    "xwoba":  "xwOBA",
    "kbb":    "K-BB%",
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SP Tagger",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .metric-label { font-size: 0.75rem; color: #6b7280; }
  .tag-pill {
    display: inline-block; padding: 2px 8px; border-radius: 9999px;
    font-size: 0.75rem; font-weight: 600; margin: 1px;
  }
  .ev-score { font-size: 1.1rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _current_week(today: date) -> int:
    season_start = date.fromisoformat(SEASON_START)
    return max(1, math.floor((today - season_start).days / 7) + 1)


@st.cache_data(ttl=3600, show_spinner=False)
def _run_pipeline(run_date: str, week: int) -> list[dict]:
    """Fetch and process all roster pitchers. Cached for 1 hour."""
    results: list[dict] = []
    progress = st.progress(0, text="Fetching Statcast data…")
    names    = list(ROSTER.keys())

    for i, name in enumerate(names):
        cfg = ROSTER[name]
        progress.progress((i + 1) / len(names), text=f"Processing {name}…")
        try:
            r = process_pitcher(name, cfg, week, run_date)
            results.append(r)
        except Exception as exc:
            st.warning(f"⚠ {name}: {exc}")

    progress.empty()
    results.sort(key=lambda x: x["ev_score"], reverse=True)
    return results


def _tag_dist_html(tag_probs: dict[str, float], modal: str) -> str:
    """Render tag distribution as colored pills sorted by probability."""
    parts = sorted(tag_probs.items(), key=lambda x: -x[1])
    pills = []
    for tag, p in parts:
        if p < 0.005:
            continue
        color = TAG_COLORS[tag]
        weight = "700" if tag == modal else "500"
        pills.append(
            f'<span class="tag-pill" style="background:{color}22; color:{color}; '
            f'font-weight:{weight}">{TAG_ABBREV[tag]} {p:.0%}</span>'
        )
    return " ".join(pills)


def _stacked_bar_chart(results: list[dict]) -> go.Figure:
    """Horizontal stacked bar chart of tag probabilities per pitcher."""
    names = [r["name"] for r in results]
    tags  = list(TAG_VALUE.keys())

    fig = go.Figure()
    for tag in tags:
        probs = [r["tag_probs"].get(tag, 0) for r in results]
        fig.add_trace(go.Bar(
            name=TAG_ABBREV[tag],
            y=names,
            x=probs,
            orientation="h",
            marker_color=TAG_COLORS[tag],
            text=[f"{p:.0%}" if p >= 0.05 else "" for p in probs],
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate=f"<b>%{{y}}</b> — {tag}: %{{x:.1%}}<extra></extra>",
        ))

    fig.update_layout(
        barmode="stack",
        height=max(260, len(results) * 42),
        margin=dict(l=0, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(tickformat=".0%", range=[0, 1]),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
    )
    return fig


def _metric_posterior_chart(summaries: dict[str, dict]) -> go.Figure:
    """Horizontal bar chart with P10–P90 error bars for each metric."""
    metric_keys = ["bb_pct", "k_pct", "kbb", "hr_fb", "gb_pct", "xwoba"]
    labels  = [METRIC_LABELS[m] for m in metric_keys]
    means   = [summaries[m]["mean"] for m in metric_keys]
    lows    = [summaries[m]["mean"] - summaries[m]["p10"] for m in metric_keys]
    highs   = [summaries[m]["p90"] - summaries[m]["mean"] for m in metric_keys]

    fig = go.Figure(go.Bar(
        x=labels,
        y=means,
        error_y=dict(type="data", symmetric=False, array=highs, arrayminus=lows,
                     color="#94a3b8", thickness=2),
        marker_color="#3b82f6",
        hovertemplate="<b>%{x}</b><br>mean %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=0, r=0, t=8, b=8),
        yaxis=dict(tickformat=".3f"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=11),
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("SP Tagger")
    today      = date.today()
    run_date   = st.date_input("Report date", value=today, max_value=today)
    auto_week  = _current_week(run_date)
    week_input = st.number_input(
        "Season week", min_value=1, max_value=30,
        value=auto_week,
        help=f"Auto-computed from {SEASON_START}",
    )

    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    show_verbose = st.toggle("Show metric posteriors", value=True)
    st.caption(f"Season start: {SEASON_START}")
    st.caption(f"CB penalty: {CB_PENALTY:+.2f} per unit P(CB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

run_date_str = run_date.isoformat()

st.title(f"SP Tag Report — Week {week_input}")
st.caption(f"Data through {run_date_str} · CB penalty = {CB_PENALTY}")

with st.spinner("Loading pitcher data…"):
    results = _run_pipeline(run_date_str, int(week_input))

if not results:
    st.error("No pitcher data returned. Check Statcast connectivity.")
    st.stop()

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

st.subheader("GS Priority Rankings")

summary_rows = []
for rank, r in enumerate(results, 1):
    tp   = r["tag_probs"]
    conf = r["confidence"]
    summary_rows.append({
        "#":       rank,
        "Pitcher": r["name"],
        "Type":    ROSTER_TYPE_LABEL.get(r.get("roster_type", ""), "—"),
        "Team":    r["team"] or "—",
        "EV":      round(r["ev_score"], 3),
        "Conf":    conf,
        "Starts":  r["n_starts"],
        "BF":      r["total_bf"],
        "Ace":     tp["Ace"],
        "AP":      tp["Ace Potential"],
        "Ace+AP":  tp["Ace"] + tp["Ace Potential"],
        "WH":      tp["Workhorse"],
        "TB":      tp["Toby"],
        "CB":      tp["Cherry Bomb"],
        "Modal":   r["modal_tag"],
    })

summary_df = pd.DataFrame(summary_rows).set_index("#")


def _color_conf(val: str) -> str:
    return f"color: {CONF_COLORS.get(val, '#000')}"


def _color_ev(val: float) -> str:
    if val >= 0.7:
        return "color: #22c55e; font-weight: bold"
    elif val <= 0.1:
        return "color: #ef4444"
    return ""


def _highlight_roster(row: pd.Series) -> list[str]:
    if row["Type"] == "Roster":
        return ["background-color: #22c55e18"] * len(row)
    return [""] * len(row)


styled = (
    summary_df.style
    .apply(_highlight_roster, axis=1)
    .applymap(_color_conf, subset=["Conf"])
    .applymap(_color_ev,   subset=["EV"])
    .format({"EV": "{:.3f}", "Ace": "{:.0%}", "AP": "{:.0%}", "Ace+AP": "{:.0%}", "WH": "{:.0%}", "TB": "{:.0%}", "CB": "{:.0%}"})
)

SUMMARY_COL_ORDER = ["Pitcher", "Type", "Team", "EV", "Conf", "Starts", "BF",
                     "Ace", "AP", "Ace+AP", "WH", "TB", "CB", "Modal"]

st.dataframe(styled, use_container_width=True, height=min(400, 60 + len(results) * 38),
             column_order=SUMMARY_COL_ORDER)

# ---------------------------------------------------------------------------
# Tag distribution chart
# ---------------------------------------------------------------------------

st.subheader("Tag Distribution")
st.plotly_chart(_stacked_bar_chart(results), use_container_width=True)

# ---------------------------------------------------------------------------
# Per-pitcher detail
# ---------------------------------------------------------------------------

if show_verbose:
    st.subheader("Pitcher Detail")
    for r in results:
        conf        = r["confidence"]
        conf_color  = CONF_COLORS[conf]
        dagger      = " †" if conf == "LOW" else ""
        header_label = (
            f"{r['name']}  ·  EV {r['ev_score']:.3f}  ·  "
            f":{conf.lower()}[{conf}{dagger}]  ·  "
            f"{r['n_starts']} starts / {r['total_bf']} BF"
        )

        with st.expander(header_label):
            col_left, col_right = st.columns([1, 1])

            with col_left:
                # Tag probability pills
                st.markdown(
                    _tag_dist_html(r["tag_probs"], r["modal_tag"]),
                    unsafe_allow_html=True,
                )
                st.caption(f"Modal: **{r['modal_tag']}**  ·  shrinkage {r['mean_shrinkage']:.3f}")

                # Per-metric detail table
                s = r["summaries"]
                md_rows = []
                for metric in METRICS + ["kbb"]:
                    ms = s[metric]
                    det = r["metric_details"].get(metric, {})
                    md_rows.append({
                        "Metric":    METRIC_LABELS.get(metric, metric),
                        "Mean":      f"{ms['mean']:.3f}",
                        "P10":       f"{ms['p10']:.3f}",
                        "P90":       f"{ms['p90']:.3f}",
                        "n_eff":     f"{det.get('n_eff', 0):.1f}" if det else "—",
                        "Shrinkage": f"{det.get('shrinkage', 0):.3f}" if det else "—",
                    })
                st.dataframe(
                    pd.DataFrame(md_rows),
                    hide_index=True,
                    use_container_width=True,
                    height=255,
                )

            with col_right:
                st.plotly_chart(
                    _metric_posterior_chart(r["summaries"]),
                    use_container_width=True,
                    key=f"posterior_{r['name']}",
                )

                # Applied overrides
                if r["overrides"]:
                    st.caption("**Prior overrides:**")
                    for metric, ov in r["overrides"].items():
                        parts = ", ".join(f"{k}={v}" for k, v in ov.items())
                        st.caption(f"  {METRIC_LABELS.get(metric, metric)}: {parts}")
