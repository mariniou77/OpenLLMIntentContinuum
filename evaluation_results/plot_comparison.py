#!/usr/bin/env python3
"""
Ablation Study Comparative Plots

Reads summary.json (and raw CSVs) from each exp_*/ sibling directory and
produces 7 publication-quality figures saved as both PDF and PNG.

Usage:
    cd evaluation_results/
    python3 plot_comparison.py

Output files (in the same directory):
    fig_01_intent_satisfaction_rate.{pdf,png}
    fig_02_violation_resolution.{pdf,png}
    fig_03_action_type_distribution.{pdf,png}
    fig_04_inference_latency.{pdf,png}
    fig_05_token_usage.{pdf,png}
    fig_06_locust_response_time_curve.{pdf,png}
    fig_07_ema_response_time.{pdf,png}
"""

import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no display required on server)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

UPPER_THRESHOLD = 2.0  # seconds
LOWER_THRESHOLD = 1.0  # seconds

EXPERIMENT_ORDER = [
    "exp_01_baseline",
    "exp_02_vertical_only",
    "exp_03_horizontal_only",
    "exp_04_service_placement_only",
    "exp_05_flow_scheduling_only",
    "exp_06_vertical_horizontal",
    "exp_07_vertical_horizontal_flow",
    "exp_08_full_system",
    "exp_09_cloud_llm_baseline",
]

SHORT_LABELS = {
    "exp_01_baseline":                "Baseline",
    "exp_02_vertical_only":           "Vertical",
    "exp_03_horizontal_only":         "Horizontal",
    "exp_04_service_placement_only":  "Placement",
    "exp_05_flow_scheduling_only":    "Flow Sched.",
    "exp_06_vertical_horizontal":     "V+H",
    "exp_07_vertical_horizontal_flow":"V+H+Flow",
    "exp_08_full_system":             "Full System",
    "exp_09_cloud_llm_baseline":      "Cloud LLM",
}

ACTION_COLORS = {
    "increase_cpu":     "#4C72B0",
    "reduce_cpu":       "#64B5CD",
    "add_replica":      "#DD8452",
    "remove_replica":   "#F0B67F",
    "service_placement":"#55A868",
    "flow_scheduling":  "#C44E52",
    "unknown":          "#8C8C8C",
}

SCRIPT_DIR = Path(__file__).parent


# ── Data loading ──────────────────────────────────────────────────────────────

def load_summaries() -> dict:
    """Return {exp_name: summary_dict} for all available experiments."""
    summaries = {}
    for name in EXPERIMENT_ORDER:
        p = SCRIPT_DIR / name / "summary.json"
        if p.exists():
            with open(p) as f:
                summaries[name] = json.load(f)
    return summaries


def load_locust_history(exp_name: str) -> list[dict]:
    """Return rows from locust_results_stats_history.csv."""
    p = SCRIPT_DIR / exp_name / "locust_results_stats_history.csv"
    if not p.exists():
        return []
    rows = []
    with open(p, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_ema_timeline(exp_name: str) -> list[dict]:
    """Return EMA timeline from intent_loop_log.json."""
    p = SCRIPT_DIR / exp_name / "intent_loop_log.json"
    if not p.exists():
        return []
    with open(p) as f:
        data = json.load(f)
    return data.get("ema_timeline", [])


def load_llm_interactions(exp_name: str) -> list[dict]:
    """Return rows from llm_interactions.csv."""
    p = SCRIPT_DIR / exp_name / "llm_interactions.csv"
    if not p.exists():
        return []
    rows = []
    with open(p, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save(fig, stem: str) -> None:
    for ext in ("pdf", "png"):
        path = SCRIPT_DIR / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"  Saved {path}")
    plt.close(fig)


def _bar_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)


# ── Figure 1 — Intent Satisfaction Rate ──────────────────────────────────────

def fig_intent_satisfaction_rate(summaries: dict) -> None:
    names = [n for n in EXPERIMENT_ORDER if n in summaries]
    labels = [SHORT_LABELS[n] for n in names]
    values = [
        (summaries[n].get("intent_satisfaction_rate") or 0) * 100
        for n in names
    ]
    colors = ["#c0c0c0" if n == "exp_01_baseline" else "#4C72B0" for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.6)
    ax.set_ylabel("Intent Satisfaction Rate (%)")
    ax.set_title("Intent Satisfaction Rate by Experiment Configuration")
    ax.set_ylim(0, 110)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=9)
    _bar_style(ax)
    plt.xticks(rotation=20, ha="right")
    _save(fig, "fig_01_intent_satisfaction_rate")


# ── Figure 2 — Violation Resolution Breakdown ────────────────────────────────

def fig_violation_resolution(summaries: dict) -> None:
    names = [n for n in EXPERIMENT_ORDER if n in summaries]
    labels = [SHORT_LABELS[n] for n in names]
    resolved = [summaries[n].get("violations_resolved", 0) for n in names]
    unresolved = [
        max(0, summaries[n].get("violations_detected", 0) - summaries[n].get("violations_resolved", 0))
        for n in names
    ]

    x = np.arange(len(names))
    width = 0.6

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, resolved, width, label="Resolved", color="#55A868")
    ax.bar(x, unresolved, width, bottom=resolved, label="Unresolved", color="#C44E52", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Number of Violations")
    ax.set_title("Violation Resolution by Experiment Configuration")
    ax.legend()
    _bar_style(ax)
    _save(fig, "fig_02_violation_resolution")


# ── Figure 3 — Action Type Distribution ──────────────────────────────────────

def fig_action_type_distribution(summaries: dict) -> None:
    # Exclude baseline and cloud (no actions or not run yet)
    names = [n for n in EXPERIMENT_ORDER
             if n in summaries and n not in ("exp_01_baseline", "exp_09_cloud_llm_baseline")]
    if not names:
        print("  No action-type data available yet — skipping Figure 3")
        return

    all_types = set()
    for n in names:
        all_types.update(summaries[n].get("action_type_breakdown", {}).keys())
    all_types = sorted(all_types)

    x = np.arange(len(names))
    width = 0.6 / max(len(all_types), 1)
    labels = [SHORT_LABELS[n] for n in names]

    fig, ax = plt.subplots(figsize=(11, 5))
    bottoms = np.zeros(len(names))

    for atype in all_types:
        values = [summaries[n].get("action_type_breakdown", {}).get(atype, 0) for n in names]
        color = ACTION_COLORS.get(atype, "#8C8C8C")
        ax.bar(x, values, width=0.6, bottom=bottoms, label=atype, color=color, alpha=0.85)
        bottoms += np.array(values)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Actions Executed (count)")
    ax.set_title("Action Type Distribution by Experiment")
    ax.legend(loc="upper left", fontsize=8)
    _bar_style(ax)
    _save(fig, "fig_03_action_type_distribution")


# ── Figure 4 — Inference Latency ─────────────────────────────────────────────

def fig_inference_latency(summaries: dict) -> None:
    names = [n for n in EXPERIMENT_ORDER
             if n in summaries and n != "exp_01_baseline"]
    labels = [SHORT_LABELS[n] for n in names]
    mean_lat = [summaries[n].get("mean_inference_latency_ms") or 0 for n in names]
    p95_lat = [summaries[n].get("p95_inference_latency_ms") or 0 for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, mean_lat, width, label="Mean latency (ms)", color="#4C72B0")
    ax.bar(x + width / 2, p95_lat, width, label="P95 latency (ms)", color="#DD8452", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("LLM Inference Latency (ms)")
    ax.set_title("LLM Inference Latency per Experiment")
    ax.legend()
    _bar_style(ax)
    _save(fig, "fig_04_inference_latency")


# ── Figure 5 — Token Usage ───────────────────────────────────────────────────

def fig_token_usage(summaries: dict) -> None:
    names = [n for n in EXPERIMENT_ORDER
             if n in summaries and n != "exp_01_baseline"]
    labels = [SHORT_LABELS[n] for n in names]
    prompt = [summaries[n].get("total_prompt_tokens") or 0 for n in names]
    completion = [summaries[n].get("total_completion_tokens") or 0 for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, prompt, width, label="Prompt tokens", color="#4C72B0")
    ax.bar(x + width / 2, completion, width, label="Completion tokens", color="#55A868", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Total Tokens")
    ax.set_title("Total Token Usage per Experiment")
    ax.legend()
    _bar_style(ax)
    _save(fig, "fig_05_token_usage")


# ── Figure 6 — Locust p95 Response Time Curve ────────────────────────────────

def fig_locust_p95_curve(summaries: dict) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    any_data = False

    for name in EXPERIMENT_ORDER:
        if name not in summaries:
            continue
        rows = load_locust_history(name)
        if not rows:
            continue

        try:
            times, p95s = [], []
            t0 = None
            for row in rows:
                if row.get("Name") not in ("/resize", "Aggregated"):
                    continue
                ts = float(row.get("Timestamp", 0))
                p95 = float(row.get("95%", 0))
                if t0 is None:
                    t0 = ts
                times.append((ts - t0) / 60)
                p95s.append(p95 / 1000)  # ms → s

            if times:
                ax.plot(times, p95s, label=SHORT_LABELS[name], alpha=0.8, linewidth=1.4)
                any_data = True
        except (ValueError, KeyError):
            continue

    if not any_data:
        print("  No Locust history data available yet — skipping Figure 6")
        plt.close(fig)
        return

    ax.axhline(UPPER_THRESHOLD, color="red", linestyle="--", linewidth=1, alpha=0.7, label=f"Upper threshold ({UPPER_THRESHOLD}s)")
    ax.axhline(LOWER_THRESHOLD, color="blue", linestyle="--", linewidth=1, alpha=0.7, label=f"Lower threshold ({LOWER_THRESHOLD}s)")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("p95 Response Time (s)")
    ax.set_title("Locust p95 Response Time Over Time — All Experiments")
    ax.legend(fontsize=8, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    _save(fig, "fig_06_locust_response_time_curve")


# ── Figure 7 — EMA Response Time Over Time ───────────────────────────────────

def fig_ema_response_time(summaries: dict) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    any_data = False

    for name in EXPERIMENT_ORDER:
        if name not in summaries:
            continue
        timeline = load_ema_timeline(name)
        if not timeline:
            continue

        try:
            times, emas = [], []
            viol_upper_t, viol_upper_v = [], []
            viol_lower_t, viol_lower_v = [], []

            t0 = None
            for point in timeline:
                ts_str = point.get("timestamp", "")
                ema = point.get("ema")
                if ema is None:
                    continue
                # Parse ISO timestamp to elapsed minutes
                from datetime import datetime as _dt
                ts = _dt.fromisoformat(ts_str)
                if t0 is None:
                    t0 = ts
                elapsed = (ts - t0).total_seconds() / 60
                times.append(elapsed)
                emas.append(ema)

                vtype = point.get("violation")
                if vtype == "UPPER_THRESHOLD_EXCEEDED":
                    viol_upper_t.append(elapsed)
                    viol_upper_v.append(ema)
                elif vtype == "LOWER_THRESHOLD_EXCEEDED":
                    viol_lower_t.append(elapsed)
                    viol_lower_v.append(ema)

            if times:
                line, = ax.plot(times, emas, label=SHORT_LABELS[name],
                                alpha=0.75, linewidth=1.3)
                # Violation markers (small, same colour)
                if viol_upper_t:
                    ax.scatter(viol_upper_t, viol_upper_v,
                               color=line.get_color(), marker="^", s=25, alpha=0.8, zorder=5)
                if viol_lower_t:
                    ax.scatter(viol_lower_t, viol_lower_v,
                               color=line.get_color(), marker="v", s=25, alpha=0.8, zorder=5)
                any_data = True
        except Exception:
            continue

    if not any_data:
        print("  No EMA timeline data available yet — skipping Figure 7")
        plt.close(fig)
        return

    ax.axhline(UPPER_THRESHOLD, color="red", linestyle="--", linewidth=1.2,
               alpha=0.8, label=f"Upper threshold ({UPPER_THRESHOLD}s)")
    ax.axhline(LOWER_THRESHOLD, color="blue", linestyle="--", linewidth=1.2,
               alpha=0.8, label=f"Lower threshold ({LOWER_THRESHOLD}s)")
    ax.fill_between([0, ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 20],
                    LOWER_THRESHOLD, UPPER_THRESHOLD,
                    alpha=0.06, color="green", label="SLO band")

    # Legend: experiment lines + violation marker explanation
    upper_marker = mpatches.Patch(color="grey", label="▲ Upper violation")
    lower_marker = mpatches.Patch(color="grey", label="▼ Lower violation")
    handles, leg_labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [upper_marker, lower_marker],
              fontsize=7, loc="upper right", ncol=2)

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("EMA Response Time (s)")
    ax.set_title("EMA Response Time Over Time — All Experiments")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    _save(fig, "fig_07_ema_response_time")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    summaries = load_summaries()
    if not summaries:
        print("No summary.json files found in sibling exp_*/ directories.")
        print("Run experiments first with:  python3 run_ablation.py --all")
        sys.exit(1)

    print(f"Loaded summaries for: {list(summaries.keys())}\n")

    print("Generating Figure 1 — Intent Satisfaction Rate...")
    fig_intent_satisfaction_rate(summaries)

    print("Generating Figure 2 — Violation Resolution Breakdown...")
    fig_violation_resolution(summaries)

    print("Generating Figure 3 — Action Type Distribution...")
    fig_action_type_distribution(summaries)

    print("Generating Figure 4 — Inference Latency...")
    fig_inference_latency(summaries)

    print("Generating Figure 5 — Token Usage...")
    fig_token_usage(summaries)

    print("Generating Figure 6 — Locust p95 Response Time Curve...")
    fig_locust_p95_curve(summaries)

    print("Generating Figure 7 — EMA Response Time Over Time...")
    fig_ema_response_time(summaries)

    print("\nAll figures saved.")


if __name__ == "__main__":
    main()
