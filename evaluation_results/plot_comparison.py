#!/usr/bin/env python3
"""
Ablation Study Comparative Plots

Reads summary.json (or summary_aggregated.json for repeated runs) from each
exp_*/ sibling directory and produces publication-quality figures.

With --repeat N runs, aggregated results live in exp_XX_agg/summary_aggregated.json
and contain <field>_mean / <field>_std keys. This script prefers the _agg/ dir
automatically and adds error bars wherever std is available.

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
    fig_08_time_normalised_isr.{pdf,png}
    fig_09_ema_time_in_band.{pdf,png}
"""

import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Experiment registry ───────────────────────────────────────────────────────

# Normal-load ablation suite
ABLATION_ORDER = [
    "exp_01_baseline",
    "exp_02_vertical_only",
    "exp_03_horizontal_only",
    "exp_04_service_placement_only",
    "exp_05_flow_scheduling_only",
    "exp_06_vertical_horizontal",
    "exp_07_vertical_horizontal_flow",
    "exp_08_full_system",
    "exp_09_cloud_llm_baseline",
    "exp_hpa_baseline",
]

# Stress-load experiments (2.7× peak users)
STRESS_ORDER = [
    "exp_10_stress_service_placement_only",
    "exp_11_stress_flow_scheduling_only",
    "exp_12_stress_full_system",
]

EXPERIMENT_ORDER = ABLATION_ORDER + STRESS_ORDER
STRESS_SET = set(STRESS_ORDER)

SHORT_LABELS = {
    "exp_01_baseline":                        "Baseline",
    "exp_02_vertical_only":                   "Vertical",
    "exp_03_horizontal_only":                 "Horizontal",
    "exp_04_service_placement_only":          "Placement",
    "exp_05_flow_scheduling_only":            "Flow Sched.",
    "exp_06_vertical_horizontal":             "V+H",
    "exp_07_vertical_horizontal_flow":        "V+H+Flow",
    "exp_08_full_system":                     "Full System",
    "exp_09_cloud_llm_baseline":              "Cloud LLM",
    "exp_hpa_baseline":                       "HPA (K8s)",
    "exp_10_stress_service_placement_only":   "S: Placement",
    "exp_11_stress_flow_scheduling_only":     "S: Flow Sched.",
    "exp_12_stress_full_system":              "S: Full System",
}

ACTION_COLORS = {
    "increase_cpu":      "#4C72B0",
    "reduce_cpu":        "#64B5CD",
    "add_replica":       "#DD8452",
    "remove_replica":    "#F0B67F",
    "service_placement": "#55A868",
    "flow_scheduling":   "#C44E52",
    "unknown":           "#8C8C8C",
}

UPPER_THRESHOLD = 2.0
LOWER_THRESHOLD = 1.0

SCRIPT_DIR = Path(__file__).parent


# ── Data loading ──────────────────────────────────────────────────────────────

def load_summaries() -> dict:
    """
    Return {exp_name: summary_dict} for all available experiments.
    Prefers exp_XX_agg/summary_aggregated.json (repeated runs) over
    exp_XX/summary.json (single run).
    """
    summaries = {}
    for name in EXPERIMENT_ORDER:
        agg_path = SCRIPT_DIR / f"{name}_agg" / "summary_aggregated.json"
        single_path = SCRIPT_DIR / name / "summary.json"
        if agg_path.exists():
            with open(agg_path) as f:
                summaries[name] = json.load(f)
        elif single_path.exists():
            with open(single_path) as f:
                summaries[name] = json.load(f)
    return summaries


def _timeseries_dir(exp_name: str) -> Path:
    """
    For time-series figures, return the directory that contains the raw CSV/JSON.
    With repeated runs, use _run1/ as the representative single run.
    """
    run1 = SCRIPT_DIR / f"{exp_name}_run1"
    if run1.exists():
        return run1
    return SCRIPT_DIR / exp_name


def load_locust_history(exp_name: str) -> list:
    p = _timeseries_dir(exp_name) / "locust_results_stats_history.csv"
    if not p.exists():
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def load_ema_timeline(exp_name: str) -> list:
    p = _timeseries_dir(exp_name) / "intent_loop_log.json"
    if not p.exists():
        return []
    with open(p) as f:
        return json.load(f).get("ema_timeline", [])


# ── Metric accessors ──────────────────────────────────────────────────────────

def _val(s: dict, field: str, default=0):
    """
    Read a metric from a summary dict.
    Aggregated summaries use '<field>_mean'; single-run summaries use '<field>'.
    """
    mean_key = f"{field}_mean"
    if mean_key in s:
        v = s[mean_key]
    else:
        v = s.get(field)
    return v if v is not None else default


def _std(s: dict, field: str):
    """Return std for a metric, or None if not available (single-run)."""
    return s.get(f"{field}_std")


def _action_breakdown(s: dict) -> dict:
    """Return action type counts, preferring _mean version for aggregated data."""
    return s.get("action_type_breakdown_mean") or s.get("action_type_breakdown") or {}


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


def _stress_hatch(name: str) -> str:
    """Return hatch pattern for stress experiments so they're visually distinct."""
    return "//" if name in STRESS_SET else ""


# ── Figure 1 — Intent Satisfaction Rate ──────────────────────────────────────

def fig_intent_satisfaction_rate(summaries: dict) -> None:
    names = [n for n in EXPERIMENT_ORDER if n in summaries]
    labels = [SHORT_LABELS[n] for n in names]
    values = [_val(summaries[n], "intent_satisfaction_rate") * 100 for n in names]
    errs   = [(_std(summaries[n], "intent_satisfaction_rate") or 0) * 100 for n in names]
    colors = []
    for n in names:
        if n == "exp_01_baseline":
            colors.append("#c0c0c0")
        elif n == "exp_09_cloud_llm_baseline":
            colors.append("#E377C2")
        elif n == "exp_hpa_baseline":
            colors.append("#E88C1F")
        elif n in STRESS_SET:
            colors.append("#8C8C8C")
        else:
            colors.append("#4C72B0")

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, values, color=colors, edgecolor="white", width=0.6,
           hatch=[_stress_hatch(n) for n in names])
    has_err = any(e > 0 for e in errs)
    if has_err:
        ax.errorbar(x, values, yerr=errs, fmt="none", color="black",
                    capsize=4, linewidth=1.2)

    ax.set_ylabel("Intent Satisfaction Rate (%)")
    ax.set_title("Intent Satisfaction Rate by Experiment Configuration")
    ax.set_ylim(0, 115)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")

    for i, (val, err) in enumerate(zip(values, errs)):
        label = f"{val:.0f}%" if err == 0 else f"{val:.0f}%\n±{err:.0f}%"
        ax.text(x[i], val + 2, label, ha="center", va="bottom", fontsize=8)

    # Vertical separator before stress experiments
    if any(n in STRESS_SET for n in names):
        sep_idx = next(i for i, n in enumerate(names) if n in STRESS_SET)
        ax.axvline(sep_idx - 0.5, color="gray", linestyle=":", linewidth=1)
        ax.text(sep_idx - 0.4, ax.get_ylim()[1] * 0.95, "← normal load | stress load →",
                fontsize=7, color="gray", ha="left")

    _bar_style(ax)
    _save(fig, "fig_01_intent_satisfaction_rate")


# ── Figure 2 — Violation Resolution Breakdown ────────────────────────────────

def fig_violation_resolution(summaries: dict) -> None:
    names = [n for n in EXPERIMENT_ORDER if n in summaries]
    labels = [SHORT_LABELS[n] for n in names]
    resolved   = [_val(summaries[n], "violations_resolved") for n in names]
    detected   = [_val(summaries[n], "violations_detected") for n in names]
    unresolved = [max(0, d - r) for d, r in zip(detected, resolved)]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, resolved,   0.6, label="Resolved",   color="#55A868",
           hatch=[_stress_hatch(n) for n in names])
    ax.bar(x, unresolved, 0.6, bottom=resolved, label="Unresolved", color="#C44E52",
           alpha=0.7, hatch=[_stress_hatch(n) for n in names])

    if any(n in STRESS_SET for n in names):
        sep_idx = next(i for i, n in enumerate(names) if n in STRESS_SET)
        ax.axvline(sep_idx - 0.5, color="gray", linestyle=":", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Number of Violations")
    ax.set_title("Violation Resolution by Experiment Configuration")
    ax.legend()
    _bar_style(ax)
    _save(fig, "fig_02_violation_resolution")


# ── Figure 3 — Action Type Distribution ──────────────────────────────────────

def fig_action_type_distribution(summaries: dict) -> None:
    # Exclude monitor-only conditions (no LLM actions to show) and the cloud LLM
    # baseline (handled separately in latency/token figures).
    _no_actions = {"exp_01_baseline", "exp_hpa_baseline", "exp_09_cloud_llm_baseline"}
    names = [n for n in EXPERIMENT_ORDER
             if n in summaries and n not in _no_actions]
    if not names:
        print("  No action-type data available yet — skipping Figure 3")
        return

    all_types = set()
    for n in names:
        all_types.update(_action_breakdown(summaries[n]).keys())
    all_types = sorted(all_types)

    x = np.arange(len(names))
    labels = [SHORT_LABELS[n] for n in names]
    bottoms = np.zeros(len(names))

    fig, ax = plt.subplots(figsize=(12, 5))
    for atype in all_types:
        values = [_action_breakdown(summaries[n]).get(atype, 0) for n in names]
        ax.bar(x, values, width=0.6, bottom=bottoms,
               label=atype, color=ACTION_COLORS.get(atype, "#8C8C8C"), alpha=0.85,
               hatch=[_stress_hatch(n) for n in names])
        bottoms += np.array(values, dtype=float)

    if any(n in STRESS_SET for n in names):
        sep_idx = next(i for i, n in enumerate(names) if n in STRESS_SET)
        ax.axvline(sep_idx - 0.5, color="gray", linestyle=":", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Actions Executed (mean per run)")
    ax.set_title("Action Type Distribution by Experiment")
    ax.legend(loc="upper left", fontsize=8)
    _bar_style(ax)
    _save(fig, "fig_03_action_type_distribution")


# ── Figure 4 — Inference Latency ─────────────────────────────────────────────

def fig_inference_latency(summaries: dict) -> None:
    names = [n for n in EXPERIMENT_ORDER
             if n in summaries and _val(summaries[n], "mean_inference_latency_ms") > 0]
    if not names:
        print("  No LLM latency data available — skipping Figure 4")
        return

    labels   = [SHORT_LABELS[n] for n in names]
    mean_lat = [_val(summaries[n], "mean_inference_latency_ms") for n in names]
    p95_lat  = [_val(summaries[n], "p95_inference_latency_ms") for n in names]
    mean_err = [_std(summaries[n], "mean_inference_latency_ms") or 0 for n in names]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, mean_lat, width, label="Mean (ms)", color="#4C72B0",
           hatch=[_stress_hatch(n) for n in names])
    ax.bar(x + width / 2, p95_lat,  width, label="P95 (ms)",  color="#DD8452",
           alpha=0.85, hatch=[_stress_hatch(n) for n in names])
    if any(e > 0 for e in mean_err):
        ax.errorbar(x - width / 2, mean_lat, yerr=mean_err,
                    fmt="none", color="black", capsize=3, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("LLM Inference Latency (ms)")
    ax.set_title("LLM Inference Latency per Experiment")
    ax.legend()
    _bar_style(ax)
    _save(fig, "fig_04_inference_latency")


# ── Figure 5 — Token Usage ────────────────────────────────────────────────────

def fig_token_usage(summaries: dict) -> None:
    names = [n for n in EXPERIMENT_ORDER
             if n in summaries and _val(summaries[n], "total_prompt_tokens") > 0]
    if not names:
        print("  No token data available — skipping Figure 5")
        return

    labels     = [SHORT_LABELS[n] for n in names]
    prompt     = [_val(summaries[n], "total_prompt_tokens") for n in names]
    completion = [_val(summaries[n], "total_completion_tokens") for n in names]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, prompt,     width, label="Prompt tokens",     color="#4C72B0",
           hatch=[_stress_hatch(n) for n in names])
    ax.bar(x + width / 2, completion, width, label="Completion tokens", color="#55A868",
           alpha=0.85, hatch=[_stress_hatch(n) for n in names])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Total Tokens (mean per run)")
    ax.set_title("Total Token Usage per Experiment")
    ax.legend()
    _bar_style(ax)
    _save(fig, "fig_05_token_usage")


# ── Figure 6 — Locust p95 Response Time Curve ────────────────────────────────

def fig_locust_p95_curve(summaries: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=False)
    titles = ["Normal Load (Ablation Suite)", "Stress Load (exp_10–12)"]
    groups = [ABLATION_ORDER, STRESS_ORDER]

    for ax, group, title in zip(axes, groups, titles):
        any_data = False
        for name in group:
            if name not in summaries:
                continue
            rows = load_locust_history(name)
            if not rows:
                continue
            try:
                times, p95s, t0 = [], [], None
                for row in rows:
                    if row.get("Name") not in ("/resize", "Aggregated"):
                        continue
                    ts = float(row.get("Timestamp", 0))
                    p95 = float(row.get("95%", 0))
                    if t0 is None:
                        t0 = ts
                    times.append((ts - t0) / 60)
                    p95s.append(p95 / 1000)
                if times:
                    ax.plot(times, p95s, label=SHORT_LABELS[name], alpha=0.8, linewidth=1.4)
                    any_data = True
            except (ValueError, KeyError):
                continue

        if not any_data:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        ax.axhline(UPPER_THRESHOLD, color="red",  linestyle="--", linewidth=1,
                   alpha=0.7, label=f"Upper ({UPPER_THRESHOLD}s)")
        ax.axhline(LOWER_THRESHOLD, color="blue", linestyle="--", linewidth=1,
                   alpha=0.7, label=f"Lower ({LOWER_THRESHOLD}s)")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("p95 Response Time (s)")
        ax.set_title(title)
        ax.legend(fontsize=7, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("Locust p95 Response Time Over Time", fontsize=12)
    plt.tight_layout()
    _save(fig, "fig_06_locust_response_time_curve")


# ── Figure 7 — EMA Response Time Over Time ───────────────────────────────────

def fig_ema_response_time(summaries: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=False)
    titles = ["Normal Load (Ablation Suite)", "Stress Load (exp_10–12)"]
    groups = [ABLATION_ORDER, STRESS_ORDER]

    for ax, group, title in zip(axes, groups, titles):
        any_data = False
        for name in group:
            if name not in summaries:
                continue
            timeline = load_ema_timeline(name)
            if not timeline:
                continue
            try:
                from datetime import datetime as _dt
                times, emas = [], []
                viol_upper_t, viol_upper_v = [], []
                viol_lower_t, viol_lower_v = [], []
                t0 = None

                for point in timeline:
                    ema = point.get("ema")
                    if ema is None:
                        continue
                    ts = _dt.fromisoformat(point.get("timestamp", ""))
                    if t0 is None:
                        t0 = ts
                    elapsed = (ts - t0).total_seconds() / 60
                    times.append(elapsed)
                    emas.append(ema)

                    vtype = point.get("violation")
                    if not point.get("grace_period") and not point.get("cooldown"):
                        if vtype == "UPPER_THRESHOLD_EXCEEDED":
                            viol_upper_t.append(elapsed)
                            viol_upper_v.append(ema)
                        elif vtype == "LOWER_THRESHOLD_EXCEEDED":
                            viol_lower_t.append(elapsed)
                            viol_lower_v.append(ema)

                if times:
                    line, = ax.plot(times, emas, label=SHORT_LABELS[name],
                                    alpha=0.75, linewidth=1.3)
                    if viol_upper_t:
                        ax.scatter(viol_upper_t, viol_upper_v, color=line.get_color(),
                                   marker="^", s=25, alpha=0.8, zorder=5)
                    if viol_lower_t:
                        ax.scatter(viol_lower_t, viol_lower_v, color=line.get_color(),
                                   marker="v", s=25, alpha=0.8, zorder=5)
                    any_data = True
            except Exception:
                continue

        if not any_data:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        ax.axhline(UPPER_THRESHOLD, color="red",  linestyle="--", linewidth=1.2, alpha=0.8)
        ax.axhline(LOWER_THRESHOLD, color="blue", linestyle="--", linewidth=1.2, alpha=0.8)
        xlim = ax.get_xlim()
        ax.fill_between([xlim[0], xlim[1] if xlim[1] > 0 else 20],
                        LOWER_THRESHOLD, UPPER_THRESHOLD, alpha=0.06, color="green")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("EMA Response Time (s)")
        ax.set_title(title)
        upper_m = mpatches.Patch(color="grey", label="▲ Upper violation")
        lower_m = mpatches.Patch(color="grey", label="▼ Lower violation")
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles + [upper_m, lower_m], fontsize=7,
                  loc="upper right", ncol=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("EMA Response Time Over Time (▲ upper violation  ▼ lower violation)", fontsize=12)
    plt.tight_layout()
    _save(fig, "fig_07_ema_response_time")


# ── Figure 8 — Time-Normalised ISR ───────────────────────────────────────────

def fig_time_normalised_isr(summaries: dict) -> None:
    names  = [n for n in EXPERIMENT_ORDER if n in summaries]
    labels = [SHORT_LABELS[n] for n in names]
    values = [_val(summaries[n], "time_normalised_isr") for n in names]
    errs   = [_std(summaries[n], "time_normalised_isr") or 0 for n in names]
    colors = ["#c0c0c0" if n == "exp_01_baseline"
              else "#E377C2" if n == "exp_09_cloud_llm_baseline"
              else "#E88C1F" if n == "exp_hpa_baseline"
              else "#8C8C8C" if n in STRESS_SET
              else "#4C72B0"
              for n in names]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, values, color=colors, edgecolor="white", width=0.6,
           hatch=[_stress_hatch(n) for n in names])
    if any(e > 0 for e in errs):
        ax.errorbar(x, values, yerr=errs, fmt="none", color="black",
                    capsize=4, linewidth=1.2)

    for i, (val, err) in enumerate(zip(values, errs)):
        if val > 0:
            label = f"{val:.3f}" if err == 0 else f"{val:.3f}\n±{err:.3f}"
            ax.text(x[i], val + max(errs) * 0.05 if errs else val * 0.02,
                    label, ha="center", va="bottom", fontsize=8)

    if any(n in STRESS_SET for n in names):
        sep_idx = next(i for i, n in enumerate(names) if n in STRESS_SET)
        ax.axvline(sep_idx - 0.5, color="gray", linestyle=":", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Violations Resolved per Minute")
    ax.set_title("Time-Normalised ISR\n(corrects for cooldown suppression effect)")
    _bar_style(ax)
    _save(fig, "fig_08_time_normalised_isr")


# ── Figure 9 — EMA Time-in-Band ──────────────────────────────────────────────

def fig_ema_time_in_band(summaries: dict) -> None:
    names  = [n for n in EXPERIMENT_ORDER if n in summaries]
    labels = [SHORT_LABELS[n] for n in names]
    values = [_val(summaries[n], "ema_time_in_band_pct") for n in names]
    errs   = [_std(summaries[n], "ema_time_in_band_pct") or 0 for n in names]
    colors = ["#c0c0c0" if n == "exp_01_baseline"
              else "#E377C2" if n == "exp_09_cloud_llm_baseline"
              else "#E88C1F" if n == "exp_hpa_baseline"
              else "#8C8C8C" if n in STRESS_SET
              else "#4C72B0"
              for n in names]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, values, color=colors, edgecolor="white", width=0.6,
           hatch=[_stress_hatch(n) for n in names])
    if any(e > 0 for e in errs):
        ax.errorbar(x, values, yerr=errs, fmt="none", color="black",
                    capsize=4, linewidth=1.2)

    for i, (val, err) in enumerate(zip(values, errs)):
        if val > 0:
            label = f"{val:.0f}%" if err == 0 else f"{val:.0f}%\n±{err:.0f}%"
            ax.text(x[i], val + 1.5, label, ha="center", va="bottom", fontsize=8)

    if any(n in STRESS_SET for n in names):
        sep_idx = next(i for i, n in enumerate(names) if n in STRESS_SET)
        ax.axvline(sep_idx - 0.5, color="gray", linestyle=":", linewidth=1)

    ax.set_ylim(0, 115)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("EMA Time-in-Band (%)")
    ax.set_title("EMA Time Within SLO Band [1.0s – 2.0s]\n"
                 "(% of monitoring cycles with 1.0 ≤ EMA ≤ 2.0)")
    _bar_style(ax)
    _save(fig, "fig_09_ema_time_in_band")


# ── K8s Resource CSV helpers ─────────────────────────────────────────────────

def _run_dirs(exp_name: str) -> list:
    """
    Return existing run directories for an experiment in priority order.
    Handles both multi-run (_run1/_run2/_run3) and single-run ({exp_name}/) layouts.
    """
    dirs = []
    for suffix in ("_run1", "_run2", "_run3"):
        d = SCRIPT_DIR / f"{exp_name}{suffix}"
        if d.exists():
            dirs.append(d)
    # Single-run layout (4th suite): results in {exp_name}/ directly
    if not dirs:
        d = SCRIPT_DIR / exp_name
        if d.exists():
            dirs.append(d)
    return dirs


def _load_pod_cpu(exp_name: str, service_prefix: str = "microservice3") -> list:
    """
    Load per-pod CPU data for a service across all available run directories.

    Returns a list of dicts, one per run:
        {"elapsed_min": [...], "cpu_m": [...]}
    where cpu_m is the SUM of all matching pods at each timestamp (handles replicas).
    Supports both multi-run (_run1/_run2/_run3) and single-run ({exp_name}/) layouts.
    """
    runs = []
    for run_dir in _run_dirs(exp_name):
        csv_path = run_dir / "k8s_pod_resources.csv"
        if not csv_path.exists():
            continue
        # Aggregate CPU across all pods matching the service prefix at each timestamp
        from collections import defaultdict
        ts_cpu: dict = defaultdict(int)
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                pod = row.get("pod_name", "")
                if not pod.startswith(service_prefix):
                    continue
                cpu_str = row.get("cpu_m", "0").strip()
                if not cpu_str:
                    continue
                try:
                    cpu_val = int(cpu_str.rstrip("m"))
                except ValueError:
                    continue
                ts_cpu[row["timestamp"]] += cpu_val

        if not ts_cpu:
            continue
        timestamps = sorted(ts_cpu.keys())
        from datetime import datetime as _dt
        t0 = _dt.fromisoformat(timestamps[0].replace("Z", "+00:00"))
        elapsed = [(
            (_dt.fromisoformat(ts.replace("Z", "+00:00")) - t0).total_seconds() / 60
        ) for ts in timestamps]
        cpu_vals = [ts_cpu[ts] for ts in timestamps]
        runs.append({"elapsed_min": elapsed, "cpu_m": cpu_vals})
    return runs


def _load_pod_memory(exp_name: str, service_prefix: str = "microservice3") -> list:
    """
    Load per-pod memory data (MiB) for a service across all available run directories.

    Returns a list of dicts, one per run:
        {"elapsed_min": [...], "mem_mi": [...]}
    where mem_mi is the SUM of all matching pods at each timestamp.
    Supports both multi-run (_run1/_run2/_run3) and single-run ({exp_name}/) layouts.
    """
    runs = []
    for run_dir in _run_dirs(exp_name):
        csv_path = run_dir / "k8s_pod_resources.csv"
        if not csv_path.exists():
            continue
        from collections import defaultdict
        ts_mem: dict = defaultdict(int)
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                pod = row.get("pod_name", "")
                if not pod.startswith(service_prefix):
                    continue
                mem_str = row.get("memory_mi", "0").strip()
                if not mem_str:
                    continue
                try:
                    mem_val = int(mem_str.rstrip("Mi"))
                except ValueError:
                    continue
                ts_mem[row["timestamp"]] += mem_val

        if not ts_mem:
            continue
        timestamps = sorted(ts_mem.keys())
        from datetime import datetime as _dt
        t0 = _dt.fromisoformat(timestamps[0].replace("Z", "+00:00"))
        elapsed = [(
            (_dt.fromisoformat(ts.replace("Z", "+00:00")) - t0).total_seconds() / 60
        ) for ts in timestamps]
        mem_vals = [ts_mem[ts] for ts in timestamps]
        runs.append({"elapsed_min": elapsed, "mem_mi": mem_vals})
    return runs


def _load_node_cpu(exp_name: str) -> list:
    """
    Load per-node CPU data for a single run (run1 as representative).

    Returns a dict: {node_name: {"elapsed_min": [...], "cpu_m": [...]}}
    Supports both multi-run (_run1/_run2/_run3) and single-run ({exp_name}/) layouts.
    """
    run_dirs = _run_dirs(exp_name)
    if not run_dirs:
        return {}
    run_dir = run_dirs[0]
    csv_path = run_dir / "k8s_node_resources.csv"
    if not csv_path.exists():
        return {}

    from collections import defaultdict
    from datetime import datetime as _dt
    node_data: dict = defaultdict(lambda: {"ts": [], "cpu": []})

    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            node = row.get("node", "")
            cpu_str = row.get("cpu_cores", "0").strip()
            if not cpu_str:
                continue
            try:
                cpu_m = int(cpu_str.rstrip("m")) if cpu_str.endswith("m") else int(float(cpu_str) * 1000)
            except ValueError:
                continue
            node_data[node]["ts"].append(row["timestamp"])
            node_data[node]["cpu"].append(cpu_m)

    result = {}
    for node, data in node_data.items():
        if not data["ts"]:
            continue
        t0 = _dt.fromisoformat(data["ts"][0].replace("Z", "+00:00"))
        elapsed = [(
            (_dt.fromisoformat(ts.replace("Z", "+00:00")) - t0).total_seconds() / 60
        ) for ts in data["ts"]]
        result[node] = {"elapsed_min": elapsed, "cpu_m": data["cpu"]}
    return result


def _aggregate_runs(runs: list, grid_step: float = 0.25) -> tuple:
    """
    Interpolate runs onto a common time grid and return (grid, mean, std).
    grid_step is in minutes.
    """
    if not runs:
        return np.array([]), np.array([]), np.array([])
    max_time = min(max(r["elapsed_min"]) for r in runs)
    grid = np.arange(0, max_time, grid_step)
    interp_runs = []
    for r in runs:
        interp_runs.append(np.interp(grid, r["elapsed_min"], r["cpu_m"]))
    arr = np.array(interp_runs)
    return grid, arr.mean(axis=0), arr.std(axis=0)


# ── Figure 10 — Pod CPU Usage Over Time (all 4 services) ─────────────────────

SERVICES = [
    ("microservice1-deployment", "ms1"),
    ("microservice2-deployment", "ms2"),
    ("microservice3-deployment", "ms3"),
    ("microservice4-deployment", "ms4"),
]


def fig_pod_cpu_over_time(summaries: dict) -> None:
    """
    2×2 grid of subplots — one per microservice (ms1–ms4).
    Each subplot overlays all ablation experiments.
    Multi-run experiments show mean ± shaded std band; single-run shows a line.
    """
    names = [n for n in ABLATION_ORDER if n in summaries]
    if not names:
        print("  No experiments available — skipping Figure 10")
        return

    cmap = plt.get_cmap("tab10")
    exp_colors = {n: cmap(i % 10) for i, n in enumerate(names)}

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
    axes_flat = [axes[r][c] for r in range(2) for c in range(2)]

    for ax, (svc_prefix, svc_short) in zip(axes_flat, SERVICES):
        any_data = False
        for name in names:
            runs = _load_pod_cpu(name, service_prefix=svc_prefix)
            if not runs:
                continue
            color = exp_colors[name]
            if len(runs) >= 2:
                grid, mean, std = _aggregate_runs(runs)
                ax.plot(grid, mean, label=SHORT_LABELS.get(name, name),
                        color=color, linewidth=1.5)
                ax.fill_between(grid, mean - std, mean + std,
                                alpha=0.18, color=color)
            else:
                r = runs[0]
                ax.plot(r["elapsed_min"], r["cpu_m"],
                        label=SHORT_LABELS.get(name, name),
                        color=color, linewidth=1.5)
            any_data = True

        if not any_data:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")

        ax.set_title(svc_short)
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel(f"{svc_short} Pod CPU (millicores, sum of replicas)")
        ax.legend(fontsize=7, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("Pod CPU Usage Over Time — All Services\n"
                 "(shaded band = ±1 std across 3 runs)", fontsize=12)
    plt.tight_layout()
    _save(fig, "fig_10_pod_cpu_over_time")


# ── Figure 11 — Node CPU Usage Over Time ─────────────────────────────────────

def fig_node_cpu_over_time(summaries: dict) -> None:
    """
    Per-node CPU usage over time for the key comparison experiments.
    Shows a horizontal dashed line at 800m (80% of 1000m node capacity assumed)
    as the service_placement trigger threshold context.

    One subplot per experiment (up to 4 key experiments).
    Uses run1 as the representative run for each.
    """
    # Show only the most meaningful experiments for node-level view
    priority = ["exp_01_baseline", "exp_08_full_system",
                "exp_09_cloud_llm_baseline", "exp_hpa_baseline"]
    names = [n for n in priority if n in summaries]
    if not names:
        print("  No experiments available — skipping Figure 11")
        return

    ncols = min(len(names), 2)
    nrows = (len(names) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    node_colors = {"master": "#4C72B0", "worker1": "#DD8452", "worker2": "#55A868"}

    for i, name in enumerate(names):
        ax = axes_flat[i]
        node_data = _load_node_cpu(name)
        any_data = False

        for node, data in sorted(node_data.items()):
            color = node_colors.get(node, "#8C8C8C")
            ax.plot(data["elapsed_min"], data["cpu_m"],
                    label=node, color=color, linewidth=1.3, alpha=0.85)
            any_data = True

        # service_placement fires when sFlow cpu_utilization > 40% of the node.
        # Each node has 4 CPUs (4000m allocatable), so 40% = 1600m in millicores.
        ax.axhline(1600, color="red", linestyle="--", linewidth=1,
                   alpha=0.6, label="SP trigger: 40% of 4-core node (1600m)")

        if not any_data:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")

        ax.set_title(SHORT_LABELS.get(name, name))
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Node CPU (millicores)")
        ax.legend(fontsize=7, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)

    # Hide unused subplots
    for j in range(len(names), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Node CPU Usage Over Time (run 1)\n"
                 "Red dashed = service_placement trigger (sFlow cpu_util > 40% = 1600m on 4-core nodes)", fontsize=11)
    plt.tight_layout()
    _save(fig, "fig_11_node_cpu_over_time")


# ── Figure 12 — Node Memory Usage Over Time ───────────────────────────────────

def fig_node_memory_over_time(summaries: dict) -> None:
    """
    Per-node RAM usage (MiB) over time for key comparison experiments.
    Uses run1 as the representative run for each experiment.
    Node capacity is 8192 MiB (8 GiB per node on Chameleon KVM).
    """
    priority = ["exp_01_baseline", "exp_08_full_system",
                "exp_09_cloud_llm_baseline", "exp_hpa_baseline"]
    names = [n for n in priority if n in summaries]
    if not names:
        print("  No experiments available — skipping Figure 12")
        return

    NODE_CAPACITY_MIB = 8192  # 8 GiB per node (Chameleon KVM @TACC)

    ncols = min(len(names), 2)
    nrows = (len(names) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    node_colors = {"master": "#4C72B0", "worker1": "#DD8452", "worker2": "#55A868"}

    for i, name in enumerate(names):
        ax = axes_flat[i]
        _rdirs = _run_dirs(name)
        run_dir = _rdirs[0] if _rdirs else None
        csv_path = run_dir / "k8s_node_resources.csv" if run_dir else None
        any_data = False

        if csv_path and csv_path.exists():
            from collections import defaultdict
            from datetime import datetime as _dt
            node_data: dict = defaultdict(lambda: {"ts": [], "mem": []})

            with open(csv_path, newline="") as f:
                for row in csv.DictReader(f):
                    node = row.get("node", "")
                    mem_str = row.get("memory_mi", "").strip()
                    if not mem_str:
                        continue
                    try:
                        mem_mib = int(mem_str.rstrip("Mi"))
                    except ValueError:
                        continue
                    node_data[node]["ts"].append(row["timestamp"])
                    node_data[node]["mem"].append(mem_mib)

            for node, data in sorted(node_data.items()):
                if not data["ts"]:
                    continue
                t0 = _dt.fromisoformat(data["ts"][0].replace("Z", "+00:00"))
                elapsed = [(
                    (_dt.fromisoformat(ts.replace("Z", "+00:00")) - t0).total_seconds() / 60
                ) for ts in data["ts"]]
                color = node_colors.get(node, "#8C8C8C")
                ax.plot(elapsed, data["mem"], label=node, color=color,
                        linewidth=1.3, alpha=0.85)
                any_data = True

        # 80% memory capacity reference line
        ax.axhline(NODE_CAPACITY_MIB * 0.8, color="orange", linestyle="--",
                   linewidth=1, alpha=0.7, label=f"80% capacity ({int(NODE_CAPACITY_MIB*0.8):,} MiB)")

        if not any_data:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")

        ax.set_ylim(0, NODE_CAPACITY_MIB * 1.05)
        ax.set_title(SHORT_LABELS.get(name, name))
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Node RAM Usage (MiB)")
        ax.legend(fontsize=7, loc="lower right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)

    for j in range(len(names), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Node RAM Usage Over Time (run 1)\n"
                 "Orange dashed = 80% of 8 GiB node capacity (6,554 MiB)", fontsize=11)
    plt.tight_layout()
    _save(fig, "fig_12_node_memory_over_time")


# ── Figure 13 — Pod Memory Usage Over Time (all 4 services) ──────────────────

def fig_pod_memory_over_time(summaries: dict) -> None:
    """
    2×2 grid of subplots — one per microservice (ms1–ms4).
    Each subplot overlays all ablation experiments.
    Multi-run experiments show mean ± shaded std band; single-run shows a line.
    """
    names = [n for n in ABLATION_ORDER if n in summaries]
    if not names:
        print("  No experiments available — skipping Figure 13")
        return

    cmap = plt.get_cmap("tab10")
    exp_colors = {n: cmap(i % 10) for i, n in enumerate(names)}

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
    axes_flat = [axes[r][c] for r in range(2) for c in range(2)]

    for ax, (svc_prefix, svc_short) in zip(axes_flat, SERVICES):
        any_data = False
        for name in names:
            runs = _load_pod_memory(name, service_prefix=svc_prefix)
            if not runs:
                continue
            color = exp_colors[name]
            if len(runs) >= 2:
                # Reuse _aggregate_runs by temporarily aliasing mem_mi → cpu_m key
                runs_compat = [{"elapsed_min": r["elapsed_min"], "cpu_m": r["mem_mi"]}
                               for r in runs]
                grid, mean, std = _aggregate_runs(runs_compat)
                ax.plot(grid, mean, label=SHORT_LABELS.get(name, name),
                        color=color, linewidth=1.5)
                ax.fill_between(grid, mean - std, mean + std,
                                alpha=0.18, color=color)
            else:
                r = runs[0]
                ax.plot(r["elapsed_min"], r["mem_mi"],
                        label=SHORT_LABELS.get(name, name),
                        color=color, linewidth=1.5)
            any_data = True

        if not any_data:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")

        ax.set_title(svc_short)
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel(f"{svc_short} Pod Memory (MiB, sum of replicas)")
        ax.legend(fontsize=7, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("Pod Memory Usage Over Time — All Services\n"
                 "(shaded band = ±1 std across 3 runs)", fontsize=12)
    plt.tight_layout()
    _save(fig, "fig_13_pod_memory_over_time")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    summaries = load_summaries()
    if not summaries:
        print("No summary.json files found in sibling exp_*/ directories.")
        print("Run experiments first with:  python3 run_ablation.py --all --repeat 3")
        sys.exit(1)

    print(f"Loaded summaries for: {list(summaries.keys())}\n")

    steps = [
        ("Figure 1  — Intent Satisfaction Rate",      fig_intent_satisfaction_rate),
        ("Figure 2  — Violation Resolution",           fig_violation_resolution),
        ("Figure 3  — Action Type Distribution",       fig_action_type_distribution),
        ("Figure 4  — Inference Latency",              fig_inference_latency),
        ("Figure 5  — Token Usage",                    fig_token_usage),
        ("Figure 6  — Locust p95 Response Curve",      fig_locust_p95_curve),
        ("Figure 7  — EMA Response Time Over Time",    fig_ema_response_time),
        ("Figure 8  — Time-Normalised ISR",            fig_time_normalised_isr),
        ("Figure 9  — EMA Time-in-Band",               fig_ema_time_in_band),
        ("Figure 10 — Pod CPU Over Time (all services)", fig_pod_cpu_over_time),
        ("Figure 11 — Node CPU Over Time",              fig_node_cpu_over_time),
        ("Figure 12 — Node Memory Over Time",           fig_node_memory_over_time),
        ("Figure 13 — Pod Memory Over Time (all services)", fig_pod_memory_over_time),
    ]

    for title, fn in steps:
        print(f"Generating {title}...")
        fn(summaries)

    print("\nAll figures saved.")


if __name__ == "__main__":
    main()
