#!/usr/bin/env python3
"""
Ablation / 4-Way Comparative Plots (publication styling)

Reads summary.json (or summary_aggregated.json for repeated runs) from each
exp_*/ sibling directory and produces publication-quality figures.

Paper styling (11th experiment benchmark):
  - Scenario nomenclature: Full System -> LATS,
    Cloud LLM -> ChatGPT4o, baseline -> No Management; HPA / VPA / HPA+VPA stay literal.
  - Deterministic left-to-right order: LATS, ChatGPT4o, HPA, VPA, HPA+VPA, No Management.
  - No variance graphics (error bars, +/- labels, std bands removed).
  - Aggregate figures titled "(Mean Aggregated - N Runs)"; genuinely single-run
    time-series titled "(Representative Run)".
  - EMA time-series are sorted by timestamp (de-duplicated) before plotting to
    remove the chronological back-and-forth artifact.

Usage:
    cd evaluation_results/11th_experiment/
    python3 ../plot_comparison.py                      # write figures next to the runs
    python3 ../plot_comparison.py --out-dir paper_plots  # write into paper_plots/

Note: input data is always read from the script's own directory; --out-dir only
changes where the PDFs are written.
"""

import argparse
from typing import Any
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
    "exp_vpa_baseline",
    "exp_hpa_vpa_baseline",
]

# Stress-load experiments (2.7× peak users)
STRESS_ORDER = [
    "exp_10_stress_service_placement_only",
    "exp_11_stress_flow_scheduling_only",
    "exp_12_stress_full_system",
]

EXPERIMENT_ORDER = ABLATION_ORDER + STRESS_ORDER
STRESS_SET = set(STRESS_ORDER)

# Paper nomenclature (11th experiment 4-way benchmark).
SHORT_LABELS = {
    "exp_01_baseline":                        "No Management",
    "exp_02_vertical_only":                   "Vertical",
    "exp_03_horizontal_only":                 "Horizontal",
    "exp_04_service_placement_only":          "Placement",
    "exp_05_flow_scheduling_only":            "Flow Sched.",
    "exp_06_vertical_horizontal":             "V+H",
    "exp_07_vertical_horizontal_flow":        "V+H+Flow",
    "exp_08_full_system":                     "LATS",
    "exp_09_cloud_llm_baseline":              "ChatGPT4o",
    "exp_hpa_baseline":                       "HPA",
    "exp_vpa_baseline":                       "VPA",
    "exp_hpa_vpa_baseline":                   "HPA+VPA",
    "exp_10_stress_service_placement_only":   "S: Placement",
    "exp_11_stress_flow_scheduling_only":     "S: Flow Sched.",
    "exp_12_stress_full_system":              "S: Full System",
}

# Deterministic paper render order (left-to-right) for the benchmark.
PAPER_ORDER = [
    "exp_08_full_system",         # LATS (hero)
    "exp_09_cloud_llm_baseline",  # ChatGPT4o
    "exp_hpa_baseline",           # HPA
    "exp_vpa_baseline",           # VPA
    "exp_hpa_vpa_baseline",       # HPA+VPA
    "exp_01_baseline",            # No Management
]

# Consistent per-scenario colours across every figure.
SCENARIO_COLORS = {
    "exp_08_full_system":         "#4C72B0",  # LATS (hero, blue)
    "exp_09_cloud_llm_baseline":  "#55A868",  # ChatGPT4o (green)
    "exp_hpa_baseline":           "#E88C1F",  # HPA (amber)
    "exp_vpa_baseline":           "#8172B3",  # VPA (purple)
    "exp_hpa_vpa_baseline":       "#C44E52",  # HPA+VPA (red)
    "exp_01_baseline":            "#BFBFBF",  # No Management (grey)
}

N_RUNS = 3  # repeats per scenario in the 11th experiment

ACTION_COLORS = {
    "increase_cpu":      "#4C72B0",
    "reduce_cpu":        "#64B5CD",
    "add_replica":       "#DD8452",
    "remove_replica":    "#F0B67F",
    "service_placement": "#55A868",
    "flow_scheduling":   "#C44E52",
    "unknown":           "#8C8C8C",
}

UPPER_THRESHOLD = 3.0
LOWER_THRESHOLD = 0.5

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR  # overridden by --out-dir


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


def _action_breakdown(s: dict) -> dict:
    """Return action type counts, preferring _mean version for aggregated data."""
    return s.get("action_type_breakdown_mean") or s.get("action_type_breakdown") or {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save(fig, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{stem}.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"  Saved {path}")
    plt.close(fig)


def _bar_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)


def _agg_title(base: str) -> str:
    """Append the multi-run aggregation tag to an aggregate-figure title."""
    return f"{base}\n(Mean Aggregated - {N_RUNS} Runs)"


def _paper_names(summaries: dict) -> list:
    """Scenario keys present in `summaries`, in deterministic paper order.
    Falls back to EXPERIMENT_ORDER for older (non-4-way) suites."""
    names = [n for n in PAPER_ORDER if n in summaries]
    if names:
        return names
    return [n for n in EXPERIMENT_ORDER if n in summaries]


def _scenario_color(name: str) -> str:
    return SCENARIO_COLORS.get(name, "#4C72B0")


# ── Figure 1 — Intent Satisfaction Rate ──────────────────────────────────────

def fig_intent_satisfaction_rate(summaries: dict) -> None:
    names = _paper_names(summaries)
    if not names:
        print("  No data — skipping Figure 1")
        return
    labels = [SHORT_LABELS[n] for n in names]
    values = [_val(summaries[n], "intent_satisfaction_rate") * 100 for n in names]
    colors = [_scenario_color(n) for n in names]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, values, color=colors, edgecolor="white", width=0.6)

    ax.set_ylabel("Intent Satisfaction Rate (%)")
    ax.set_title(_agg_title("Intent Satisfaction Rate by Scenario"))
    ax.set_ylim(0, 115)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    for i, val in enumerate(values):
        ax.text(x[i], val + 2, f"{val:.0f}%", ha="center", va="bottom", fontsize=9)

    _bar_style(ax)
    _save(fig, "fig_01_intent_satisfaction_rate")


# ── Figure 2 — Violation Resolution Breakdown ────────────────────────────────

def fig_violation_resolution(summaries: dict) -> None:
    names = _paper_names(summaries)
    if not names:
        print("  No data — skipping Figure 2")
        return
    labels = [SHORT_LABELS[n] for n in names]
    resolved   = [_val(summaries[n], "violations_resolved") for n in names]
    detected   = [_val(summaries[n], "violations_detected") for n in names]
    unresolved = [max(0, d - r) for d, r in zip(detected, resolved)]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, resolved,   0.6, label="Resolved",   color="#55A868")
    ax.bar(x, unresolved, 0.6, bottom=resolved, label="Unresolved", color="#C44E52", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Number of Violations")
    ax.set_title(_agg_title("Violation Resolution by Scenario"))
    ax.legend()
    _bar_style(ax)
    _save(fig, "fig_02_violation_resolution")


# ── Figure 3 — Action Type Distribution ──────────────────────────────────────

def fig_action_type_distribution(summaries: dict) -> None:
    # Only the two LLM-driven scenarios produce actions to break down.
    _no_actions = {"exp_01_baseline", "exp_hpa_baseline"}
    names = [n for n in _paper_names(summaries) if n not in _no_actions]
    if not names:
        print("  No action-type data available yet — skipping Figure 3")
        return

    all_types = set()
    for n in names:
        all_types.update(_action_breakdown(summaries[n]).keys())
    all_types = sorted(all_types)
    if not all_types:
        print("  No action-type data available yet — skipping Figure 3")
        return

    x = np.arange(len(names))
    labels = [SHORT_LABELS[n] for n in names]
    bottoms = np.zeros(len(names))

    # Narrow bars + compact canvas (user request for fig_03).
    fig, ax = plt.subplots(figsize=(6, 5))
    for atype in all_types:
        values = [_action_breakdown(summaries[n]).get(atype, 0) for n in names]
        ax.bar(x, values, width=0.35, bottom=bottoms,
               label=atype, color=ACTION_COLORS.get(atype, "#8C8C8C"), alpha=0.9)
        bottoms += np.array(values, dtype=float)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="center")
    ax.set_xlim(-0.7, len(names) - 0.3)
    ax.set_ylabel("Actions Executed (mean per run)")
    ax.set_title(_agg_title("Action Type Distribution"))
    ax.legend(loc="upper right", fontsize=8)
    _bar_style(ax)
    _save(fig, "fig_03_action_type_distribution")


# ── Figure 4 — Inference Latency ─────────────────────────────────────────────

def fig_inference_latency(summaries: dict) -> None:
    names = [n for n in _paper_names(summaries)
             if _val(summaries[n], "mean_inference_latency_ms") > 0]
    if not names:
        print("  No LLM latency data available — skipping Figure 4")
        return

    labels   = [SHORT_LABELS[n] for n in names]
    mean_lat = [_val(summaries[n], "mean_inference_latency_ms") for n in names]
    p95_lat  = [_val(summaries[n], "p95_inference_latency_ms") for n in names]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, mean_lat, width, label="Mean (ms)", color="#4C72B0")
    ax.bar(x + width / 2, p95_lat,  width, label="P95 (ms)",  color="#DD8452", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("LLM Inference Latency (ms)")
    ax.set_title(_agg_title("LLM Inference Latency per Scenario"))
    ax.legend()
    _bar_style(ax)
    _save(fig, "fig_04_inference_latency")


# ── Figure 5 — Token Usage ────────────────────────────────────────────────────

def fig_token_usage(summaries: dict) -> None:
    names = [n for n in _paper_names(summaries)
             if _val(summaries[n], "total_prompt_tokens") > 0]
    if not names:
        print("  No token data available — skipping Figure 5")
        return

    labels     = [SHORT_LABELS[n] for n in names]
    prompt     = [_val(summaries[n], "total_prompt_tokens") for n in names]
    completion = [_val(summaries[n], "total_completion_tokens") for n in names]

    x = np.arange(len(names))
    width = 0.35
    # Slightly reduced horizontal width (user request for fig_05).
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, prompt,     width, label="Prompt tokens",     color="#4C72B0")
    ax.bar(x + width / 2, completion, width, label="Completion tokens", color="#55A868", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Total Tokens (mean per run)")
    ax.set_title(_agg_title("Total Token Usage per Scenario"))
    ax.legend()
    _bar_style(ax)
    _save(fig, "fig_05_token_usage")


# ── Figure 7 — Raw + EMA Response Time Over Time (per scenario) ───────────────

def fig_ema_response_time(summaries: dict) -> None:
    """One PDF per scenario (representative run) with the chronological sort fix."""
    from datetime import datetime as _dt

    names = _paper_names(summaries)
    if not names:
        print("  No EMA timeline data available — skipping Figure 7")
        return

    generated = 0
    for name in names:
        timeline = load_ema_timeline(name)
        if not timeline:
            continue
        # --- chronological cleanse: sort by timestamp, drop duplicates ---
        timeline = sorted(timeline, key=lambda p: p.get("timestamp") or "")
        try:
            times, emas = [], []
            rt_times, rts = [], []
            viol_t, viol_v = [], []   # single unified "Violation" series
            t0, last_ts = None, None

            for point in timeline:
                ema = point.get("ema")
                if ema is None:
                    continue
                ts_raw = point.get("timestamp", "")
                if ts_raw == last_ts:          # de-duplicate identical timestamps
                    continue
                last_ts = ts_raw
                ts = _dt.fromisoformat(ts_raw)
                if t0 is None:
                    t0 = ts
                elapsed = (ts - t0).total_seconds() / 60
                times.append(elapsed)
                emas.append(ema)
                rt_val = point.get("rt")
                if rt_val is not None:
                    rt_times.append(elapsed)
                    rts.append(rt_val)

                vtype = point.get("violation")
                if not point.get("grace_period") and not point.get("cooldown"):
                    if vtype in ("UPPER_THRESHOLD_EXCEEDED", "LOWER_THRESHOLD_EXCEEDED"):
                        viol_t.append(elapsed)
                        viol_v.append(ema)

            if not times:
                continue

            fig, ax = plt.subplots(figsize=(10, 4))
            if rts:
                ax.plot(rt_times, rts, color="#AAAAAA", linewidth=0.8, alpha=0.55,
                        zorder=1, label="Raw RT (per request)")
            ax.plot(times, emas, color="#4C72B0", linewidth=1.5, alpha=0.9,
                    zorder=3, label="EMA RT")
            if viol_t:
                ax.scatter(viol_t, viol_v, color="#C44E52", marker="o", s=36,
                           alpha=0.9, zorder=5, label="Violation")

            ax.axhline(UPPER_THRESHOLD, color="red",  linestyle="--", linewidth=1.2,
                       alpha=0.8, label=f"Upper threshold ({UPPER_THRESHOLD}s)")
            ax.axhline(LOWER_THRESHOLD, color="blue", linestyle="--", linewidth=1.2,
                       alpha=0.8, label=f"Lower threshold ({LOWER_THRESHOLD}s)")
            x_end = max(times) if times else 30
            ax.fill_between([0, x_end], LOWER_THRESHOLD, UPPER_THRESHOLD,
                            alpha=0.07, color="green", label="Target band")
            ax.set_xlabel("Time (minutes)")
            ax.set_ylabel("Response Time (s)")
            ax.set_title(f"Raw + EMA Response Time — {SHORT_LABELS.get(name, name)}\n"
                         f"(Representative Run)")
            ax.legend(fontsize=8, loc="upper right")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.yaxis.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            _save(fig, f"fig_07_{name}_ema_response_time")
            generated += 1
        except Exception:
            continue

    if generated == 0:
        print("  No EMA timeline files produced any data — skipping Figure 7")


# ── Figure 9 — EMA Time-in-Band ──────────────────────────────────────────────

def fig_ema_time_in_band(summaries: dict) -> None:
    names = _paper_names(summaries)
    if not names:
        print("  No data — skipping Figure 9")
        return
    labels = [SHORT_LABELS[n] for n in names]
    values = [_val(summaries[n], "ema_time_in_band_pct") for n in names]
    colors = [_scenario_color(n) for n in names]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, values, color=colors, edgecolor="white", width=0.6)
    for i, val in enumerate(values):
        if val > 0:
            ax.text(x[i], val + 1.5, f"{val:.0f}%", ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, 115)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("EMA Time-in-Band (%)")
    ax.set_title(_agg_title("EMA Time Within SLO Band [0.5s - 3.0s]"))
    _bar_style(ax)
    _save(fig, "fig_09_ema_time_in_band")


# ── Figure 14 — Mean Time-to-Recovery (MTTR) ─────────────────────────────────

def fig_mttr(summaries: dict) -> None:
    """Mean seconds to recover from an upper-threshold violation (lower is better)."""
    names = _paper_names(summaries)
    if not names:
        print("  No data — skipping Figure 14")
        return
    labels = [SHORT_LABELS[n] for n in names]
    values = [_val(summaries[n], "mttr_mean_s") or 0 for n in names]
    colors = [_scenario_color(n) for n in names]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, values, color=colors, edgecolor="white", width=0.6)
    ymax = max(values) if any(v > 0 for v in values) else 1
    for i, val in enumerate(values):
        if val > 0:
            ax.text(x[i], val + ymax * 0.02, f"{val:.0f}s", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Mean Time-to-Recovery (s)")
    ax.set_title(_agg_title("MTTR — Recovery from Upper-Threshold Violation (lower is better)"))
    _bar_style(ax)
    _save(fig, "fig_14_mttr")


# ── Figure 15 — Steady-State Time-in-Band (holding quality) ──────────────────

def fig_steady_state_tib(summaries: dict) -> None:
    """TiB measured only AFTER first recovery — holding quality, cold-start ramp excluded."""
    names = _paper_names(summaries)
    if not names:
        print("  No data — skipping Figure 15")
        return
    labels = [SHORT_LABELS[n] for n in names]
    values = [_val(summaries[n], "steady_state_tib_pct") or 0 for n in names]
    colors = [_scenario_color(n) for n in names]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, values, color=colors, edgecolor="white", width=0.6)
    for i, val in enumerate(values):
        if val > 0:
            ax.text(x[i], val + 1.5, f"{val:.0f}%", ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, 115)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Steady-State Time-in-Band (%)")
    ax.set_title(_agg_title("Steady-State EMA Time-in-Band (holding quality, post-recovery)"))
    _bar_style(ax)
    _save(fig, "fig_15_steady_state_tib")


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
    if not dirs:
        d = SCRIPT_DIR / exp_name
        if d.exists():
            dirs.append(d)
    return dirs


def _load_pod_cpu(exp_name: str, service_prefix: str = "microservice3") -> list:
    """
    Load per-pod CPU data for a service across all available run directories.
    Returns a list of dicts, one per run: {"elapsed_min": [...], "cpu_m": [...]}
    where cpu_m is the SUM of all matching pods at each timestamp (handles replicas).
    """
    runs = []
    for run_dir in _run_dirs(exp_name):
        csv_path = run_dir / "k8s_pod_resources.csv"
        if not csv_path.exists():
            continue
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
    Returns a list of dicts, one per run: {"elapsed_min": [...], "mem_mi": [...]}.
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


def _load_node_cpu(exp_name: str) -> dict:
    """
    Load per-node CPU data for a single run (run1 as representative).
    Returns {node_name: {"elapsed_min": [...], "cpu_m": [...]}}.
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
    """Interpolate runs onto a common time grid and return (grid, mean, std). grid_step in minutes."""
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
    """2×2 grid, one subplot per microservice; mean line per scenario (no std band)."""
    names = _paper_names(summaries)
    if not names:
        print("  No experiments available — skipping Figure 10")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
    axes_flat = [axes[r][c] for r in range(2) for c in range(2)]

    for ax, (svc_prefix, svc_short) in zip(axes_flat, SERVICES):
        any_data = False
        for name in names:
            runs = _load_pod_cpu(name, service_prefix=svc_prefix)
            if not runs:
                continue
            color = _scenario_color(name)
            if len(runs) >= 2:
                grid, mean, _ = _aggregate_runs(runs)
                ax.plot(grid, mean, label=SHORT_LABELS.get(name, name),
                        color=color, linewidth=1.6)
            else:
                r = runs[0]
                ax.plot(r["elapsed_min"], r["cpu_m"],
                        label=SHORT_LABELS.get(name, name), color=color, linewidth=1.6)
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

    fig.suptitle(_agg_title("Pod CPU Usage Over Time — All Services"), fontsize=12)
    plt.tight_layout()
    _save(fig, "fig_10_pod_cpu_over_time")


# ── Figure 11 — Node CPU Usage Over Time ─────────────────────────────────────

def fig_node_cpu_over_time(summaries: dict) -> None:
    """Per-node CPU over time for the key scenarios (representative run each)."""
    names = [n for n in PAPER_ORDER if n in summaries]
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

        # service_placement fires when sFlow cpu_utilization > 40% of a 4-core (4000m) node.
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

    for j in range(len(names), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Node CPU Usage Over Time (Representative Run)\n"
                 "Red dashed = service_placement trigger (40% = 1600m on 4-core nodes)", fontsize=11)
    plt.tight_layout()
    _save(fig, "fig_11_node_cpu_over_time")


# ── Figure 12 — Node Memory Usage Over Time ───────────────────────────────────

def fig_node_memory_over_time(summaries: dict) -> None:
    """Per-node RAM (MiB) over time for key scenarios (representative run each)."""
    names = [n for n in PAPER_ORDER if n in summaries]
    if not names:
        print("  No experiments available — skipping Figure 12")
        return

    NODE_CAPACITY_MIB = 8192  # 8 GiB per node (Chameleon KVM@TACC)
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

    fig.suptitle("Node RAM Usage Over Time (Representative Run)\n"
                 "Orange dashed = 80% of 8 GiB node capacity (6,554 MiB)", fontsize=11)
    plt.tight_layout()
    _save(fig, "fig_12_node_memory_over_time")


# ── Figure 13 — Pod Memory Usage Over Time (all 4 services) ──────────────────

def fig_pod_memory_over_time(summaries: dict) -> None:
    """2×2 grid, one subplot per microservice; mean line per scenario (no std band)."""
    names = _paper_names(summaries)
    if not names:
        print("  No experiments available — skipping Figure 13")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
    axes_flat = [axes[r][c] for r in range(2) for c in range(2)]

    for ax, (svc_prefix, svc_short) in zip(axes_flat, SERVICES):
        any_data = False
        for name in names:
            runs = _load_pod_memory(name, service_prefix=svc_prefix)
            if not runs:
                continue
            color = _scenario_color(name)
            if len(runs) >= 2:
                runs_compat = [{"elapsed_min": r["elapsed_min"], "cpu_m": r["mem_mi"]} for r in runs]
                grid, mean, _ = _aggregate_runs(runs_compat)
                ax.plot(grid, mean, label=SHORT_LABELS.get(name, name),
                        color=color, linewidth=1.6)
            else:
                r = runs[0]
                ax.plot(r["elapsed_min"], r["mem_mi"],
                        label=SHORT_LABELS.get(name, name), color=color, linewidth=1.6)
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

    fig.suptitle(_agg_title("Pod Memory Usage Over Time — All Services"), fontsize=12)
    plt.tight_layout()
    _save(fig, "fig_13_pod_memory_over_time")


# ── Paper-Style Figures (EMA over time + resource usage) ─────────────────────

LOAD_STAGES_SEQ = [8, 8, 20, 20, 20, 20, 20, 8, 8, 20, 20]
_STAGE_DUR_S = 120
_STAGE_BOUNDS_S = [i * _STAGE_DUR_S for i in range(len(LOAD_STAGES_SEQ))]
_EXP_TOTAL_S = 1320
_STAGE_CENTERS_S = [
    (_STAGE_BOUNDS_S[i] + (_STAGE_BOUNDS_S[i + 1] if i + 1 < len(_STAGE_BOUNDS_S) else _EXP_TOTAL_S)) / 2
    for i in range(len(LOAD_STAGES_SEQ))
]

# Per-scenario line styles for the overlaid EMA figure (paper Fig B).
_PAPERB_STYLES: dict[str, Any] = {
    "exp_01_baseline":           dict(color=SCENARIO_COLORS["exp_01_baseline"],
                                      linestyle=":",  linewidth=1.5, zorder=2, label="No Management"),
    "exp_hpa_baseline":          dict(color=SCENARIO_COLORS["exp_hpa_baseline"],
                                      linestyle="--", linewidth=1.5, zorder=3, label="HPA"),
    "exp_vpa_baseline":          dict(color=SCENARIO_COLORS["exp_vpa_baseline"],
                                      linestyle="--", linewidth=1.5, zorder=3, label="VPA"),
    "exp_hpa_vpa_baseline":      dict(color=SCENARIO_COLORS["exp_hpa_vpa_baseline"],
                                      linestyle="--", linewidth=1.5, zorder=3, label="HPA+VPA"),
    "exp_09_cloud_llm_baseline": dict(color=SCENARIO_COLORS["exp_09_cloud_llm_baseline"],
                                      linestyle="--", linewidth=1.6, zorder=4, label="ChatGPT4o"),
    "exp_08_full_system":        dict(color=SCENARIO_COLORS["exp_08_full_system"],
                                      linestyle="-",  linewidth=2.0, zorder=5, label="LATS"),
}
# Draw worst→best so the hero line lands on top; legend is re-ordered to PAPER_ORDER.
_PAPERB_DRAW_ORDER = ["exp_01_baseline", "exp_hpa_baseline", "exp_vpa_baseline",
                      "exp_hpa_vpa_baseline", "exp_09_cloud_llm_baseline",
                      "exp_08_full_system"]


def _ema_seconds_from_timeline(timeline: list) -> tuple:
    """Build (times_s, emas) from an EMA timeline, sorted by timestamp + de-duplicated."""
    from datetime import datetime as _dt
    rows = []
    for pt in timeline:
        ema = pt.get("ema")
        if ema is None:
            continue
        ts_raw = pt.get("timestamp", "")
        try:
            ts = _dt.fromisoformat(ts_raw)
        except ValueError:
            continue
        rows.append((ts, ema))
    if not rows:
        return [], []
    rows.sort(key=lambda r: r[0])               # chronological sort (fix back-and-forth)
    dedup = {}
    for ts, ema in rows:                          # keep last value per identical timestamp
        dedup[ts] = ema
    items = sorted(dedup.items(), key=lambda r: r[0])
    t0 = items[0][0]
    times = [(ts - t0).total_seconds() for ts, _ in items]
    emas = [e for _, e in items]
    return times, emas


def _load_ema_seconds(exp_name: str) -> tuple:
    """Representative-run EMA timeline (run1) → (times_s, emas)."""
    return _ema_seconds_from_timeline(load_ema_timeline(exp_name))


def _load_ema_runs_seconds(exp_name: str) -> list:
    """All runs' EMA timelines → list of (times_s, emas), each sorted/de-duplicated."""
    out = []
    for d in _run_dirs(exp_name):
        p = d / "intent_loop_log.json"
        if not p.exists():
            continue
        with open(p) as f:
            timeline = json.load(f).get("ema_timeline", [])
        t, e = _ema_seconds_from_timeline(timeline)
        if t:
            out.append((t, e))
    return out


def _aggregate_ema_runs(runs: list, grid_step: float = 5.0) -> tuple:
    """Interpolate per-run EMA onto a common second-grid, return (grid, mean)."""
    if not runs:
        return np.array([]), np.array([])
    max_t = min(max(t) for t, _ in runs)
    if max_t <= 0:
        return np.array([]), np.array([])
    grid = np.arange(0, max_t, grid_step)
    stacked = [np.interp(grid, t, e) for t, e in runs]
    return grid, np.mean(stacked, axis=0)


def _draw_load_stage_markers(ax) -> None:
    """Green vertical lines at stage boundaries + load numbers above the plot."""
    trans = ax.get_xaxis_transform()
    for bnd in _STAGE_BOUNDS_S[1:]:
        ax.axvline(bnd, color="green", linestyle="-", linewidth=0.8, alpha=0.8, zorder=1)
    for center, load in zip(_STAGE_CENTERS_S, LOAD_STAGES_SEQ):
        ax.text(center, 1.03, str(load), transform=trans,
                ha="center", va="bottom", color="green",
                fontsize=9, fontweight="bold", clip_on=False)


def fig_paper_ema_single_run(summaries: dict) -> None:
    """EMA_RT over time for LATS, representative run."""
    times, emas = _load_ema_seconds("exp_08_full_system")
    if not times:
        print("  No EMA data for exp_08_full_system — skipping Paper Fig A")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    _draw_load_stage_markers(ax)
    ax.plot(times, emas, color="#4C72B0", linewidth=1.6, label="EMA_RT", zorder=3)
    ax.axhline(UPPER_THRESHOLD, color="blue", linewidth=1.5, zorder=2)
    ax.axhline(LOWER_THRESHOLD, color="blue", linewidth=1.5, zorder=2)
    ax.text(_EXP_TOTAL_S, UPPER_THRESHOLD, "  Max", color="blue",
            ha="left", va="center", fontsize=8, clip_on=False)
    ax.text(_EXP_TOTAL_S, LOWER_THRESHOLD, "  Min", color="blue",
            ha="left", va="center", fontsize=8, clip_on=False)

    ax.set_xlim(0, _EXP_TOTAL_S)
    ax.set_ylim(0, max(max(emas) * 1.2, UPPER_THRESHOLD * 1.5))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Response Time (s)")
    ax.set_title("EMA Response Time — LATS (Representative Run)",
                 fontsize=10, pad=24)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    _save(fig, "fig_paper_A_ema_single_run")


def fig_paper_ema_all_scenarios(summaries: dict) -> None:
    """All 4 scenarios overlaid, EMA mean-aggregated across the 3 runs."""
    fig, ax = plt.subplots(figsize=(7.5, 4))
    _draw_load_stage_markers(ax)

    plotted = {}
    y_max = UPPER_THRESHOLD * 1.5
    for exp_name in _PAPERB_DRAW_ORDER:     # worst→best so hero is on top
        if exp_name not in summaries:
            continue
        runs = _load_ema_runs_seconds(exp_name)
        grid, mean = _aggregate_ema_runs(runs)
        if len(grid) == 0:
            continue
        (line,) = ax.plot(grid, mean, **_PAPERB_STYLES[exp_name])
        plotted[exp_name] = line
        y_max = max(y_max, float(np.max(mean)) * 1.1)

    if not plotted:
        print("  No EMA timeline data — skipping Paper Fig B")
        plt.close(fig)
        return

    ax.axhline(UPPER_THRESHOLD, color="blue", linewidth=1.5, zorder=1)
    ax.axhline(LOWER_THRESHOLD, color="blue", linewidth=1.5, zorder=1)
    ax.text(_EXP_TOTAL_S, UPPER_THRESHOLD, "  Max", color="blue",
            ha="left", va="center", fontsize=8, clip_on=False)
    ax.text(_EXP_TOTAL_S, LOWER_THRESHOLD, "  Min", color="blue",
            ha="left", va="center", fontsize=8, clip_on=False)

    ax.set_xlim(0, _EXP_TOTAL_S)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Response Time (s)")
    ax.set_title(_agg_title("EMA Response Time — All Scenarios"), fontsize=10, pad=24)

    # Legend in deterministic paper order.
    handles = [plotted[n] for n in PAPER_ORDER if n in plotted]
    labels = [_PAPERB_STYLES[n]["label"] for n in PAPER_ORDER if n in plotted]
    ax.legend(handles, labels, loc="upper right", fontsize=8, framealpha=0.85)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    _save(fig, "fig_paper_B_ema_all_scenarios")


def _total_mean_resources(exp_name: str) -> tuple:
    """Time-averaged CPU (cores) + Memory (MiB) summed across all 4 microservice pods."""
    total_cpu, total_mem, has_data = 0.0, 0.0, False
    for svc_prefix, _ in SERVICES:
        cpu_runs = _load_pod_cpu(exp_name, service_prefix=svc_prefix)
        mem_runs = _load_pod_memory(exp_name, service_prefix=svc_prefix)
        if cpu_runs:
            total_cpu += np.mean([np.mean(r["cpu_m"]) for r in cpu_runs]) / 1000
            has_data = True
        if mem_runs:
            total_mem += np.mean([np.mean(r["mem_mi"]) for r in mem_runs])
    return (total_cpu, total_mem) if has_data else (None, None)


def fig_paper_resource_usage(summaries: dict) -> None:
    """Total normalised CPU (cores) + Memory (MiB) per scenario, dual y-axis.
    Order: LATS, ChatGPT4o, HPA, VPA, HPA+VPA, No Management."""
    order = [
        ("exp_08_full_system",        "LATS"),
        ("exp_09_cloud_llm_baseline", "ChatGPT4o"),
        ("exp_hpa_baseline",          "HPA"),
        ("exp_vpa_baseline",          "VPA"),
        ("exp_hpa_vpa_baseline",      "HPA+\nVPA"),
        ("exp_01_baseline",           "No\nManagement"),
    ]
    labels, cpu_vals, mem_vals = [], [], []
    for exp_name, label in order:
        if exp_name not in summaries:
            continue
        cpu, mem = _total_mean_resources(exp_name)
        if cpu is None:
            continue
        labels.append(label)
        cpu_vals.append(cpu)
        mem_vals.append(mem)

    if not labels:
        print("  No k8s resource data (k8s_pod_resources.csv) — skipping Paper Fig C")
        return

    x = np.arange(len(labels))
    width = 0.35
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - width / 2, cpu_vals, width,
                    facecolor="#E05C40", hatch="..", edgecolor="black", linewidth=0.5,
                    label="CPU (core)", zorder=3)
    bars2 = ax2.bar(x + width / 2, mem_vals, width,
                    facecolor="#3DBFBF", hatch="//", edgecolor="black", linewidth=0.5,
                    label="Mem (MiB)", zorder=3)

    for val, bar in zip(cpu_vals, bars1):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + max(cpu_vals) * 0.02,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    for val, bar in zip(mem_vals, bars2):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + max(mem_vals) * 0.02,
                 f"{int(val)}", ha="center", va="bottom", fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("CPU (core)", color="#E05C40")
    ax1.tick_params(axis="y", labelcolor="#E05C40")
    ax2.set_ylabel("Mem (MiB)", color="#1A9A9A")
    ax2.tick_params(axis="y", labelcolor="#1A9A9A")
    ax1.set_ylim(0, max(cpu_vals) * 1.3)
    ax2.set_ylim(0, max(mem_vals) * 1.3)
    ax1.set_title(_agg_title("Normalised Resource Usage per Scenario"), fontsize=10)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax1.set_axisbelow(True)
    plt.tight_layout()
    _save(fig, "fig_paper_C_resource_usage")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(description="Generate comparative paper figures.")
    parser.add_argument("--out-dir", default=None,
                        help="Directory to write the PDF figures into (default: script dir). "
                             "Input data is always read from the script dir.")
    args = parser.parse_args()
    if args.out_dir:
        OUTPUT_DIR = Path(args.out_dir)
        if not OUTPUT_DIR.is_absolute():
            OUTPUT_DIR = SCRIPT_DIR / OUTPUT_DIR
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summaries = load_summaries()
    if not summaries:
        print("No summary.json files found in sibling exp_*/ directories.")
        print("Run experiments first with:  python3 run_suite.py")
        sys.exit(1)

    print(f"Loaded summaries for: {list(summaries.keys())}")
    print(f"Writing figures to: {OUTPUT_DIR}\n")

    steps = [
        ("Figure 1  — Intent Satisfaction Rate",                fig_intent_satisfaction_rate),
        ("Figure 2  — Violation Resolution",                    fig_violation_resolution),
        ("Figure 3  — Action Type Distribution",                fig_action_type_distribution),
        ("Figure 4  — Inference Latency",                       fig_inference_latency),
        ("Figure 5  — Token Usage",                             fig_token_usage),
        ("Figure 7  — Raw + EMA Response Time (per scenario)",  fig_ema_response_time),
        ("Figure 9  — EMA Time-in-Band (headline SLO metric)",  fig_ema_time_in_band),
        ("Figure 14 — MTTR (headline responsiveness metric)",   fig_mttr),
        ("Figure 15 — Steady-State Time-in-Band (holding quality)", fig_steady_state_tib),
        ("Figure 10 — Pod CPU Over Time (all services)",        fig_pod_cpu_over_time),
        ("Figure 11 — Node CPU Over Time",                      fig_node_cpu_over_time),
        ("Figure 12 — Node Memory Over Time",                   fig_node_memory_over_time),
        ("Figure 13 — Pod Memory Over Time (all services)",     fig_pod_memory_over_time),
        ("Paper Fig A — EMA Single Run, LATS", fig_paper_ema_single_run),
        ("Paper Fig B — EMA All Scenarios (mean-aggregated)",   fig_paper_ema_all_scenarios),
        ("Paper Fig C — Normalised Resource Usage",             fig_paper_resource_usage),
    ]

    for title, fn in steps:
        print(f"Generating {title}...")
        fn(summaries)

    print("\nAll figures saved.")


if __name__ == "__main__":
    main()
