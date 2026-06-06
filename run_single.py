#!/usr/bin/env python3
"""
Single-Run Launcher for Targeted Scenario Evaluation

Triggers exactly one experiment run. The cluster must already be reset and
verified (via cluster_reset_verify.py) before calling this script.

Usage:
    python3 run_single.py --scenario baseline
    python3 run_single.py --scenario full
    python3 run_single.py --scenario cloud
    python3 run_single.py --scenario hpa
    python3 run_single.py --scenario full \\
        --load-pattern 15,30,45,60,45,30,15 \\
        --results-root evaluation_results/5th_experiment
    python3 run_single.py --scenario baseline --run-id 1 --results-root evaluation_results/5th_experiment
    python3 run_single.py --scenario baseline --run-id 2 --results-root evaluation_results/5th_experiment
    # ↑ run-id 2 == total-runs default (2) → auto-aggregates to exp_01_baseline_agg/

Scenarios:
    baseline  ->  exp_01_baseline              (monitor-only, no LLM)
    full      ->  exp_08_full_system            (all actions, local Qwen3.5:4b)
    cloud     ->  exp_09_cloud_llm_baseline     (all actions, GPT-4o via OpenAI API)
    hpa       ->  exp_hpa_baseline              (K8s HPA @ 40% CPU, no intent loop)

For the 'cloud' scenario, export OPENAI_API_KEY before running:
    export OPENAI_API_KEY=sk-...
    python3 run_single.py --scenario cloud
"""

import argparse
import os
import sys
import time

import run_ablation
from run_ablation import EXPERIMENT_MATRIX, aggregate_runs
from run_experiment import get_master_info, ssh_cmd

# ── Scenario registry ────────────────────────────────────────────────────────

SCENARIOS = {
    "baseline": "exp_01_baseline",
    "full":     "exp_08_full_system",
    "cloud":    "exp_09_cloud_llm_baseline",
    "hpa":      "exp_hpa_baseline",
}

# Inline exp_def for HPA — not in EXPERIMENT_MATRIX so run_ablation.py stays untouched.
# monitor_only=True: intent loop records EMA and detects violations but takes zero actions.
# K8s HPA is solely responsible for all scaling decisions.
HPA_EXP_DEF = {
    "label": "HPA Baseline (K8s Native Autoscaler)",
    "monitor_only": True,
    "actions": {
        "horizontal_scaling": False,
        "vertical_scaling": False,
        "service_placement": False,
        "flow_scheduling": False,
    },
}

DEFAULT_RESULTS_ROOT = "evaluation_results/6th_experiment"


# ── HPA lifecycle helpers ────────────────────────────────────────────────────

def _create_hpa(user: str, master: str) -> None:
    """Create HPA objects (40% CPU target) for all 4 deployments on the K8s master."""
    hpa_specs = [
        ("microservice1-deployment", 1, 5),
        ("microservice2-deployment", 1, 5),
        ("microservice3-deployment", 1, 5),
        ("microservice4-deployment", 1, 3),
    ]
    print("  Creating HPA objects (40% CPU target)...")
    for dep, min_rep, max_rep in hpa_specs:
        cmd = (
            f"kubectl autoscale deployment {dep}"
            f" --cpu-percent=40 --min={min_rep} --max={max_rep}"
        )
        result = ssh_cmd(user, master, cmd)
        status = "created" if result.returncode == 0 else f"FAILED: {result.stderr.strip()}"
        print(f"  HPA {dep}: {status}")
    print("  Waiting 15s for HPA controller to initialise and read first metrics sample...")
    time.sleep(15)


def _delete_hpa(user: str, master: str) -> None:
    """Remove all HPA objects from the default namespace."""
    result = ssh_cmd(user, master, "kubectl delete hpa --all -n default 2>/dev/null || true")
    msg = result.stdout.strip() or "none found"
    print(f"  HPA objects deleted: {msg}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one experiment scenario manually (cluster must be pre-reset)."
    )
    parser.add_argument(
        "--scenario", choices=list(SCENARIOS.keys()), required=True,
        help="Which scenario to run: baseline | full | cloud | hpa"
    )
    parser.add_argument(
        "--results-root", default=DEFAULT_RESULTS_ROOT,
        help=f"Root output directory (default: {DEFAULT_RESULTS_ROOT})"
    )
    parser.add_argument(
        "--load-pattern", default=None,
        help="Comma-separated Locust user counts per stage, e.g. 15,30,45,60,45,30,15"
    )
    parser.add_argument(
        "--debug-llm", action="store_true",
        help="Enable LLM debug logging in the intent loop."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would run without executing."
    )
    parser.add_argument(
        "--run-id", type=int, default=None, metavar="N",
        help="Which repeat number this is (1-based). Output goes to {exp_key}_runN/. "
             "When run-id == total-runs, auto-aggregates all runs into {exp_key}_agg/."
    )
    parser.add_argument(
        "--total-runs", type=int, default=3, metavar="N",
        help="Total number of repeats planned for this scenario. "
             "Aggregation fires automatically when run-id == total-runs (default: 2)."
    )
    args = parser.parse_args()

    is_hpa = (args.scenario == "hpa")

    # HPA uses an inline exp_def; all other scenarios pull from EXPERIMENT_MATRIX.
    exp_key = SCENARIOS[args.scenario]
    if is_hpa:
        exp_def = HPA_EXP_DEF
    else:
        exp_def = EXPERIMENT_MATRIX.get(exp_key)
        if exp_def is None:
            print(f"ERROR: '{exp_key}' not found in EXPERIMENT_MATRIX.")
            sys.exit(1)

    # Inject custom load pattern if provided.
    if args.load_pattern:
        exp_def = dict(exp_def)
        exp_def["load_pattern"] = [int(x) for x in args.load_pattern.split(",")]

    # Determine output directory name.
    # With --run-id: {exp_key}_run{N}/  (e.g. exp_01_baseline_run1/)
    # Without: {exp_key}/  (single-run, backward-compatible)
    run_id = args.run_id
    run_name = f"{exp_key}_run{run_id}" if run_id is not None else exp_key

    results_root = args.results_root

    # Redirect all output paths inside run_ablation to the chosen results root.
    run_ablation.RESULTS_ROOT = results_root
    os.makedirs(results_root, exist_ok=True)

    load_pattern = exp_def.get("load_pattern", run_ablation.DEFAULT_LOAD_PATTERN)
    run_label = f"run {run_id}/{args.total_runs}" if run_id is not None else "single run"
    print(f"\n{'='*70}")
    print(f"  SCENARIO     : {args.scenario}  ({exp_key})")
    print(f"  RUN          : {run_label}  →  {run_name}/")
    print(f"  LOAD PATTERN : {load_pattern}  (peak {max(load_pattern)} users)")
    print(f"  OUTPUT       : {os.path.join(results_root, run_name)}/")
    print(f"  NOTE         : Cluster reset must already be verified before this step.")
    print(f"{'='*70}")

    if args.dry_run:
        print("  [DRY RUN] — no experiment executed.")
        return

    # For HPA: create objects before the run, delete after.
    hpa_user = hpa_master = None
    if is_hpa:
        hpa_user, hpa_master = get_master_info("config.yaml")
        _create_hpa(hpa_user, hpa_master)

    run_ablation.run_experiment(
        name=run_name,
        exp_def=exp_def,
        dry_run=False,
        skip_reset=True,
        debug_llm=args.debug_llm,
    )

    # Auto-aggregate when this is the final planned run.
    if run_id is not None and run_id == args.total_runs:
        run_dirs = [
            os.path.join(results_root, f"{exp_key}_run{i}")
            for i in range(1, args.total_runs + 1)
            if os.path.isdir(os.path.join(results_root, f"{exp_key}_run{i}"))
        ]
        if run_dirs:
            agg_dir = os.path.join(results_root, f"{exp_key}_agg")
            aggregate_runs(run_dirs, agg_dir, exp_key, exp_def["label"])

    if is_hpa:
        print("\n  Removing HPA objects...")
        _delete_hpa(hpa_user, hpa_master)

    print(f"\n  ✅ Run complete — results in {os.path.join(results_root, run_name)}/")
    if run_id is not None and run_id == args.total_runs:
        print(f"     Aggregated → {os.path.join(results_root, exp_key + '_agg')}/")
    print(f"     Run plot_comparison.py inside {results_root}/ to generate figures.")


if __name__ == "__main__":
    main()
