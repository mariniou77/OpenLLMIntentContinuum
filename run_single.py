#!/usr/bin/env python3
"""
Single-Run Launcher for the Targeted 3-Scenario Evaluation

Triggers exactly one experiment run. The cluster must already be reset and
verified (via cluster_reset_verify.py) before calling this script.

When --run-id 3 is passed, aggregation over run1/run2/run3 is performed
automatically after the run completes, producing {exp_key}_agg/summary_aggregated.json
identical to the --repeat 3 output from run_ablation.py.

Usage:
    python3 run_single.py --scenario baseline --run-id 1
    python3 run_single.py --scenario full     --run-id 2
    python3 run_single.py --scenario cloud    --run-id 3
    python3 run_single.py --scenario hpa      --run-id 1
    python3 run_single.py --scenario full     --run-id 1 \\
        --results-root evaluation_results/3rd_experiment

Scenarios:
    baseline  ->  exp_01_baseline              (monitor-only, no LLM)
    full      ->  exp_08_full_system            (all actions, local Qwen3.5:4b)
    cloud     ->  exp_09_cloud_llm_baseline     (all actions, GPT-4o via OpenAI API)
    hpa       ->  exp_hpa_baseline              (K8s HPA @ 70% CPU, no intent loop)

For the 'cloud' scenario, export OPENAI_API_KEY before running:
    export OPENAI_API_KEY=sk-...
    python3 run_single.py --scenario cloud --run-id 1
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

DEFAULT_RESULTS_ROOT = "evaluation_results/3rd_experiment"


# ── HPA lifecycle helpers ────────────────────────────────────────────────────

def _create_hpa(user: str, master: str) -> None:
    """Create HPA objects (70% CPU target) for all 4 deployments on the K8s master."""
    hpa_specs = [
        ("microservice1-deployment", 1, 5),
        ("microservice2-deployment", 1, 5),
        ("microservice3-deployment", 1, 5),
        ("microservice4-deployment", 1, 3),
    ]
    print("  Creating HPA objects (70% CPU target)...")
    for dep, min_rep, max_rep in hpa_specs:
        cmd = (
            f"kubectl autoscale deployment {dep}"
            f" --cpu-percent=70 --min={min_rep} --max={max_rep}"
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
        "--run-id", type=int, choices=[1, 2, 3], required=True,
        help="Run number (1, 2, or 3). Running id=3 auto-aggregates all three runs."
    )
    parser.add_argument(
        "--results-root", default=DEFAULT_RESULTS_ROOT,
        help=f"Root output directory (default: {DEFAULT_RESULTS_ROOT})"
    )
    parser.add_argument(
        "--debug-llm", action="store_true",
        help="Enable LLM debug logging in the intent loop."
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

    run_name = f"{exp_key}_run{args.run_id}"
    results_root = args.results_root

    # Redirect all output paths inside run_ablation to the chosen results root.
    run_ablation.RESULTS_ROOT = results_root
    os.makedirs(results_root, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  SCENARIO : {args.scenario}  ({exp_key})")
    print(f"  RUN ID   : {args.run_id} / 3")
    print(f"  OUTPUT   : {os.path.join(results_root, run_name)}/")
    print(f"  NOTE     : Cluster reset must already be verified before this step.")
    print(f"{'='*70}")

    # For HPA: create objects before the run, delete them after.
    hpa_user = hpa_master = None
    if is_hpa:
        hpa_user, hpa_master = get_master_info("config.yaml")
        _create_hpa(hpa_user, hpa_master)

    # Run the experiment with skip_reset=True — user already reset and verified.
    run_ablation.run_experiment(
        name=run_name,
        exp_def=exp_def,
        dry_run=False,
        skip_reset=True,
        debug_llm=args.debug_llm,
    )

    if is_hpa:
        print("\n  Removing HPA objects...")
        _delete_hpa(hpa_user, hpa_master)

    # ── Auto-aggregate after the third run ───────────────────────────────────
    if args.run_id == 3:
        print(f"\n{'='*70}")
        print(f"  Run 3 complete — aggregating all 3 runs for {exp_key} …")
        print(f"{'='*70}")

        run_dirs = [
            os.path.join(results_root, f"{exp_key}_run{i}")
            for i in [1, 2, 3]
        ]
        agg_dir = os.path.join(results_root, f"{exp_key}_agg")

        # Warn about any missing run directories so the user knows what's included.
        for d in run_dirs:
            summary = os.path.join(d, "summary.json")
            if not os.path.exists(summary):
                print(f"  ⚠  Missing {summary} — that run will be excluded from aggregation.")

        aggregate_runs(run_dirs, agg_dir, exp_key, exp_def["label"])
        print(f"\n  ✅ Aggregated results → {agg_dir}/summary_aggregated.json")
        print(f"     Run plot_comparison.py inside {results_root}/ to generate figures.")


if __name__ == "__main__":
    main()
