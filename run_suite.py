#!/usr/bin/env python3
"""
Automated Experiment Suite Runner

Runs all 4 scenarios × TOTAL_RUNS repeats with automatic cluster reset+verification
between every single run. Fails hard if the cluster cannot be verified after
MAX_VERIFY_RETRIES attempts.

Usage:
    python3 run_suite.py
    python3 run_suite.py --dry-run
    python3 run_suite.py --debug-llm
    python3 run_suite.py --results-root evaluation_results/7th_experiment
    python3 run_suite.py --skip-cloud      # if no OPENAI_API_KEY available

For the cloud scenario, export the API key first:
    export OPENAI_API_KEY=sk-...
    python3 run_suite.py
"""

import argparse
import os
import sys
import time

import yaml

import run_ablation
from run_ablation import EXPERIMENT_MATRIX, aggregate_runs, wait_for_cluster_stable
from cluster_reset_verify import verify_cluster
from run_experiment import get_master_info, reset_cluster, ssh_cmd
from run_single import SCENARIOS, HPA_EXP_DEF, _create_hpa, _delete_hpa

DEFAULT_RESULTS_ROOT = "evaluation_results/7th_experiment"
TOTAL_RUNS = 3
MAX_VERIFY_RETRIES = 3
VERIFY_RETRY_SLEEP_S = 120
CONFIG_PATH = "config.yaml"

SUITE_ORDER = ["full", "cloud", "hpa", "baseline"]


def _reset_and_verify(config_path: str, max_retries: int, retry_sleep_s: int) -> bool:
    """Reset cluster to defaults, wait for stability, then verify. Retry up to max_retries."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    master_ip = cfg["endpoints"]["kubernetes_master"]
    ssh_user = cfg["endpoints"].get("kubernetes_user", "cc")
    user, master = get_master_info(config_path)

    for attempt in range(1, max_retries + 1):
        print(f"\n--- Reset+Verify (attempt {attempt}/{max_retries}) ---")

        reset_cluster(config_path)

        # Purge any leftover HPA objects before verify (verify checks for 0 HPA)
        result = ssh_cmd(user, master, "kubectl delete hpa --all -n default 2>/dev/null || true")
        leftover = result.stdout.strip()
        if leftover:
            print(f"  HPA purged: {leftover}")
        else:
            print("  No leftover HPA objects")

        wait_for_cluster_stable(master_ip=master_ip, ssh_user=ssh_user)

        if verify_cluster(config_path):
            return True

        if attempt < max_retries:
            print(f"\n  VERIFY FAILED — sleeping {retry_sleep_s}s before retry {attempt + 1}...")
            time.sleep(retry_sleep_s)

    return False


def run_suite(results_root: str, total_runs: int, dry_run: bool,
              debug_llm: bool, skip_cloud: bool) -> None:
    os.makedirs(results_root, exist_ok=True)
    run_ablation.RESULTS_ROOT = results_root

    if not skip_cloud and not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not set. Cloud scenario will be skipped.")
        print("  Set it with: export OPENAI_API_KEY=sk-...")
        skip_cloud = True

    user, master = get_master_info(CONFIG_PATH)

    completed: list = []
    skipped: list = []

    for scenario_name in SUITE_ORDER:
        if scenario_name == "cloud" and skip_cloud:
            print(f"\n{'=' * 70}")
            print(f"  SKIPPING: cloud (no OPENAI_API_KEY)")
            print(f"{'=' * 70}")
            skipped.append("cloud")
            continue

        exp_key = SCENARIOS[scenario_name]
        is_hpa = (scenario_name == "hpa")
        exp_def = HPA_EXP_DEF if is_hpa else EXPERIMENT_MATRIX.get(exp_key)

        if exp_def is None:
            print(f"ERROR: '{exp_key}' not found in EXPERIMENT_MATRIX — skipping scenario")
            skipped.append(scenario_name)
            continue

        for run_id in range(1, total_runs + 1):
            run_name = f"{exp_key}_run{run_id}"
            run_dir = os.path.join(results_root, run_name)
            summary_path = os.path.join(run_dir, "summary.json")

            # Resume support: skip already-completed runs so the suite can be restarted
            if os.path.exists(summary_path):
                print(f"\n  SKIP (already completed): {run_name}")
                completed.append(run_name)
                continue

            print(f"\n{'=' * 70}")
            print(f"  SCENARIO  : {scenario_name}  ({exp_key})")
            print(f"  RUN       : {run_id}/{total_runs}  →  {run_name}/")
            print(f"  OUTPUT    : {os.path.join(results_root, run_name)}/")
            print(f"{'=' * 70}")

            if dry_run:
                print("  [DRY RUN] — skipping reset, verify, and experiment execution.")
                continue

            # Hard gate: full reset+verify before every single run
            if not _reset_and_verify(CONFIG_PATH, MAX_VERIFY_RETRIES, VERIFY_RETRY_SLEEP_S):
                print(
                    f"\n  ABORT: Cluster verification failed after {MAX_VERIFY_RETRIES} "
                    f"attempts before {run_name}. Stopping suite."
                )
                sys.exit(1)

            if is_hpa:
                _create_hpa(user, master)

            run_ablation.run_experiment(
                name=run_name,
                exp_def=exp_def,
                dry_run=False,
                skip_reset=True,
                debug_llm=debug_llm,
            )
            completed.append(run_name)

            if is_hpa:
                print("\n  Removing HPA objects after run...")
                _delete_hpa(user, master)

        # Aggregate once all runs for this scenario are done
        run_dirs = [
            os.path.join(results_root, f"{exp_key}_run{i}")
            for i in range(1, total_runs + 1)
            if os.path.exists(os.path.join(results_root, f"{exp_key}_run{i}", "summary.json"))
        ]
        if len(run_dirs) == total_runs:
            agg_dir = os.path.join(results_root, f"{exp_key}_agg")
            aggregate_runs(run_dirs, agg_dir, exp_key, exp_def["label"])
        elif run_dirs:
            print(
                f"  WARNING: Only {len(run_dirs)}/{total_runs} runs complete — "
                f"skipping aggregation for {exp_key}"
            )

    print(f"\n{'=' * 70}")
    print(f"  SUITE COMPLETE")
    print(f"  Completed : {len(completed)} runs")
    if skipped:
        print(f"  Skipped   : {skipped}")
    print(f"  Results   : {results_root}/")
    print(f"  Next step : python3 plot_comparison.py  (run from inside {results_root}/)")
    print(f"{'=' * 70}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full experiment suite (4 scenarios × 3 runs) with automatic "
            "cluster reset+verification between every run."
        )
    )
    parser.add_argument(
        "--results-root", default=DEFAULT_RESULTS_ROOT,
        help=f"Root output directory (default: {DEFAULT_RESULTS_ROOT})"
    )
    parser.add_argument(
        "--total-runs", type=int, default=TOTAL_RUNS,
        help=f"Repeats per scenario (default: {TOTAL_RUNS})"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the planned run order without executing anything.")
    parser.add_argument("--debug-llm", action="store_true",
                        help="Enable LLM debug logging in the intent loop.")
    parser.add_argument("--skip-cloud", action="store_true",
                        help="Skip the cloud LLM scenario entirely.")
    args = parser.parse_args()

    run_suite(
        results_root=args.results_root,
        total_runs=args.total_runs,
        dry_run=args.dry_run,
        debug_llm=args.debug_llm,
        skip_cloud=args.skip_cloud,
    )


if __name__ == "__main__":
    main()
