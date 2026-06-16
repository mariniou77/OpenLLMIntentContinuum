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
    hpa       ->  exp_hpa_baseline              (K8s HPA @ 50% CPU, no intent loop)
    vpa       ->  exp_vpa_baseline              (K8s VPA Auto, CPU+mem, no intent loop)
    hpa_vpa   ->  exp_hpa_vpa_baseline          (K8s HPA@CPU + VPA@memory, no intent loop)

Note: vpa / hpa_vpa require the VPA admission controller installed on the cluster
(autoscaler hack/vpa-up.sh) with the updater patched --min-replicas=1.

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
    "vpa":      "exp_vpa_baseline",
    "hpa_vpa":  "exp_hpa_vpa_baseline",
}

# Inline exp_defs for the K8s-native autoscaler baselines — not in EXPERIMENT_MATRIX
# so run_ablation.py stays untouched.
# monitor_only=True: the intent loop records EMA and detects violations but takes zero
# actions. The native K8s autoscaler(s) are solely responsible for all scaling decisions.
_NO_ACTIONS = {
    "horizontal_scaling": False,
    "vertical_scaling": False,
    "service_placement": False,
    "flow_scheduling": False,
}

HPA_EXP_DEF = {
    "label": "HPA Baseline (K8s Native Autoscaler)",
    "monitor_only": True,
    "actions": dict(_NO_ACTIONS),
}

# Pure VPA: updateMode "Auto" (evict & recreate), VPA manages CPU+memory requests.
VPA_EXP_DEF = {
    "label": "VPA Baseline (K8s Vertical Autoscaler)",
    "monitor_only": True,
    "actions": dict(_NO_ACTIONS),
}

# HPA+VPA co-orchestration: VPA manages memory only (controlledResources: [memory]) so
# it never fights HPA over CPU; HPA scales replicas on CPU @50% (the existing pattern).
HPA_VPA_EXP_DEF = {
    "label": "HPA+VPA Baseline (K8s Horizontal+Vertical)",
    "monitor_only": True,
    "actions": dict(_NO_ACTIONS),
}

DEFAULT_RESULTS_ROOT = "evaluation_results/11th_experiment"


# ── HPA lifecycle helpers ────────────────────────────────────────────────────

def _create_hpa(user: str, master: str) -> None:
    """Create responsive HPA objects (50% CPU target) for all 4 deployments.

    Exp13: uses autoscaling/v2 with an explicit behavior block — scaleDown
    stabilizationWindowSeconds=30 (vs the controller-manager default 300s) and a fast
    scaleUp — so HPA reacts within the 24-min run, symmetric with the responsive VPA
    recommender. ms3 max is capped at 3 to match config.yaml (worker1 capacity).
    """
    # (deployment, min_replicas, max_replicas)
    hpa_specs = [
        ("microservice1-deployment", 1, 5),
        ("microservice2-deployment", 1, 5),
        ("microservice3-deployment", 1, 3),
        ("microservice4-deployment", 1, 3),
    ]
    print("  Creating responsive HPA objects (50% CPU, scaleDown stabilization 30s)...")
    docs = []
    for dep, min_rep, max_rep in hpa_specs:
        docs.append(f"""---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {dep}
  namespace: default
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {dep}
  minReplicas: {min_rep}
  maxReplicas: {max_rep}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 50
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 30
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15""")
    manifest = "\n".join(docs)
    cmd = "cat <<'EOF' | kubectl apply -f -\n" + manifest + "\nEOF"
    result = ssh_cmd(user, master, cmd)
    if result.returncode == 0:
        print("  HPA objects: " + (result.stdout.strip() or "created"))
    else:
        print(f"  HPA objects: FAILED: {result.stderr.strip()}")
    print("  Waiting 15s for HPA controller to initialise and read first metrics sample...")
    time.sleep(15)


def _delete_hpa(user: str, master: str) -> None:
    """Remove all HPA objects from the default namespace."""
    result = ssh_cmd(user, master, "kubectl delete hpa --all -n default 2>/dev/null || true")
    msg = result.stdout.strip() or "none found"
    print(f"  HPA objects deleted: {msg}")


# ── VPA lifecycle helpers ─────────────────────────────────────────────────────

# Deployments under autoscaler control (ms1–ms4; db excluded), mirroring _create_hpa.
# The application container the harness sizes via `kubectl set resources -c nginx`.
_VPA_DEPLOYMENTS = [
    "microservice1-deployment",
    "microservice2-deployment",
    "microservice3-deployment",
    "microservice4-deployment",
]
_VPA_CONTAINER = "nginx"


def _vpa_manifest(controlled_resources=None) -> str:
    """Build a multi-doc VerticalPodAutoscaler manifest (one VPA per deployment).

    updateMode "Auto" lets the VPA updater evict & recreate pods to apply new requests.
    When `controlled_resources=["memory"]` is given, VPA manages memory only and leaves
    CPU to HPA (the HPA+VPA co-orchestration case); otherwise it manages CPU+memory.
    """
    controlled_line = ""
    if controlled_resources:
        items = "".join(f"        - {r}\n" for r in controlled_resources)
        controlled_line = "      controlledResources:\n" + items
    docs = []
    for dep in _VPA_DEPLOYMENTS:
        docs.append(
            f"""---
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: {dep}-vpa
  namespace: default
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {dep}
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: {_VPA_CONTAINER}
      minAllowed:
        cpu: 200m
        memory: 256Mi
      maxAllowed:
        cpu: 1200m
        memory: 2048Mi
{controlled_line}"""
        )
    return "\n".join(docs)


def _create_vpa(user: str, master: str, controlled_resources=None) -> None:
    """Create VPA objects (updateMode Auto) for all 4 deployments on the K8s master.

    Pass controlled_resources=["memory"] for the HPA+VPA case so VPA never fights HPA
    over CPU.
    """
    scope = ",".join(controlled_resources) if controlled_resources else "cpu+memory"
    print(f"  Creating VPA objects (updateMode=Auto, manages {scope})...")
    manifest = _vpa_manifest(controlled_resources)
    cmd = "cat <<'EOF' | kubectl apply -f -\n" + manifest + "\nEOF"
    result = ssh_cmd(user, master, cmd)
    if result.returncode == 0:
        print("  VPA objects: " + (result.stdout.strip() or "created"))
    else:
        print(f"  VPA objects: FAILED: {result.stderr.strip()}")
    print("  Waiting 15s for the VPA recommender to produce a first recommendation...")
    time.sleep(15)


def _delete_vpa(user: str, master: str) -> None:
    """Remove all VPA objects from the default namespace."""
    result = ssh_cmd(user, master, "kubectl delete vpa --all -n default 2>/dev/null || true")
    msg = result.stdout.strip() or "none found"
    print(f"  VPA objects deleted: {msg}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one experiment scenario manually (cluster must be pre-reset)."
    )
    parser.add_argument(
        "--scenario", choices=list(SCENARIOS.keys()), required=True,
        help="Which scenario to run: baseline | full | cloud | hpa | vpa | hpa_vpa"
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

    # K8s-native autoscaler scenarios use inline exp_defs and a create/delete lifecycle.
    is_hpa = args.scenario in ("hpa", "hpa_vpa")  # needs HPA objects
    is_vpa = args.scenario in ("vpa", "hpa_vpa")  # needs VPA objects

    # The native-autoscaler scenarios use inline exp_defs; everything else pulls from
    # EXPERIMENT_MATRIX.
    inline_defs = {
        "hpa":     HPA_EXP_DEF,
        "vpa":     VPA_EXP_DEF,
        "hpa_vpa": HPA_VPA_EXP_DEF,
    }
    exp_key = SCENARIOS[args.scenario]
    if args.scenario in inline_defs:
        exp_def = inline_defs[args.scenario]
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

    # For HPA/VPA scenarios: create autoscaler objects before the run, delete after.
    # HPA+VPA restricts VPA to memory so it never fights HPA over the CPU signal.
    as_user = as_master = None
    if is_hpa or is_vpa:
        as_user, as_master = get_master_info("config.yaml")
        if is_hpa:
            _create_hpa(as_user, as_master)
        if is_vpa:
            controlled = ["memory"] if args.scenario == "hpa_vpa" else None
            _create_vpa(as_user, as_master, controlled_resources=controlled)

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

    if is_hpa or is_vpa:
        print("\n  Removing autoscaler objects...")
        assert as_user is not None and as_master is not None
        if is_hpa:
            _delete_hpa(as_user, as_master)
        if is_vpa:
            _delete_vpa(as_user, as_master)

    print(f"\n  ✅ Run complete — results in {os.path.join(results_root, run_name)}/")
    if run_id is not None and run_id == args.total_runs:
        print(f"     Aggregated → {os.path.join(results_root, exp_key + '_agg')}/")
    print(f"     Run plot_comparison.py inside {results_root}/ to generate figures.")


if __name__ == "__main__":
    main()
