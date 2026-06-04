#!/usr/bin/env python3
"""
Ablation Study & Baseline Comparison Runner

Orchestrates all 9 evaluation experiments under identical Locust load conditions.
Each run is isolated in evaluation_results/exp_<name>/ and exports uniform telemetry.

Usage:
    python3 run_ablation.py --list                            # Show experiment matrix
    python3 run_ablation.py --experiment exp_02_vertical_only # Run one experiment
    python3 run_ablation.py --all                             # Run all 8 (sequential)
    python3 run_ablation.py --dry-run --all                   # Preview without executing
    python3 run_ablation.py --skip-reset --experiment exp_08  # Skip cluster reset
"""

import argparse
import copy
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

# ── Reuse helpers from run_experiment.py ────────────────────────────────────
from run_experiment import (
    reset_cluster,
    warmup_llm,
    start_locust_locally,
    stop_locust_locally,
    run_load_schedule,
    collect_summary,
    DEFAULT_LOAD_PATTERN,
    DEFAULT_INTERVAL,
    DEFAULT_SPAWN_RATE,
    LOCUST_WEB_PORT,
)

# ── Experiment Matrix ────────────────────────────────────────────────────────
EXPERIMENT_MATRIX = {
    "exp_01_baseline": {
        "label": "Baseline (No Management)",
        "monitor_only": True,
        "actions": {
            "horizontal_scaling": False,
            "vertical_scaling": False,
            "service_placement": False,
            "flow_scheduling": False,
        },
    },
    "exp_02_vertical_only": {
        "label": "Vertical Scaling Only",
        "monitor_only": False,
        "actions": {
            "horizontal_scaling": False,
            "vertical_scaling": True,
            "service_placement": False,
            "flow_scheduling": False,
        },
    },
    "exp_03_horizontal_only": {
        "label": "Horizontal Scaling Only",
        "monitor_only": False,
        "actions": {
            "horizontal_scaling": True,
            "vertical_scaling": False,
            "service_placement": False,
            "flow_scheduling": False,
        },
    },
    "exp_04_service_placement_only": {
        "label": "Service Placement Only",
        "monitor_only": False,
        "actions": {
            "horizontal_scaling": False,
            "vertical_scaling": False,
            "service_placement": True,
            "flow_scheduling": False,
        },
    },
    "exp_05_flow_scheduling_only": {
        "label": "Flow Scheduling Only",
        "monitor_only": False,
        "actions": {
            "horizontal_scaling": False,
            "vertical_scaling": False,
            "service_placement": False,
            "flow_scheduling": True,
        },
    },
    "exp_06_vertical_horizontal": {
        "label": "Vertical + Horizontal Scaling",
        "monitor_only": False,
        "actions": {
            "horizontal_scaling": True,
            "vertical_scaling": True,
            "service_placement": False,
            "flow_scheduling": False,
        },
    },
    "exp_07_vertical_horizontal_flow": {
        "label": "Vertical + Horizontal + Flow Scheduling",
        "monitor_only": False,
        "actions": {
            "horizontal_scaling": True,
            "vertical_scaling": True,
            "service_placement": False,
            "flow_scheduling": True,
        },
    },
    "exp_08_full_system": {
        "label": "Full System (All Actions)",
        "monitor_only": False,
        "actions": {
            "horizontal_scaling": True,
            "vertical_scaling": True,
            "service_placement": True,
            "flow_scheduling": True,
        },
    },
    "exp_09_cloud_llm_baseline": {
        "label": "Cloud LLM Baseline (GPT-4o)",
        "monitor_only": False,
        "actions": {
            "horizontal_scaling": True,
            "vertical_scaling": True,
            "service_placement": True,
            "flow_scheduling": True,
        },
        "llm_override": {
            "provider": "openai",
            "model": "gpt-4o",
            "max_tokens": 512,
        },
    },

    # ── Stress experiments — peak 80 users (≈2.7× normal) to push node CPU above 80%
    # and saturate SDN links, activating service_placement and flow_scheduling triggers.
    # These directly demonstrate the two action types that produce 0 candidates at normal load.
    "exp_10_stress_service_placement_only": {
        "label": "Stress Load: Service Placement Only",
        "monitor_only": False,
        "load_pattern": [20, 40, 60, 80, 60, 40, 20],
        "actions": {
            "horizontal_scaling": False,
            "vertical_scaling": False,
            "service_placement": True,
            "flow_scheduling": False,
        },
    },
    "exp_11_stress_flow_scheduling_only": {
        "label": "Stress Load: Flow Scheduling Only",
        "monitor_only": False,
        "load_pattern": [20, 40, 60, 80, 60, 40, 20],
        "actions": {
            "horizontal_scaling": False,
            "vertical_scaling": False,
            "service_placement": False,
            "flow_scheduling": True,
        },
    },
    "exp_12_stress_full_system": {
        "label": "Stress Load: Full System (All Actions)",
        "monitor_only": False,
        "load_pattern": [20, 40, 60, 80, 60, 40, 20],
        "actions": {
            "horizontal_scaling": True,
            "vertical_scaling": True,
            "service_placement": True,
            "flow_scheduling": True,
        },
    },
}

BASE_CONFIG = "config.yaml"
RESULTS_ROOT = "evaluation_results"

# llm-server SSH details for resource_monitor.sh
LLM_SERVER_IP = "129.114.26.41"
LLM_SERVER_USER = "cc"
RESOURCE_MONITOR_REMOTE_PATH = "/home/cc/resource_monitor.sh"


# ── SSH helper ───────────────────────────────────────────────────────────────

def _ssh(host: str, user: str, cmd: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", "-i", "/home/cc/.ssh/id_ed25519",
         "-o", "StrictHostKeyChecking=no",
         f"{user}@{host}", cmd],
        capture_output=True, text=True, timeout=timeout
    )


# ── Cluster stabilisation wait ───────────────────────────────────────────────

def _parse_cpu_millicores(value: str) -> int:
    """Convert kubectl CPU strings like '372m', '1', '2500m' to millicores int."""
    value = value.strip()
    if value.endswith("m"):
        return int(value[:-1])
    try:
        return int(float(value) * 1000)
    except ValueError:
        return 0


def _get_cluster_total_cpu_m(master_ip: str, ssh_user: str) -> Optional[int]:
    """SSH to master, run kubectl top nodes, return total CPU millicores across all nodes."""
    result = _ssh(master_ip, ssh_user, "kubectl top nodes --no-headers 2>/dev/null", timeout=15)
    if result.returncode != 0 or not result.stdout.strip():
        return None
    total = 0
    for line in result.stdout.strip().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            total += _parse_cpu_millicores(parts[1])
    return total


def wait_for_cluster_stable(
    master_ip: str, ssh_user: str,
    max_wait_s: int = 180, threshold_m: int = 500
) -> None:
    """
    Block until total cluster CPU drops below threshold_m millicores (or timeout).
    Called after reset_cluster() to ensure experiments start from a clean baseline.
    """
    print(f"  ⏳ Waiting for cluster CPU to stabilise (threshold: {threshold_m}m, max: {max_wait_s}s)...")
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        total = _get_cluster_total_cpu_m(master_ip, ssh_user)
        if total is not None and total < threshold_m:
            print(f"  ✅ Cluster stable: {total}m total CPU")
            return
        used_str = f"{total}m" if total is not None else "unknown"
        print(f"     CPU still at {used_str} — waiting 15s...")
        time.sleep(15)
    print(f"  ⚠️  Cluster did not stabilise within {max_wait_s}s — proceeding anyway")


# ── K8s resource monitor (background SSH loop on sdn-controller) ─────────────

def start_k8s_monitor(output_dir: str, master_ip: str, ssh_user: str) -> subprocess.Popen:
    """
    Launch a background bash loop on the local machine that polls
    kubectl top nodes/pods every 15 seconds via SSH and appends to CSV files.
    """
    node_csv = os.path.join(output_dir, "k8s_node_resources.csv")
    pod_csv = os.path.join(output_dir, "k8s_pod_resources.csv")

    script = f"""#!/bin/bash
echo "timestamp,node,cpu_cores,memory_mi" > {node_csv}
echo "timestamp,namespace,pod_name,cpu_m,memory_mi" > {pod_csv}
while true; do
    TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no {ssh_user}@{master_ip} \\
        "kubectl top nodes --no-headers 2>/dev/null" | \\
        awk -v ts="$TS" '{{print ts","$1","$2","$4}}' >> {node_csv}
    ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no {ssh_user}@{master_ip} \\
        "kubectl top pods --all-namespaces --no-headers 2>/dev/null" | \\
        awk -v ts="$TS" '{{print ts","$1","$2","$3","$5}}' >> {pod_csv}
    sleep 15
done
"""
    script_path = os.path.join(output_dir, "_k8s_monitor.sh")
    with open(script_path, "w") as f:
        f.write(script)
    os.chmod(script_path, 0o755)
    log = open(os.path.join(output_dir, "k8s_monitor.log"), "w")
    proc = subprocess.Popen(["bash", script_path], stdout=log, stderr=subprocess.STDOUT)
    return proc


def stop_k8s_monitor(proc: subprocess.Popen) -> None:
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


# ── LLM-server resource monitor ─────────────────────────────────────────────

def start_llm_resource_monitor(output_dir: str) -> Optional[str]:
    """Start resource_monitor.sh on llm-server via SSH (background process)."""
    remote_csv = f"/tmp/llm_server_resources_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    cmd = f"bash {RESOURCE_MONITOR_REMOTE_PATH} start {remote_csv}"
    result = _ssh(LLM_SERVER_IP, LLM_SERVER_USER, cmd, timeout=15)
    if result.returncode != 0:
        print(f"  ⚠️  Could not start resource_monitor.sh on llm-server: {result.stderr.strip()}")
        return None
    print(f"  ✅ resource_monitor.sh started on llm-server (remote: {remote_csv})")
    return remote_csv


def stop_and_fetch_llm_monitor(remote_csv: Optional[str], output_dir: str) -> None:
    """Stop resource_monitor.sh on llm-server and scp the CSV back."""
    if not remote_csv:
        return
    _ssh(LLM_SERVER_IP, LLM_SERVER_USER, "bash " + RESOURCE_MONITOR_REMOTE_PATH + " stop", timeout=10)
    local_csv = os.path.join(output_dir, "llm_server_resources.csv")
    subprocess.run(
        ["scp", "-i", "/home/cc/.ssh/id_ed25519",
         "-o", "StrictHostKeyChecking=no",
         f"{LLM_SERVER_USER}@{LLM_SERVER_IP}:{remote_csv}", local_csv],
        capture_output=True, timeout=30
    )
    print(f"  ✅ llm-server resource CSV fetched → {local_csv}")


# ── Intent loop launcher ─────────────────────────────────────────────────────

def start_intent_loop(
    config_path: str, duration_minutes: int, output_dir: str,
    monitor_only: bool = False, debug_llm: bool = False
) -> tuple:
    output_file = os.path.join(output_dir, "intent_loop_log.json")
    cmd = [
        "python3", "main.py",
        "--config", config_path,
        "--time-window", str(duration_minutes),
        "--output", output_file,
        "--log-level", "INFO",
    ]
    if monitor_only:
        cmd.append("--monitor-only")
    if debug_llm:
        cmd.append("--debug-llm")

    log_file = open(os.path.join(output_dir, "intent_loop.log"), "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    print(f"  🧠 Intent loop started (PID {proc.pid}, monitor_only={monitor_only})")
    return proc, log_file


# ── Config builder ───────────────────────────────────────────────────────────

def build_experiment_config(exp_def: dict, base_config_path: str, output_dir: str) -> str:
    """
    Load base config.yaml, override the actions block, and write a
    per-experiment config file into output_dir.  Returns the path.
    """
    with open(base_config_path) as f:
        cfg = yaml.safe_load(f)

    cfg["actions"] = copy.deepcopy(exp_def["actions"])

    if "llm_override" in exp_def:
        cfg.setdefault("llm", {}).update(exp_def["llm_override"])

    config_path = os.path.join(output_dir, "experiment_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return config_path


# ── Per-experiment runner ────────────────────────────────────────────────────

def run_experiment(
    name: str,
    exp_def: dict,
    dry_run: bool = False,
    skip_reset: bool = False,
    debug_llm: bool = False,
):
    label = exp_def["label"]
    monitor_only = exp_def["monitor_only"]
    enabled = [k for k, v in exp_def["actions"].items() if v]

    output_dir = os.path.join(RESULTS_ROOT, name)
    os.makedirs(output_dir, exist_ok=True)

    # Per-experiment load pattern override (e.g. stress experiments use higher user counts)
    load_pattern = exp_def.get("load_pattern", DEFAULT_LOAD_PATTERN)
    interval = DEFAULT_INTERVAL
    spawn_rate = DEFAULT_SPAWN_RATE
    total_duration = len(load_pattern) * interval + 60  # extra 60s buffer
    duration_minutes = (total_duration // 60) + 1

    print("\n" + "=" * 70)
    print(f"  EXPERIMENT: {name}")
    print(f"  Label:      {label}")
    print(f"  Actions:    {enabled if enabled else '(none — monitor-only)'}")
    print(f"  Output:     {output_dir}/")
    print("=" * 70)

    if dry_run:
        print("  [DRY RUN] Skipping execution")
        _print_action_flags(exp_def["actions"])
        return

    # 1. Write per-experiment config
    config_path = build_experiment_config(exp_def, BASE_CONFIG, output_dir)
    print(f"  📝 Config written: {config_path}")
    _print_action_flags(exp_def["actions"])

    # 2. Reset cluster to default state + wait for CPU to quiesce
    if not skip_reset:
        print("\n  🔄 Resetting cluster...")
        reset_cluster(BASE_CONFIG)
        with open(BASE_CONFIG) as f:
            _base_cfg_tmp = yaml.safe_load(f)
        wait_for_cluster_stable(
            master_ip=_base_cfg_tmp["endpoints"]["kubernetes_master"],
            ssh_user=_base_cfg_tmp["endpoints"].get("kubernetes_user", "cc"),
        )
    else:
        print("  ⏭️  Cluster reset skipped")

    # 3. Warm up LLM (skip for baseline; skip for non-Ollama providers)
    if not monitor_only:
        provider = exp_def.get("llm_override", {}).get("provider", "ollama")
        if provider == "ollama":
            warmup_llm(BASE_CONFIG)
        else:
            print(f"  ⏭️  LLM warm-up skipped (provider={provider})")

    # 4. Start LLM-server resource monitor
    remote_csv = start_llm_resource_monitor(output_dir) if not monitor_only else None

    # 5. Start K8s node/pod monitor
    with open(BASE_CONFIG) as f:
        base_cfg = yaml.safe_load(f)
    master_ip = base_cfg["endpoints"]["kubernetes_master"]
    k8s_proc = start_k8s_monitor(output_dir, master_ip, base_cfg["endpoints"].get("kubernetes_user", "cc"))
    print(f"  📊 K8s resource monitor started")

    # 6. Start intent loop
    locust_csv_prefix = os.path.join(output_dir, "locust_results")
    intent_proc, intent_log = start_intent_loop(
        config_path, duration_minutes, output_dir,
        monitor_only=monitor_only, debug_llm=debug_llm
    )
    time.sleep(3)  # give intent loop time to initialize

    # 7. Start Locust
    initial_users = load_pattern[0]
    locust_proc, locust_log = start_locust_locally(
        BASE_CONFIG, initial_users, spawn_rate, total_duration, output_dir
    )

    if locust_proc is None:
        print("  ❌ Locust failed to start — aborting experiment")
        intent_proc.terminate()
        stop_k8s_monitor(k8s_proc)
        stop_and_fetch_llm_monitor(remote_csv, output_dir)
        return

    # 8. Run staged load schedule
    print(f"\n  🦗 Running load schedule ({len(load_pattern)} stages × {interval}s)...")
    run_load_schedule(load_pattern, interval, spawn_rate)

    # 9. Wait for Locust and intent loop to finish
    # Locust run-time equals total_duration; after schedule ends we give it 90s to autoquit
    print("\n  ⏳ Waiting for processes to complete...")
    try:
        locust_proc.wait(timeout=90)
    except subprocess.TimeoutExpired:
        stop_locust_locally(locust_proc)

    try:
        intent_proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        print("  ⚠️  Intent loop still running — terminating")
        intent_proc.terminate()
        intent_proc.wait(timeout=15)

    for lf in [intent_log, locust_log]:
        try:
            lf.close()
        except Exception:
            pass

    # 10. Stop monitors and fetch llm-server CSV
    stop_k8s_monitor(k8s_proc)
    stop_and_fetch_llm_monitor(remote_csv, output_dir)

    # 11. Export telemetry summary
    _export_telemetry(name, label, enabled, output_dir, locust_csv_prefix)

    # 12. Print collection summary
    collect_summary(output_dir, load_pattern, interval)
    print(f"\n  ✅ Experiment complete — results in {output_dir}/")


def _export_telemetry(
    name: str, label: str, enabled_actions: list,
    output_dir: str, locust_csv_prefix: str
):
    """Load the intent_loop_log.json and build summary.json via ExperimentTelemetry."""
    from experiment_telemetry import ExperimentTelemetry

    tel = ExperimentTelemetry(
        experiment_name=name,
        experiment_label=label,
        enabled_actions=enabled_actions,
        output_dir=output_dir,
    )
    tel.set_locust_csv_prefix(locust_csv_prefix)

    # Load intent loop output and reconstruct stats/history for telemetry export
    intent_log_path = os.path.join(output_dir, "intent_loop_log.json")
    checkpoint_path = intent_log_path + ".checkpoint.json"
    _load_path = intent_log_path if os.path.exists(intent_log_path) else (
        checkpoint_path if os.path.exists(checkpoint_path) else None
    )
    if _load_path:
        if _load_path == checkpoint_path:
            print(f"  ⚠️  Using checkpoint file (main JSON missing): {checkpoint_path}")
        with open(_load_path) as f:
            intent_data = json.load(f)

        # Build lightweight proxies that ExperimentTelemetry can read from
        class _StatsProxy:
            def __init__(self, data):
                self.stats = data.get("stats", {})
                self._structured = data.get("all_structured_history", [])

            def get_all_structured_history(self):
                return self._structured

        class _WatchLoopProxy:
            def __init__(self, data):
                self.stats = data.get("stats", {})
                self.decision_history = _StatsProxy(data)

        class _DecisionMakerProxy:
            def __init__(self, data):
                self._calls = data.get("llm_calls_log", [])

            def get_llm_calls_log(self):
                return self._calls

        tel.attach_watch_loop(_WatchLoopProxy(intent_data))
        tel.attach_decision_maker(_DecisionMakerProxy(intent_data))

        # Feed EMA timeline + SLO thresholds for time-in-band metric
        ema_timeline = intent_data.get("ema_timeline", [])
        exp_cfg_path = os.path.join(output_dir, "experiment_config.yaml")
        lower_threshold, upper_threshold = 1.0, 2.0
        if os.path.exists(exp_cfg_path):
            try:
                with open(exp_cfg_path) as cf:
                    exp_cfg = yaml.safe_load(cf)
                lower_threshold = exp_cfg.get("intent", {}).get("lower_threshold", 1.0)
                upper_threshold = exp_cfg.get("intent", {}).get("upper_threshold", 2.0)
            except Exception:
                pass
        tel.set_ema_data(ema_timeline, lower_threshold, upper_threshold)

    # Note: LLM calls log is embedded inside intent_loop_log.json via the
    # watch loop return dict in run_time_window().  For post-hoc export we
    # rely on the actions_log built from all_structured_history.
    # llm_interactions.csv is only fully populated when telemetry is exported
    # in-process (when run_ablation drives main.py in-process in future).
    # For now, write what we have.
    tel.export_all()


def aggregate_runs(run_dirs: list, agg_dir: str, name: str, label: str) -> None:
    """
    Read summary.json from each run directory and write summary_aggregated.json
    with mean ± std for every numeric field.
    """
    os.makedirs(agg_dir, exist_ok=True)

    summaries = []
    for d in run_dirs:
        p = os.path.join(d, "summary.json")
        if os.path.exists(p):
            with open(p) as f:
                summaries.append(json.load(f))
        else:
            print(f"  ⚠️  Missing summary.json in {d} — skipping this run")

    if not summaries:
        print(f"  ❌ No summary files found — aggregation skipped")
        return

    numeric_fields = [
        "duration_seconds",
        "total_locust_requests", "total_locust_failures",
        "violations_detected", "no_candidate_violations",
        "actions_executed", "actions_succeeded", "violations_resolved",
        "intent_satisfaction_rate", "time_normalised_isr",
        "ema_time_in_band_pct", "locust_p95_rt_ms",
        "mean_inference_latency_ms", "p95_inference_latency_ms",
        "mean_prompt_tokens", "mean_completion_tokens",
        "total_prompt_tokens", "total_completion_tokens",
    ]

    agg = {
        "experiment_name": name,
        "experiment_label": label,
        "n_runs": len(summaries),
        "run_dirs": run_dirs,
        "aggregated_at": datetime.utcnow().isoformat(),
    }

    for field in numeric_fields:
        values = [s[field] for s in summaries if s.get(field) is not None]
        if not values:
            agg[f"{field}_mean"] = None
            agg[f"{field}_std"] = None
        elif len(values) == 1:
            agg[f"{field}_mean"] = round(values[0], 4) if isinstance(values[0], float) else values[0]
            agg[f"{field}_std"] = None
        else:
            agg[f"{field}_mean"] = round(statistics.mean(values), 4)
            agg[f"{field}_std"] = round(statistics.stdev(values), 4)

    # Average action_type_breakdown across runs
    action_counts: dict = {}
    for s in summaries:
        for action, count in s.get("action_type_breakdown", {}).items():
            action_counts[action] = action_counts.get(action, 0) + count
    if action_counts:
        agg["action_type_breakdown_mean"] = {
            k: round(v / len(summaries), 1) for k, v in action_counts.items()
        }

    out_path = os.path.join(agg_dir, "summary_aggregated.json")
    with open(out_path, "w") as f:
        json.dump(agg, f, indent=2)

    isr_m = agg.get("intent_satisfaction_rate_mean")
    isr_s = agg.get("intent_satisfaction_rate_std")
    p95_m = agg.get("locust_p95_rt_ms_mean")
    p95_s = agg.get("locust_p95_rt_ms_std")
    print(f"\n  📊 Aggregated {len(summaries)} run(s) → {out_path}")
    print(f"     ISR:    {isr_m:.4f} ± {isr_s:.4f}" if isr_s is not None else
          f"     ISR:    {isr_m}")
    print(f"     P95 RT: {p95_m} ± {p95_s} ms" if p95_s is not None else
          f"     P95 RT: {p95_m} ms")


def _print_action_flags(actions: dict) -> None:
    for action, enabled in actions.items():
        status = "✅" if enabled else "❌"
        print(f"    {status} {action}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Ablation Study Runner")
    parser.add_argument("--experiment", default=None,
                        help="Name of a single experiment to run (e.g. exp_02_vertical_only)")
    parser.add_argument("--all", action="store_true",
                        help="Run all experiments sequentially")
    parser.add_argument("--list", action="store_true",
                        help="Print the experiment matrix and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run without executing")
    parser.add_argument("--skip-reset", action="store_true",
                        help="Skip cluster reset before each experiment")
    parser.add_argument("--debug-llm", action="store_true",
                        help="Enable LLM debug logging in intent loop")
    parser.add_argument("--repeat", type=int, default=1, metavar="N",
                        help="Run each experiment N times and aggregate mean ± std (default: 1)")
    parser.add_argument("--results-root", default=RESULTS_ROOT, metavar="DIR",
                        help="Root directory for experiment outputs (default: evaluation_results)")
    return parser.parse_args()


def main():
    args = parse_args()
    global RESULTS_ROOT
    RESULTS_ROOT = args.results_root

    if args.list:
        print("\nExperiment Matrix:")
        print("-" * 70)
        for name, exp in EXPERIMENT_MATRIX.items():
            enabled = [k for k, v in exp["actions"].items() if v]
            mode = "monitor-only" if exp["monitor_only"] else f"LLM [{', '.join(enabled)}]"
            print(f"  {name:<42} {exp['label']} ({mode})")
        print("-" * 70)
        return

    if args.all:
        experiments = list(EXPERIMENT_MATRIX.items())
    elif args.experiment:
        if args.experiment not in EXPERIMENT_MATRIX:
            print(f"Unknown experiment: {args.experiment}")
            print(f"Available: {list(EXPERIMENT_MATRIX.keys())}")
            sys.exit(1)
        experiments = [(args.experiment, EXPERIMENT_MATRIX[args.experiment])]
    else:
        print("Specify --experiment <name>, --all, or --list")
        sys.exit(1)

    os.makedirs(RESULTS_ROOT, exist_ok=True)

    for name, exp_def in experiments:
        if args.repeat > 1:
            # Multi-run mode: each run goes into exp_XX_run1/, _run2/, etc.
            run_dirs = []
            for run_idx in range(1, args.repeat + 1):
                run_name = f"{name}_run{run_idx}"
                print(f"\n{'='*70}")
                print(f"  REPEAT {run_idx}/{args.repeat} for {name}")
                print(f"{'='*70}")
                run_experiment(
                    name=run_name,
                    exp_def=exp_def,
                    dry_run=args.dry_run,
                    skip_reset=args.skip_reset,
                    debug_llm=args.debug_llm,
                )
                run_dirs.append(os.path.join(RESULTS_ROOT, run_name))
                if run_idx < args.repeat and not args.dry_run:
                    print("\n  ⏳ Cooling down 60s before next repeat...")
                    time.sleep(60)

            # Aggregate all runs into exp_XX_agg/
            if not args.dry_run:
                agg_dir = os.path.join(RESULTS_ROOT, f"{name}_agg")
                aggregate_runs(run_dirs, agg_dir, name, exp_def["label"])
        else:
            run_experiment(
                name=name,
                exp_def=exp_def,
                dry_run=args.dry_run,
                skip_reset=args.skip_reset,
                debug_llm=args.debug_llm,
            )

        if not args.dry_run and len(experiments) > 1:
            print("\n  ⏳ Cooling down 60s before next experiment...")
            time.sleep(60)

    print("\n✅ All experiments complete.")
    if not args.dry_run:
        print(f"   Results in: {RESULTS_ROOT}/")
        print("   Run:  python3 evaluation_results/plot_comparison.py")


if __name__ == "__main__":
    main()
