#!/usr/bin/env python3
"""
IntentContinuum Experiment Runner

Orchestrates the full experiment:
1. Resets the cluster to initial state
2. Starts Locust with a staged load pattern
3. Starts main.py (Intent Watch Loop) in parallel
4. Collects results from both
5. Generates summary

Usage:
    python3 run_experiment.py                          # Default: computing experiment
    python3 run_experiment.py --name my_experiment     # Custom name
    python3 run_experiment.py --load 10,20,15,10       # Custom load pattern
    python3 run_experiment.py --interval 120            # Custom interval between load changes
    python3 run_experiment.py --dry-run                 # Show plan without executing
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ── Default Experiment Configuration ────────────────────────────────────────
# Matches the IntentContinuum paper's computing experiment
DEFAULT_LOAD_PATTERN = [10, 20, 15, 10, 5, 20, 10]
DEFAULT_INTERVAL = 120  # seconds between load changes
DEFAULT_SPAWN_RATE = 1  # users per second


def parse_args():
    parser = argparse.ArgumentParser(description="IntentContinuum Experiment Runner")
    parser.add_argument("--name", default=None, help="Experiment name (default: auto-generated)")
    parser.add_argument("--load", default=None, help="Comma-separated load pattern (e.g., 10,20,15,10)")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL, help="Seconds between load changes")
    parser.add_argument("--spawn-rate", type=int, default=DEFAULT_SPAWN_RATE, help="Users spawned per second")
    parser.add_argument("--config", default="config.yaml", help="Path to IntentContinuum config")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    parser.add_argument("--skip-reset", action="store_true", help="Skip cluster reset")
    parser.add_argument("--debug-llm", action="store_true", help="Enable LLM debug logging")
    return parser.parse_args()


def reset_cluster(config_path: str):
    """Reset all deployments to their initial state from config."""
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    master = config["endpoints"]["kubernetes_master"]
    user = config["endpoints"]["kubernetes_user"]
    deployments = config.get("kubernetes", {}).get("deployments", [])

    print("\n🔄 Resetting cluster to initial state...")
    for dep in deployments:
        name = dep["name"]
        cpu = dep.get("default_cpu", "300m")
        mem = dep.get("default_memory", "312Mi")
        replicas = dep.get("min_replicas", 1)

        # Reset replicas
        cmd = f"kubectl scale deployment {name} --replicas={replicas}"
        subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", f"{user}@{master}", cmd],
            capture_output=True, timeout=30
        )
        print(f"  ✅ {name}: replicas={replicas}")

        # Reset resources
        cmd = f"kubectl set resources deployment {name} --limits=cpu={cpu},memory={mem} -c {name.replace('-deployment', '')}"
        subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", f"{user}@{master}", cmd],
            capture_output=True, timeout=30
        )
        print(f"  ✅ {name}: cpu={cpu}, memory={mem}")

    # Wait for pods to stabilize
    print("  ⏳ Waiting 30s for pods to stabilize...")
    time.sleep(30)

    # Verify pods are running
    cmd = "kubectl get pods --no-headers | grep -c Running"
    result = subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", f"{user}@{master}", cmd],
        capture_output=True, text=True, timeout=30
    )
    running = result.stdout.strip()
    print(f"  ✅ {running} pods running")


def start_locust(load_pattern, interval, spawn_rate, results_dir, locustfile="locustfile.py"):
    """
    Start Locust in headless mode with staged load changes.

    Locust runs as a background process. We use its command-line interface
    for the initial user count and then change load via the REST API.
    """
    total_duration = len(load_pattern) * interval
    initial_users = load_pattern[0]
    csv_prefix = os.path.join(results_dir, "locust")

    print(f"\n🦗 Starting Locust (initial={initial_users} users, duration={total_duration}s)...")

    locust_cmd = [
        "python3", "-m", "locust",
        "-f", locustfile,
        "--headless",
        "--host", "ssh://master",
        "-u", str(initial_users),
        "-r", str(spawn_rate),
        "--run-time", f"{total_duration}s",
        "--csv", csv_prefix,
        "--csv-full-history",
    ]

    locust_proc = subprocess.Popen(
        locust_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    return locust_proc, total_duration


def change_locust_load(target_users, spawn_rate):
    """Change the number of Locust users via REST API."""
    import urllib.request
    import urllib.parse

    data = urllib.parse.urlencode({
        "user_count": target_users,
        "spawn_rate": spawn_rate,
    }).encode()

    try:
        req = urllib.request.Request(
            "http://localhost:8089/swarm",
            data=data,
            method="PATCH"
        )
        urllib.request.urlopen(req, timeout=5)
        return True
    except Exception as e:
        print(f"  ⚠️  Failed to change Locust load: {e}")
        return False


def start_intent_loop(config_path, duration_minutes, results_dir, debug_llm=False):
    """Start main.py in background."""
    output_file = os.path.join(results_dir, "intent_results.json")

    cmd = [
        "python3", "main.py",
        "--config", config_path,
        "--time-window", str(duration_minutes),
        "--output", output_file,
        "--log-level", "INFO",
    ]
    if debug_llm:
        cmd.append("--debug-llm")

    print(f"\n🧠 Starting Intent Watch Loop (duration={duration_minutes}min)...")

    # Redirect output to a log file
    log_file = open(os.path.join(results_dir, "intent_loop.log"), "w")

    intent_proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    return intent_proc, log_file


def run_load_schedule(load_pattern, interval, spawn_rate):
    """
    Execute the staged load pattern by changing Locust user count at intervals.
    The first stage is already set when Locust starts, so we start from stage 1.
    """
    print(f"\n📊 Load Schedule:")
    for i, users in enumerate(load_pattern):
        start = i * interval
        end = start + interval
        print(f"  [{start:>4}s - {end:>4}s] → {users} users")

    # Wait for Locust to fully start
    time.sleep(10)

    for i, users in enumerate(load_pattern):
        if i == 0:
            # First stage already set at Locust startup
            print(f"\n⏱️  Stage 1/{len(load_pattern)}: {users} users (already active)")
            remaining = interval - 10  # We already waited 10s
            time.sleep(max(0, remaining))
        else:
            print(f"\n⏱️  Stage {i+1}/{len(load_pattern)}: changing to {users} users")
            change_locust_load(users, spawn_rate)
            time.sleep(interval)


def collect_summary(results_dir, load_pattern, interval):
    """Collect and print experiment summary."""
    print("\n" + "=" * 60)
    print("📋 EXPERIMENT SUMMARY")
    print("=" * 60)

    # Intent results
    intent_file = os.path.join(results_dir, "intent_results.json")
    if os.path.exists(intent_file):
        with open(intent_file) as f:
            intent_data = json.load(f)

        stats = intent_data.get("stats", {})
        history = intent_data.get("history", [])

        print(f"\n  Intent Watch Loop:")
        print(f"    Total requests (monitoring): {stats.get('total_requests', 'N/A')}")
        print(f"    Violations detected: {stats.get('violations_detected', 0)}")
        print(f"    Actions taken: {stats.get('actions_taken', 0)}")
        print(f"    Duration: {stats.get('start_time', '?')} → {stats.get('end_time', '?')}")

        if history:
            print(f"\n  Decision History:")
            for entry in history:
                vnum = entry.get("violation_number", "?")
                action = entry.get("decision", {}).get("action", "none")
                params = entry.get("decision", {}).get("parameters", {})
                outcome = entry.get("outcome", "pending")
                dep = params.get("deployment_name", "?")
                print(f"    V{vnum}: {action} on {dep} → {outcome}")
    else:
        print("  ⚠️  No intent results found")

    # Locust stats
    locust_stats = os.path.join(results_dir, "locust_stats.csv")
    if os.path.exists(locust_stats):
        print(f"\n  Locust stats saved to: {locust_stats}")
    
    locust_history = os.path.join(results_dir, "locust_stats_history.csv")
    if os.path.exists(locust_history):
        print(f"  Locust history saved to: {locust_history}")

    # Intent loop log
    log_file = os.path.join(results_dir, "intent_loop.log")
    if os.path.exists(log_file):
        print(f"  Intent loop log: {log_file}")

    print(f"\n  All results in: {results_dir}/")
    print("=" * 60)


def main():
    args = parse_args()

    # Parse load pattern
    if args.load:
        load_pattern = [int(x.strip()) for x in args.load.split(",")]
    else:
        load_pattern = DEFAULT_LOAD_PATTERN

    total_duration_s = len(load_pattern) * args.interval
    total_duration_min = total_duration_s / 60

    # Create experiment name and results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.name or f"experiment_{timestamp}"
    results_dir = os.path.join("results", exp_name)
    os.makedirs(results_dir, exist_ok=True)

    # Save experiment config
    exp_config = {
        "name": exp_name,
        "timestamp": timestamp,
        "load_pattern": load_pattern,
        "interval_seconds": args.interval,
        "spawn_rate": args.spawn_rate,
        "total_duration_seconds": total_duration_s,
        "config_file": args.config,
    }
    with open(os.path.join(results_dir, "experiment_config.json"), "w") as f:
        json.dump(exp_config, f, indent=2)

    # Print plan
    print("\n" + "=" * 60)
    print(f"🔬 EXPERIMENT: {exp_name}")
    print("=" * 60)
    print(f"  Load pattern: {load_pattern}")
    print(f"  Interval: {args.interval}s between changes")
    print(f"  Spawn rate: {args.spawn_rate} user/s")
    print(f"  Total duration: {total_duration_s}s ({total_duration_min:.1f} min)")
    print(f"  Results dir: {results_dir}/")

    if args.dry_run:
        print("\n  [DRY RUN] - would execute the above plan")
        return

    # Step 1: Reset cluster
    if not args.skip_reset:
        reset_cluster(args.config)
    else:
        print("\n⏭️  Skipping cluster reset")

    # Step 2: Start Intent Watch Loop
    intent_proc, intent_log = start_intent_loop(
        config_path=args.config,
        duration_minutes=int(total_duration_min) + 2,  # +2 min buffer
        results_dir=results_dir,
        debug_llm=args.debug_llm,
    )

    # Wait a moment for intent loop to initialize
    time.sleep(5)

    # Step 3: Start Locust
    locust_proc, _ = start_locust(
        load_pattern=load_pattern,
        interval=args.interval,
        spawn_rate=args.spawn_rate,
        results_dir=results_dir,
    )

    # Step 4: Execute load schedule
    try:
        run_load_schedule(load_pattern, args.interval, args.spawn_rate)
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")
    finally:
        # Step 5: Cleanup
        print("\n🛑 Stopping processes...")

        # Stop Locust
        if locust_proc.poll() is None:
            locust_proc.send_signal(signal.SIGINT)
            try:
                locust_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                locust_proc.kill()
            print("  ✅ Locust stopped")

        # Wait for Intent Loop to finish (it has its own timer)
        print("  ⏳ Waiting for Intent Watch Loop to finish...")
        try:
            intent_proc.wait(timeout=180)  # 3 min max wait
        except subprocess.TimeoutExpired:
            intent_proc.send_signal(signal.SIGINT)
            intent_proc.wait(timeout=15)
        print("  ✅ Intent Watch Loop stopped")

        intent_log.close()

        # Step 6: Summary
        collect_summary(results_dir, load_pattern, args.interval)


if __name__ == "__main__":
    main()
