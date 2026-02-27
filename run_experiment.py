#!/usr/bin/env python3
"""
IntentContinuum Experiment Runner

Orchestrates the full experiment:
1. Resets the cluster to initial state
2. Starts Locust on the MASTER node via SSH (direct HTTP access to app)
3. Starts main.py (Intent Watch Loop) locally on SDN controller
4. Changes Locust load via REST API at each interval
5. Collects results and generates summary

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
import urllib.request
import urllib.parse
from datetime import datetime
from pathlib import Path


# ── Default Experiment Configuration ────────────────────────────────────────
# Matches the IntentContinuum paper's computing experiment
DEFAULT_LOAD_PATTERN = [10, 20, 15, 10, 5, 20, 10]
DEFAULT_INTERVAL = 120  # seconds between load changes
DEFAULT_SPAWN_RATE = 1  # users per second
LOCUST_WEB_PORT = 8089


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


def get_master_info(config_path: str):
    """Read master node connection info from config."""
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    master = config["endpoints"]["kubernetes_master"]
    user = config["endpoints"]["kubernetes_user"]
    return user, master


def ssh_cmd(user, host, command, timeout=30):
    """Run a command on a remote host via SSH."""
    result = subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
         f"{user}@{host}", command],
        capture_output=True, text=True, timeout=timeout
    )
    return result


def reset_cluster(config_path: str):
    """Reset all deployments to their initial state from config."""
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    user, master = get_master_info(config_path)
    deployments = config.get("kubernetes", {}).get("deployments", [])

    print("\n🔄 Resetting cluster to initial state...")
    for dep in deployments:
        name = dep["name"]
        cpu = dep.get("default_cpu", "300m")
        mem = dep.get("default_memory", "312Mi")
        replicas = dep.get("min_replicas", 1)

        # Reset replicas
        ssh_cmd(user, master, f"kubectl scale deployment {name} --replicas={replicas}")
        print(f"  ✅ {name}: replicas={replicas}")

        # Get actual container name from the deployment
        result = ssh_cmd(user, master,
            f'kubectl get deployment {name} -o jsonpath="{{.spec.template.spec.containers[0].name}}"')
        container_name = result.stdout.strip().strip("'\"")
        if container_name:
            ssh_cmd(user, master,
                f"kubectl set resources deployment {name} --limits=cpu={cpu},memory={mem} -c {container_name}")
            print(f"  ✅ {name}: cpu={cpu}, memory={mem} (container: {container_name})")
        else:
            print(f"  ⚠️  {name}: could not determine container name, skipping resource reset")

    # Wait for pods to stabilize
    print("  ⏳ Waiting 30s for pods to stabilize...")
    time.sleep(30)

    # Verify pods are running
    result = ssh_cmd(user, master, "kubectl get pods --no-headers | grep -c Running")
    running = result.stdout.strip()
    print(f"  ✅ {running} pods running")


def start_locust_on_master(user, master, initial_users, spawn_rate, total_duration):
    """
    Start Locust on the master node via SSH.
    Locust runs in the background with web UI enabled for REST API control.
    The locustfile.py must already exist on the master at ~/locustfile.py.
    """
    print(f"\n🦗 Starting Locust on master node ({master})...")

    # Kill any existing Locust process
    ssh_cmd(user, master, "pkill -f 'locust' 2>/dev/null || true")
    time.sleep(2)

    # Write a startup script on the master, then execute it
    # This avoids all quoting issues with nested SSH commands
    startup_script = (
        f"#!/bin/bash\n"
        f"source ~/locust-env/bin/activate\n"
        f"cd ~\n"
        f"locust -f ~/locustfile.py "
        f"--host http://192.168.100.100:5001 "
        f"-u {initial_users} -r {spawn_rate} "
        f"--run-time {total_duration}s "
        f"--web-port {LOCUST_WEB_PORT} "
        f"--autostart "
        f"--autoquit 30 "
        f"--csv ~/locust_results "
        f"--csv-full-history "
        f"> ~/locust.log 2>&1\n"
    )

    # Write the script to master
    write_cmd = f"cat > ~/start_locust.sh << 'SCRIPT_EOF'\n{startup_script}SCRIPT_EOF"
    ssh_cmd(user, master, write_cmd, timeout=10)
    ssh_cmd(user, master, "chmod +x ~/start_locust.sh", timeout=10)

    # Launch the script in background using nohup + ssh -f
    subprocess.Popen(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
         "-f", f"{user}@{master}",
         "nohup ~/start_locust.sh &"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(3)

    # Wait for Locust web API to become available
    locust_api = f"http://{master}:{LOCUST_WEB_PORT}/stats/requests"
    print(f"  ⏳ Waiting for Locust web API at {locust_api}...")
    for attempt in range(30):
        try:
            response = urllib.request.urlopen(locust_api, timeout=3)
            data = json.loads(response.read())
            print(f"  ✅ Locust web API ready (state: {data.get('state', 'unknown')})")
            return True
        except Exception:
            time.sleep(1)

    print("  ❌ Locust web API not available after 30s")
    return False


def change_locust_load(master, target_users, spawn_rate):
    """Change the number of Locust users via REST API on the master node."""
    data = urllib.parse.urlencode({
        "user_count": target_users,
        "spawn_rate": spawn_rate,
    }).encode()

    try:
        req = urllib.request.Request(
            f"http://{master}:{LOCUST_WEB_PORT}/swarm",
            data=data,
            method="PATCH"
        )
        urllib.request.urlopen(req, timeout=5)
        return True
    except Exception as e:
        print(f"  ⚠️  Failed to change Locust load: {e}")
        return False


def stop_locust_on_master(user, master):
    """Stop Locust on the master node."""
    print("  🦗 Stopping Locust on master...")
    ssh_cmd(user, master, "pkill -f 'locust' 2>/dev/null || true")
    time.sleep(2)
    print("  ✅ Locust stopped")


def collect_locust_results(user, master, results_dir):
    """Copy Locust CSV results from master to local results directory."""
    print("  📥 Collecting Locust results from master...")
    csv_files = ["locust_results_stats.csv", "locust_results_stats_history.csv",
                 "locust_results_failures.csv", "locust_results_exceptions.csv"]

    for f in csv_files:
        result = subprocess.run(
            ["scp", "-o", "StrictHostKeyChecking=no",
             f"{user}@{master}:~/{f}", results_dir],
            capture_output=True, timeout=30
        )
        if result.returncode == 0:
            print(f"    ✅ {f}")

    # Also grab the locust log
    subprocess.run(
        ["scp", "-o", "StrictHostKeyChecking=no",
         f"{user}@{master}:~/locust.log", os.path.join(results_dir, "locust.log")],
        capture_output=True, timeout=30
    )


def start_intent_loop(config_path, duration_minutes, results_dir, debug_llm=False):
    """Start main.py in background on the SDN controller."""
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

    log_file = open(os.path.join(results_dir, "intent_loop.log"), "w")
    intent_proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    return intent_proc, log_file


def run_load_schedule(master, load_pattern, interval, spawn_rate):
    """Execute the staged load pattern by changing Locust user count at intervals."""
    print(f"\n📊 Load Schedule:")
    for i, users in enumerate(load_pattern):
        start = i * interval
        end = start + interval
        print(f"  [{start:>4}s - {end:>4}s] → {users} users")

    # Wait for Locust to fully ramp up initial users
    time.sleep(10)

    for i, users in enumerate(load_pattern):
        if i == 0:
            print(f"\n⏱️  Stage 1/{len(load_pattern)}: {users} users (already active)")
            remaining = interval - 10
            time.sleep(max(0, remaining))
        else:
            print(f"\n⏱️  Stage {i+1}/{len(load_pattern)}: changing to {users} users")
            success = change_locust_load(master, users, spawn_rate)
            if success:
                print(f"  ✅ Load changed to {users} users")
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
    locust_stats = os.path.join(results_dir, "locust_results_stats.csv")
    if os.path.exists(locust_stats):
        print(f"\n  Locust stats: {locust_stats}")

    locust_history = os.path.join(results_dir, "locust_results_stats_history.csv")
    if os.path.exists(locust_history):
        print(f"  Locust history: {locust_history}")

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

    # Get master connection info
    user, master = get_master_info(args.config)

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
        "master_node": master,
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
    print(f"  Master node: {user}@{master}")
    print(f"  Locust: runs on master, controlled via REST API")
    print(f"  Results dir: {results_dir}/")

    if args.dry_run:
        print("\n  [DRY RUN] - would execute the above plan")
        return

    # Step 1: Reset cluster
    if not args.skip_reset:
        reset_cluster(args.config)
    else:
        print("\n⏭️  Skipping cluster reset")

    # Step 2: Start Intent Watch Loop (on SDN controller)
    intent_proc, intent_log = start_intent_loop(
        config_path=args.config,
        duration_minutes=int(total_duration_min) + 2,
        results_dir=results_dir,
        debug_llm=args.debug_llm,
    )

    # Wait for intent loop to initialize
    time.sleep(5)

    # Step 3: Start Locust on master node
    locust_ok = start_locust_on_master(
        user=user,
        master=master,
        initial_users=load_pattern[0],
        spawn_rate=args.spawn_rate,
        total_duration=total_duration_s,
    )

    if not locust_ok:
        print("❌ Failed to start Locust. Aborting experiment.")
        intent_proc.send_signal(signal.SIGINT)
        intent_proc.wait(timeout=15)
        intent_log.close()
        return

    # Step 4: Execute load schedule
    try:
        run_load_schedule(master, load_pattern, args.interval, args.spawn_rate)
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")
    finally:
        # Step 5: Cleanup
        print("\n🛑 Stopping processes...")

        # Stop Locust on master
        stop_locust_on_master(user, master)

        # Collect Locust results from master
        collect_locust_results(user, master, results_dir)

        # Wait for Intent Loop to finish
        print("  ⏳ Waiting for Intent Watch Loop to finish...")
        try:
            intent_proc.wait(timeout=180)
        except subprocess.TimeoutExpired:
            intent_proc.send_signal(signal.SIGINT)
            intent_proc.wait(timeout=15)
        print("  ✅ Intent Watch Loop stopped")

        intent_log.close()

        # Step 6: Summary
        collect_summary(results_dir, load_pattern, args.interval)


if __name__ == "__main__":
    main()