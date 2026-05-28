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
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.parse
from datetime import datetime
from pathlib import Path


def _find_locust() -> str:
    """Return the locust executable path, preferring a local venv."""
    venv_bin = os.path.expanduser("~/locust-venv/bin/locust")
    if os.path.exists(venv_bin):
        return venv_bin
    found = shutil.which("locust")
    return found if found else "locust"


LOCUST_BIN = _find_locust()


# ── Default Experiment Configuration ────────────────────────────────────────
# Load pattern: 20 users at peak saturate ms3 (500m CPU, ~3.7 req/s capacity) and
# trigger upper-threshold violations; 5 users ease off below lower threshold.
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

    # Ensure ms3 fwatchdog allows enough time for SSD model cold-start.
    # Only set if not already correct — avoids triggering an unnecessary rollout.
    result = ssh_cmd(user, master,
        'kubectl get deployment microservice3-deployment -o jsonpath="{.spec.template.spec.containers[0].env}"')
    if "exec_timeout" not in result.stdout:
        ssh_cmd(user, master,
            "kubectl set env deployment/microservice3-deployment exec_timeout=90s -c nginx")
        print("  ✅ microservice3-deployment: exec_timeout=90s set (fwatchdog SSD cold-start)")
    else:
        print("  ✅ microservice3-deployment: exec_timeout already configured")

    # Wait for all rollouts to complete (set resources + possible set env above)
    print("  ⏳ Waiting for rollouts to complete...")
    for dep_name in ["microservice1-deployment", "microservice2-deployment",
                     "microservice3-deployment", "microservice4-deployment"]:
        ssh_cmd(user, master, f"kubectl rollout status deployment/{dep_name} --timeout=120s", timeout=130)
    print("  ✅ All rollouts complete")

    # Always force-restart ms3 regardless of whether its resources changed.
    # kubectl set resources is idempotent: if values are unchanged it skips the rollout,
    # leaving an old pod alive that may have accumulated CLOSE_WAIT connections from
    # a previous experiment. A fresh pod guarantees a clean connection state.
    print("  ♻️  Force-restarting ms3 pod to guarantee clean connection state...")
    ssh_cmd(user, master, "kubectl delete pod -l app=microservice3 --force --grace-period=0")
    ssh_cmd(user, master, "kubectl rollout status deployment/microservice3-deployment --timeout=120s", timeout=130)
    print("  ✅ ms3 pod restarted")

    # Verify pods are running
    result = ssh_cmd(user, master, "kubectl get pods --no-headers | grep -c Running")
    running = result.stdout.strip()
    print(f"  ✅ {running} pods running")

    # Pre-warm the application to confirm the full chain is healthy before Locust starts.
    import yaml as _yaml
    with open(config_path) as _f:
        _cfg = _yaml.safe_load(_f)
    _app = _cfg.get("application", {})
    _entry = _app.get("entry_point", "http://10.56.1.209:5001/resize")
    _image = _app.get("test_image", "/home/cc/OpenLLMIntentContinuum/images/family.jpg")
    _webhooks = _app.get("webhooks", "")
    _db_url = _app.get("db_url", "")
    _logs_url = _app.get("logs_url", "")
    print("  🔥 Pre-warming application (waiting for ms3 SSD model cold-start, up to 90s)...")
    try:
        warmup_result = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
             "-X", "POST",
             "-F", f"image=@{_image}",
             "-H", "X-Special-Object: person",
             "-H", f"X-Webhooks: {_webhooks}",
             "-H", f"X-Central-DB-URL: {_db_url}",
             "-H", f"X-Logs-URL: {_logs_url}",
             _entry, "--max-time", "90"],
            capture_output=True, text=True, timeout=100
        )
        code = warmup_result.stdout.strip()
        if code == "200":
            print("  ✅ Application warmed up (HTTP 200)")
        else:
            print(f"  ⚠️  Warm-up response code: {code or 'timeout'} — continuing anyway")
    except Exception as _e:
        print(f"  ⚠️  Warm-up failed: {_e} — continuing anyway")


def warmup_llm(config_path: str):
    """
    Send a trivial prompt to Ollama to force the model into GPU VRAM before the
    experiment starts. Without this, the first real LLM call (a violation decision)
    pays a ~15s cold-load penalty instead of the normal ~1s inference time.
    """
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    ollama_url = cfg.get("endpoints", {}).get("ollama", "http://10.56.2.204:11434")
    model = cfg.get("llm", {}).get("model", "qwen3.5:4b")

    print(f"\n🔥 Pre-warming LLM ({model} on {ollama_url})...")
    url = f"{ollama_url}/api/generate"
    payload = json.dumps({
        "model": model,
        "prompt": "Hi",
        "stream": False,
        "think": False,
        "options": {"temperature": 0},
    }).encode()

    try:
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=60) as resp:
            resp.read()
        elapsed = time.time() - t0
        print(f"  ✅ LLM warmed up in {elapsed:.1f}s (model loaded into VRAM)")
    except Exception as e:
        print(f"  ⚠️  LLM warm-up failed: {e} — continuing anyway")


def start_locust_locally(config_path, initial_users, spawn_rate, total_duration, results_dir):
    """
    Start Locust locally on the SDN controller as a subprocess.
    Locust runs with web UI enabled so load can be changed via REST API.
    Uses DIRECT_CURL=1 so requests are sent without an SSH wrapper.
    """
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    app = cfg.get("application", {})
    entry_point = app.get("entry_point", "http://10.56.1.209:5001/resize")
    remote_image = app.get("test_image", "images/family.jpg")
    webhooks = app.get("webhooks", "")
    db_url = app.get("db_url", "")
    logs_url = app.get("logs_url", "")

    print(f"\n🦗 Starting Locust locally (entry: {entry_point})...")

    # Kill any stale locust process
    subprocess.run(["pkill", "-f", "locust"], capture_output=True)
    time.sleep(1)

    env = os.environ.copy()
    env.update({
        "DIRECT_CURL": "1",
        "SDN_ENTRY_POINT": entry_point,
        "REMOTE_IMAGE": remote_image,
        "WEBHOOKS": webhooks,
        "DB_URL": db_url,
        "LOGS_URL": logs_url,
    })

    cmd = [
        LOCUST_BIN, "-f", "locustfile.py",
        "--host", entry_point,
        "-u", str(initial_users), "-r", str(spawn_rate),
        "--run-time", f"{total_duration}s",
        "--web-port", str(LOCUST_WEB_PORT),
        "--autostart",
        "--autoquit", "30",
        "--csv", os.path.join(results_dir, "locust_results"),
        "--csv-full-history",
    ]

    log_file = open(os.path.join(results_dir, "locust.log"), "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)

    # Wait for Locust web API to become available
    locust_api = f"http://localhost:{LOCUST_WEB_PORT}/stats/requests"
    print(f"  ⏳ Waiting for Locust web API at {locust_api}...")
    for _ in range(30):
        try:
            response = urllib.request.urlopen(locust_api, timeout=3)
            data = json.loads(response.read())
            print(f"  ✅ Locust web API ready (state: {data.get('state', 'unknown')})")
            return proc, log_file
        except Exception:
            time.sleep(1)

    print("  ❌ Locust web API not available after 30s")
    return None, log_file


def change_locust_load(target_users, spawn_rate):
    """Change the number of Locust users via the local Locust REST API."""
    payload = json.dumps({
        "user_count": target_users,
        "spawn_rate": spawn_rate,
    }).encode()

    for method in ("POST", "PUT"):
        try:
            req = urllib.request.Request(
                f"http://localhost:{LOCUST_WEB_PORT}/swarm",
                data=payload,
                headers={"Content-Type": "application/json"},
                method=method,
            )
            urllib.request.urlopen(req, timeout=10)
            return True
        except Exception as e:
            last_err = e
    print(f"  ⚠️  Failed to change Locust load: {last_err}")
    return False


def stop_locust_locally(proc):
    """Stop the local Locust subprocess."""
    print("  🦗 Stopping Locust...")
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
    print("  ✅ Locust stopped")


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


def run_load_schedule(load_pattern, interval, spawn_rate):
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
            success = change_locust_load(users, spawn_rate)
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
    print(f"  Master node: {user}@{master} (for cluster reset only)")
    print(f"  Locust: runs locally on SDN controller, controlled via REST API")
    print(f"  Results dir: {results_dir}/")

    if args.dry_run:
        print("\n  [DRY RUN] - would execute the above plan")
        return

    # Step 1: Reset cluster
    if not args.skip_reset:
        reset_cluster(args.config)
    else:
        print("\n⏭️  Skipping cluster reset")

    # Step 1b: Warm up LLM (force model into GPU VRAM before first real decision)
    warmup_llm(args.config)

    # Step 2: Start Intent Watch Loop (on SDN controller)
    intent_proc, intent_log = start_intent_loop(
        config_path=args.config,
        duration_minutes=int(total_duration_min) + 2,
        results_dir=results_dir,
        debug_llm=args.debug_llm,
    )

    # Wait for intent loop to initialize
    time.sleep(5)

    # Step 3: Start Locust locally on SDN controller
    locust_proc, locust_log = start_locust_locally(
        config_path=args.config,
        initial_users=load_pattern[0],
        spawn_rate=args.spawn_rate,
        total_duration=total_duration_s,
        results_dir=results_dir,
    )

    if locust_proc is None:
        print("❌ Failed to start Locust. Aborting experiment.")
        intent_proc.send_signal(signal.SIGINT)
        intent_proc.wait(timeout=15)
        intent_log.close()
        locust_log.close()
        return

    # Step 4: Execute load schedule
    try:
        run_load_schedule(load_pattern, args.interval, args.spawn_rate)
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")
    finally:
        # Step 5: Cleanup
        print("\n🛑 Stopping processes...")

        stop_locust_locally(locust_proc)
        locust_log.close()

        # Wait for Intent Loop to finish
        print("  ⏳ Waiting for Intent Watch Loop to finish...")
        try:
            intent_proc.wait(timeout=180)
        except subprocess.TimeoutExpired:
            intent_proc.send_signal(signal.SIGINT)
            try:
                intent_proc.wait(timeout=120)  # Allow graceful shutdown (may need to finish a 60s wait)
            except subprocess.TimeoutExpired:
                intent_proc.kill()
                intent_proc.wait()
        print("  ✅ Intent Watch Loop stopped")

        intent_log.close()

        # Step 6: Summary
        collect_summary(results_dir, load_pattern, args.interval)


if __name__ == "__main__":
    main()