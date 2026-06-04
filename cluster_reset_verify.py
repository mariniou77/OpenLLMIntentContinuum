#!/usr/bin/env python3
"""
Cluster Reset + State Verification Gate

Run this between every experiment run. Resets the K8s cluster to its default
configuration and then verifies every deployment matches expected state.

Exit code 0 = VERIFICATION PASSED — safe to start the next run.
Exit code 1 = VERIFICATION FAILED — do NOT start the next run.

Usage:
    python3 cluster_reset_verify.py                        # reset then verify (normal use)
    python3 cluster_reset_verify.py --verify-only          # skip reset, only inspect state
    python3 cluster_reset_verify.py --config config.yaml   # custom config path
"""

import argparse
import sys

import yaml

from run_experiment import get_master_info, reset_cluster, ssh_cmd
from run_ablation import wait_for_cluster_stable


# ── Helpers ──────────────────────────────────────────────────────────────────

def _to_millicores(cpu_str: str) -> int:
    """Convert K8s CPU strings ('300m', '1', '1500m') to integer millicores."""
    s = cpu_str.strip()
    if s.endswith("m"):
        try:
            return int(s[:-1])
        except ValueError:
            return -1
    try:
        return int(float(s) * 1000)
    except ValueError:
        return -1


# ── Verification ─────────────────────────────────────────────────────────────

def verify_cluster(config_path: str = "config.yaml") -> bool:
    """
    Query K8s for each deployment's current state and compare against expected
    defaults from config.yaml.  Returns True if all checks pass.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    user, master = get_master_info(config_path)
    deployments = cfg.get("kubernetes", {}).get("deployments", [])

    print("\n=== CLUSTER STATE VERIFICATION ===")

    all_pass = True

    for dep in deployments:
        name = dep["name"]
        app_label = name.replace("-deployment", "")
        expected_replicas = dep.get("min_replicas", 1)
        expected_cpu = dep.get("default_cpu", "300m")

        # --- replicas ---
        r = ssh_cmd(user, master,
                    f'kubectl get deployment {name} -o jsonpath="{{.spec.replicas}}"')
        try:
            actual_replicas = int(r.stdout.strip())
        except ValueError:
            actual_replicas = -1
        replicas_ok = actual_replicas == expected_replicas

        # --- CPU limit ---
        r = ssh_cmd(user, master,
                    f'kubectl get deployment {name} -o jsonpath='
                    f'"{{.spec.template.spec.containers[0].resources.limits.cpu}}"')
        actual_cpu_str = r.stdout.strip().strip("'\"") or "?"
        cpu_ok = _to_millicores(actual_cpu_str) == _to_millicores(expected_cpu)

        # --- running pods ---
        r = ssh_cmd(user, master,
                    f"kubectl get pods -l app={app_label} --no-headers 2>/dev/null"
                    f" | grep -c Running || true")
        try:
            running_pods = int(r.stdout.strip())
        except ValueError:
            running_pods = 0
        pods_ok = running_pods == expected_replicas

        dep_pass = replicas_ok and cpu_ok and pods_ok
        if not dep_pass:
            all_pass = False

        status = "PASS" if dep_pass else "FAIL"
        r_mark = "✓" if replicas_ok else "✗"
        c_mark = "✓" if cpu_ok else "✗"
        p_mark = "✓" if pods_ok else "✗"
        print(
            f"  [{status}] {name:<40}"
            f"  replicas={actual_replicas} (exp {expected_replicas}) {r_mark}"
            f"  cpu={actual_cpu_str} (exp {expected_cpu}) {c_mark}"
            f"  pods_running={running_pods} {p_mark}"
        )

    # --- node CPU total ---
    r = ssh_cmd(user, master, "kubectl top nodes --no-headers 2>/dev/null")
    total_cpu = 0
    node_lines = []
    for line in r.stdout.strip().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            m = _to_millicores(parts[1])
            total_cpu += m
            node_lines.append(f"{parts[0]}={parts[1]}")

    cpu_stable = total_cpu < 500
    if not cpu_stable:
        all_pass = False
    cpu_status = "PASS" if cpu_stable else "WARN"
    cpu_mark = "✓" if cpu_stable else "⚠"
    nodes_str = "  ".join(node_lines) if node_lines else "unavailable"
    print(
        f"  [{cpu_status}] Node CPU total: {total_cpu}m < 500m threshold {cpu_mark}"
        f"  ({nodes_str})"
    )

    # --- no stray HPA objects ---
    r = ssh_cmd(user, master,
                "kubectl get hpa --all-namespaces --no-headers 2>/dev/null | wc -l")
    try:
        hpa_count = int(r.stdout.strip())
    except ValueError:
        hpa_count = 0
    hpa_ok = (hpa_count == 0)
    if not hpa_ok:
        all_pass = False
    hpa_status = "PASS" if hpa_ok else "FAIL"
    hpa_mark = "✓" if hpa_ok else "✗  (fix: kubectl delete hpa --all)"
    print(f"  [{hpa_status}] No stray HPA objects: {hpa_count} found {hpa_mark}")

    print()
    if all_pass:
        print("=== VERIFICATION PASSED — READY FOR NEXT RUN ===")
    else:
        print("=== VERIFICATION FAILED — DO NOT START NEXT RUN ===")
        print("    Fix the issues above, then re-run cluster_reset_verify.py.")

    return all_pass


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reset K8s cluster to defaults and verify state before each experiment run."
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Skip cluster reset; only inspect and report current state."
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to IntentContinuum config (default: config.yaml)."
    )
    args = parser.parse_args()

    if not args.verify_only:
        print("=== CLUSTER RESET ===")
        reset_cluster(args.config)

        # Always purge any leftover HPA objects so a previously failed HPA run
        # cannot contaminate the next experiment's scaling behaviour.
        user, master = get_master_info(args.config)
        result = ssh_cmd(user, master,
                         "kubectl delete hpa --all -n default 2>/dev/null || true")
        leftover = result.stdout.strip()
        if leftover:
            print(f"  HPA purged: {leftover}")
        else:
            print("  No leftover HPA objects (clean)")

        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        wait_for_cluster_stable(
            master_ip=cfg["endpoints"]["kubernetes_master"],
            ssh_user=cfg["endpoints"].get("kubernetes_user", "cc"),
        )

    ok = verify_cluster(args.config)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
