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


# Expected pod-to-node placement (locked via nodeSelector in deployment.yml.j2)
EXPECTED_POD_PLACEMENT = {
    "microservice1-deployment": "worker2",
    "microservice2-deployment": "worker1",
    "microservice3-deployment": "worker1",
    "microservice4-deployment": "worker2",
    "db-deployment":            "master",
}

# Pod idle threshold: all microservice + db pods must be below this CPU (millicores)
# before an experiment run starts.
POD_IDLE_CPU_THRESHOLD_M = 20


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

    # --- pod placement ---
    # Verify each deployment's running pod is on its expected node (locked by nodeSelector).
    # A mismatch means the nodeSelector was removed or the scheduler placed the pod elsewhere.
    print()
    placement_all_ok = True
    r = ssh_cmd(user, master, "kubectl get pods -o wide --no-headers 2>/dev/null")
    pod_nodes: dict = {}
    for line in r.stdout.strip().splitlines():
        parts = line.split()
        # columns: NAME READY STATUS RESTARTS AGE IP NODE ...
        if len(parts) >= 7:
            pod_name = parts[0]
            node_name = parts[6]
            for dep_name in EXPECTED_POD_PLACEMENT:
                app = dep_name.replace("-deployment", "")
                if pod_name.startswith(app + "-") or pod_name.startswith(
                    dep_name.replace("-deployment", "-dep")
                ):
                    pod_nodes[dep_name] = node_name

    # Auto-remediate placement drift BEFORE judging (belt-and-suspenders vs reset_cluster's restore):
    # a service_placement migration that slipped through gets re-pinned + rolled, then re-read.
    drifted = [d for d, n in EXPECTED_POD_PLACEMENT.items()
               if pod_nodes.get(d) is not None and pod_nodes.get(d) != n]
    if drifted:
        for dep_name in drifted:
            expected_node = EXPECTED_POD_PLACEMENT[dep_name]
            print(f"  ⟳ {dep_name} on {pod_nodes.get(dep_name)} (exp {expected_node}) — re-pinning nodeSelector...")
            node_patch = ('{"spec":{"template":{"spec":{"nodeSelector":'
                          '{"kubernetes.io/hostname":"%s"}}}}}' % expected_node)
            ssh_cmd(user, master, f"kubectl patch deployment {dep_name} --type=strategic -p '{node_patch}'")
            ssh_cmd(user, master, f"kubectl rollout status deployment/{dep_name} --timeout=120s", timeout=130)
        # Re-read placement after remediation
        r = ssh_cmd(user, master, "kubectl get pods -o wide --no-headers 2>/dev/null")
        pod_nodes = {}
        for line in r.stdout.strip().splitlines():
            parts = line.split()
            if len(parts) >= 7:
                pod_name, node_name = parts[0], parts[6]
                for dep_name in EXPECTED_POD_PLACEMENT:
                    if pod_name.startswith(dep_name.replace("-deployment", "") + "-"):
                        pod_nodes[dep_name] = node_name

    for dep_name, expected_node in EXPECTED_POD_PLACEMENT.items():
        actual_node = pod_nodes.get(dep_name, "?")
        ok = actual_node == expected_node
        if not ok:
            all_pass = False
            placement_all_ok = False
        mark = "✓" if ok else "✗"
        status = "PASS" if ok else "FAIL"
        print(
            f"  [{status}] {dep_name:<40}  node={actual_node} "
            f"(exp {expected_node}) {mark}"
        )

    if not placement_all_ok:
        print("  ⚠  Pod placement mismatch — re-apply nodeSelectors and rollout restart.")

    # --- pod CPU idle gate (replaces unreliable total-node-CPU check) ---
    # All microservice and db pods must be idle (< POD_IDLE_CPU_THRESHOLD_M)
    # before we declare the cluster ready.  Node CPU is logged for information
    # but does NOT gate the pass/fail — master idles at 250–300m from K3s
    # control-plane overhead regardless of application load.
    print()
    r = ssh_cmd(user, master, "kubectl top pods -n default --no-headers 2>/dev/null")
    pod_cpu_ok = True
    pod_cpu_lines = []
    for line in r.stdout.strip().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        pod_name, cpu_str, _ = parts[0], parts[1], parts[2]
        cpu_m = _to_millicores(cpu_str)
        is_app_pod = any(
            pod_name.startswith(dep.replace("-deployment", ""))
            for dep in EXPECTED_POD_PLACEMENT
        )
        if not is_app_pod:
            continue
        idle = cpu_m < POD_IDLE_CPU_THRESHOLD_M
        if not idle:
            pod_cpu_ok = False
            all_pass = False
        mark = "✓" if idle else "⚠"
        pod_cpu_lines.append(f"{pod_name}={cpu_str}{mark}")

    idle_status = "PASS" if pod_cpu_ok else "WARN"
    idle_summary = "  ".join(pod_cpu_lines) if pod_cpu_lines else "no data"
    print(
        f"  [{idle_status}] Pod CPU idle (< {POD_IDLE_CPU_THRESHOLD_M}m): "
        f"{idle_summary}"
    )
    if not pod_cpu_ok:
        print(f"  ⚠  Some pods are still active — wait and re-verify before starting run.")

    # --- node CPU (informational only, does not gate pass/fail) ---
    r = ssh_cmd(user, master, "kubectl top nodes --no-headers 2>/dev/null")
    node_lines = []
    for line in r.stdout.strip().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            node_lines.append(f"{parts[0]}={parts[1]}")
    nodes_str = "  ".join(node_lines) if node_lines else "unavailable"
    print(f"  [INFO] Node CPU (informational): {nodes_str}")

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
