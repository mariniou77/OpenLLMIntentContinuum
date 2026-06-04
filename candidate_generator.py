"""
Candidate Action Generator

Generates a set of 4 candidate actions (A/B/C/D) from the current
structured system state for the LLM to choose from.

This module bridges the gap between raw cluster state and the structured
candidate_actions format used by the 8-message LLM conversation. The
generated candidates follow the same priority logic as the test suite:
  1. Network congestion  → flow_scheduling
  2. Node overload       → service_placement
  3. Compute bottleneck  → increase_cpu / add_replica
  4. Over-provisioning   → reduce_cpu / remove_replica
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Action ID labels
ACTION_IDS = ["A", "B", "C", "D"]

# CPU step sizes (millicores)
CPU_INCREASE_STEP = 300
CPU_DECREASE_STEP = 100
CPU_MIN = 100
CPU_MAX = 1500


class CandidateActionGenerator:
    """Generates candidate actions from structured cluster state."""

    def __init__(self, config: dict):
        self.config = config
        self.actions_enabled = config.get("actions", {})

        # Build deployment config lookup
        self._dep_configs = {}
        for dep in config.get("kubernetes", {}).get("deployments", []):
            name = dep.get("name", "")
            if name:
                self._dep_configs[name.lower()] = dep

    def generate(self, structured_state: dict, violation_type: str) -> List[Dict[str, Any]]:
        """
        Generate up to 4 candidate actions for the LLM to choose from.

        Args:
            structured_state: Output of DataCollector.build_structured_state()
            violation_type: "UPPER_THRESHOLD_EXCEEDED" or "LOWER_THRESHOLD_EXCEEDED"

        Returns:
            List of candidate action dicts, each with id, type, and type-specific fields.
        """
        if violation_type == "UPPER_THRESHOLD_EXCEEDED":
            candidates = self._generate_upper(structured_state)
        else:
            candidates = self._generate_lower(structured_state)

        # Assign IDs (A, B, C, D) and cap at 4
        candidates = candidates[:4]
        for i, c in enumerate(candidates):
            c["id"] = ACTION_IDS[i]

        return candidates

    # ------------------------------------------------------------------ #
    #  UPPER violation — add resources / fix bottleneck
    # ------------------------------------------------------------------ #

    def _generate_upper(self, state: dict) -> list:
        candidates = []
        services = state.get("services", [])
        nodes = state.get("nodes", [])
        network = state.get("network", {})
        critical_path = state.get("application", {}).get("critical_path", [])
        congestion = network.get("congestion_score", 0)

        if not services:
            return candidates

        # Sort services on critical path by cpu_util_pct descending
        # Only include services that are in the config's deployments list (avoid offering
        # non-scalable services like db-deployment that the validator would reject)
        path_set = set(critical_path)
        on_path = [s for s in services if s["name"] in path_set and s["name"].lower() in self._dep_configs]
        on_path.sort(key=lambda s: s["cpu_util_pct"], reverse=True)
        if not on_path:
            on_path = [s for s in sorted(services, key=lambda s: s["cpu_util_pct"], reverse=True)
                       if s["name"].lower() in self._dep_configs]

        # Prefer services that hit their CPU ceiling but still have replica headroom.
        # Their cpu_util_pct is artificially low because they are being throttled, so
        # they won't rank first in the cpu-descending sort — but they ARE the bottleneck
        # and can only scale horizontally. Surfacing them as the primary target ensures
        # add_replica appears as a top candidate (A/B) rather than being crowded out.
        target_svc = None
        for svc in on_path:
            if svc.get("cpu_max_reached") and not svc.get("replica_max_reached"):
                target_svc = svc
                break
        # Standard fallback: highest-CPU service that isn't fully blocked
        if target_svc is None:
            for svc in on_path:
                if not (svc.get("cpu_max_reached") and svc.get("replica_max_reached")):
                    target_svc = svc
                    break
        # If everything is fully blocked, use the first one (LLM will see constraints)
        if target_svc is None:
            target_svc = on_path[0]

        # --- Priority 1: Network congestion → flow_scheduling ---
        if congestion > 0.8 and self.actions_enabled.get("flow_scheduling", False):
            hot_links = network.get("hot_links", [])
            desc = "Reroute traffic to bypass congested links"
            if hot_links:
                link_names = [h["name"] for h in hot_links[:2]]
                desc = f"Reroute traffic to bypass congested {', '.join(link_names)}"
            candidates.append({
                "type": "flow_scheduling",
                "description": desc,
                "new_path": []  # ONOS will compute the actual path
            })

        # --- Priority 2: Node overload → service_placement ---
        target_node = target_svc.get("node", "")
        node_cpu = self._get_node_cpu(nodes, target_node)
        if node_cpu > 80 and self.actions_enabled.get("service_placement", False):
            least_loaded = self._find_least_loaded_worker(nodes, exclude=target_node)
            if least_loaded:
                candidates.append({
                    "type": "service_placement",
                    "target": target_svc["name"],
                    "from_node": target_node,
                    "to_node": least_loaded
                })

        # --- Priority 3: Compute scaling ---
        # increase_cpu if not at max
        if not target_svc.get("cpu_max_reached") and self.actions_enabled.get("vertical_scaling", True):
            new_cpu = min(target_svc["cpu_limit_m"] + CPU_INCREASE_STEP, CPU_MAX)
            if new_cpu != target_svc["cpu_limit_m"]:
                candidates.append({
                    "type": "increase_cpu",
                    "target": target_svc["name"],
                    "from_m": target_svc["cpu_limit_m"],
                    "to_m": new_cpu
                })

        # add_replica if not at max
        if not target_svc.get("replica_max_reached") and self.actions_enabled.get("horizontal_scaling", True):
            dep_cfg = self._dep_configs.get(target_svc["name"].lower(), {})
            max_rep = dep_cfg.get("max_replicas", 5)
            new_rep = min(target_svc["replicas"] + 1, max_rep)
            if new_rep != target_svc["replicas"]:
                candidates.append({
                    "type": "add_replica",
                    "target": target_svc["name"],
                    "from_replicas": target_svc["replicas"],
                    "to_replicas": new_rep
                })

        # --- Fill remaining slots with alternatives ---
        if len(candidates) < 4:
            candidates = self._fill_upper_alternatives(
                candidates, on_path, target_svc, nodes
            )

        return candidates

    # ------------------------------------------------------------------ #
    #  LOWER violation — remove resources / save costs
    # ------------------------------------------------------------------ #

    def _generate_lower(self, state: dict) -> list:
        candidates = []
        services = state.get("services", [])

        if not services:
            return candidates

        # Separate multi-replica services from single-replica (only scalable deployments)
        scalable = [s for s in services if s["name"].lower() in self._dep_configs]
        multi_rep = [s for s in scalable if s["replicas"] > 1]
        multi_rep.sort(key=lambda s: s["cpu_util_pct"])  # lowest CPU first = most over-provisioned

        # --- remove_replica for multi-replica services ---
        for svc in multi_rep:
            if len(candidates) >= 4:
                break
            if self.actions_enabled.get("horizontal_scaling", True):
                candidates.append({
                    "type": "remove_replica",
                    "target": svc["name"],
                    "from_replicas": svc["replicas"],
                    "to_replicas": svc["replicas"] - 1
                })

        # --- reduce_cpu by absolute waste ---
        waste_list = []
        for svc in scalable:
            waste = svc["cpu_limit_m"] * (100 - svc["cpu_util_pct"]) / 100
            new_cpu = max(svc["cpu_limit_m"] - CPU_DECREASE_STEP, CPU_MIN)
            if new_cpu != svc["cpu_limit_m"]:
                waste_list.append((waste, svc, new_cpu))
        waste_list.sort(key=lambda x: x[0], reverse=True)  # highest waste first

        for waste, svc, new_cpu in waste_list:
            if len(candidates) >= 4:
                break
            # Avoid duplicate target+type
            if any(c.get("target") == svc["name"] and c["type"] == "reduce_cpu" for c in candidates):
                continue
            if self.actions_enabled.get("vertical_scaling", True):
                candidates.append({
                    "type": "reduce_cpu",
                    "target": svc["name"],
                    "from_m": svc["cpu_limit_m"],
                    "to_m": new_cpu
                })

        return candidates

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _fill_upper_alternatives(
        self, candidates: list, on_path: list, primary: dict, nodes: list
    ) -> list:
        """Fill remaining candidate slots with alternative actions on different services."""
        used_targets = {(c.get("type"), c.get("target")) for c in candidates}

        for svc in on_path:
            if len(candidates) >= 4:
                break
            if svc["name"] == primary["name"]:
                continue
            if svc.get("cpu_max_reached") and svc.get("replica_max_reached"):
                continue

            # Try increase_cpu
            if not svc.get("cpu_max_reached") and ("increase_cpu", svc["name"]) not in used_targets:
                if self.actions_enabled.get("vertical_scaling", True):
                    new_cpu = min(svc["cpu_limit_m"] + CPU_INCREASE_STEP, CPU_MAX)
                    if new_cpu != svc["cpu_limit_m"] and len(candidates) < 4:
                        candidates.append({
                            "type": "increase_cpu",
                            "target": svc["name"],
                            "from_m": svc["cpu_limit_m"],
                            "to_m": new_cpu
                        })
                        used_targets.add(("increase_cpu", svc["name"]))

            # Try add_replica
            if not svc.get("replica_max_reached") and ("add_replica", svc["name"]) not in used_targets:
                if self.actions_enabled.get("horizontal_scaling", True) and len(candidates) < 4:
                    dep_cfg = self._dep_configs.get(svc["name"].lower(), {})
                    max_rep = dep_cfg.get("max_replicas", 5)
                    new_rep = min(svc["replicas"] + 1, max_rep)
                    if new_rep != svc["replicas"]:
                        candidates.append({
                            "type": "add_replica",
                            "target": svc["name"],
                            "from_replicas": svc["replicas"],
                            "to_replicas": new_rep
                        })
                        used_targets.add(("add_replica", svc["name"]))

        return candidates

    @staticmethod
    def _get_node_cpu(nodes: list, node_name: str) -> float:
        """Get CPU utilization for a specific node."""
        for n in nodes:
            if n["name"] == node_name:
                return n.get("cpu_util_pct", 0)
        return 0.0

    @staticmethod
    def _find_least_loaded_worker(nodes: list, exclude: str = "") -> Optional[str]:
        """Find the worker node with lowest CPU, excluding master and a given node."""
        workers = [
            n for n in nodes
            if n["name"] != exclude and "master" not in n["name"].lower()
        ]
        if not workers:
            return None
        return min(workers, key=lambda n: n.get("cpu_util_pct", 100))["name"]
