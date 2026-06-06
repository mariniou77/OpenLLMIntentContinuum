"""
Candidate Filter — post-selection validator.

Catches position-bias cases where the LLM's reasoning correctly identifies
a service but writes the wrong letter ID (e.g. reasons about ms3 but outputs
"A" which maps to ms2).  The validator substitutes the first candidate whose
(action_type, target) matches the LLM's stated reasoning.

History-based Rule-3 blocking has been removed; the system now operates
on a stateless current-snapshot model.
"""

import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Maps executor-level action names (returned by LLM / action_executor) back to
# the candidate-level type names used by candidate_generator.
_EXECUTOR_TO_CANDIDATE_TYPES: Dict[str, set] = {
    "vertical_scaling":   {"increase_cpu", "reduce_cpu"},
    "horizontal_scaling": {"add_replica",  "remove_replica"},
    "service_placement":  {"service_placement"},
    "flow_scheduling":    {"flow_scheduling"},
}


def _candidate_to_executor_action(candidate: dict) -> dict:
    """
    Convert a candidate dict (from CandidateActionGenerator) to the executor
    action format returned by DecisionMaker.  Mirrors the mapping in
    decision_maker.py so the filter stays self-contained.
    """
    ctype  = candidate.get("type", "")
    target = candidate.get("target", "")

    if ctype == "increase_cpu":
        cpu_m = candidate.get("to_m", 500)
        return {
            "action": "vertical_scaling",
            "parameters": {
                "deployment_name": target,
                "cpu_limit":    f"{cpu_m}m",
                "memory_limit": f"{cpu_m + 12}Mi",
            },
        }
    if ctype == "reduce_cpu":
        cpu_m = candidate.get("to_m", 400)
        return {
            "action": "vertical_scaling",
            "parameters": {
                "deployment_name": target,
                "cpu_limit":    f"{cpu_m}m",
                "memory_limit": f"{cpu_m + 12}Mi",
            },
        }
    if ctype == "add_replica":
        return {
            "action": "horizontal_scaling",
            "parameters": {
                "deployment_name": target,
                "replicas": candidate.get("to_replicas", 2),
            },
        }
    if ctype == "remove_replica":
        return {
            "action": "horizontal_scaling",
            "parameters": {
                "deployment_name": target,
                "replicas": candidate.get("to_replicas", 1),
            },
        }
    if ctype == "service_placement":
        return {
            "action": "service_placement",
            "parameters": {
                "deployment_name": target,
                "target_node":     candidate.get("to_node", ""),
                "source_node":     candidate.get("from_node", ""),
            },
        }
    if ctype == "flow_scheduling":
        return {
            "action": "flow_scheduling",
            "parameters": {"new_path": candidate.get("new_path", [])},
        }
    return {"action": "none", "parameters": {}}


class CandidateFilter:
    """
    Post-selection validator for LLM candidate choices.

    Call validate_selection() after the LLM returns its recommendation to
    catch position-bias letter-selection errors.
    """

    def __init__(self, config: dict):
        self.config = config

    def validate_selection(
        self,
        recommendation: dict,
        valid_candidates: List[Dict[str, Any]],
    ) -> Tuple[dict, bool]:
        """
        Confirm the LLM's selected action corresponds to one of the valid
        candidates.  If not, this is a position-bias event: the LLM's
        reasoning pointed to a valid service but its selected letter resolved
        to a different candidate.

        In that case the first valid candidate is substituted and the
        override is logged.

        Args:
            recommendation:   Raw recommendation from DecisionMaker.
            valid_candidates: Candidate list that was shown to LLM.

        Returns:
            (final_recommendation, was_overridden) tuple.
            was_overridden=True means a position-bias correction was applied.
        """
        if not valid_candidates:
            return recommendation, False

        action     = recommendation.get("action", "")
        target_dep = recommendation.get("parameters", {}).get("deployment_name", "")

        valid_ctypes = _EXECUTOR_TO_CANDIDATE_TYPES.get(action, set())

        for c in valid_candidates:
            if c.get("target") == target_dep and c.get("type") in valid_ctypes:
                return recommendation, False  # Selection is valid

        # Position bias: LLM selected something outside the valid set
        first_valid_action = _candidate_to_executor_action(valid_candidates[0])
        logger.warning(
            f"CandidateFilter: position bias detected — "
            f"LLM selected '{action}' on '{target_dep}' which is not in the "
            f"candidate set.  Overriding with first candidate: "
            f"[{valid_candidates[0]['id']}] {valid_candidates[0]['type']} → "
            f"{valid_candidates[0].get('target', '?')}"
        )
        return first_valid_action, True
