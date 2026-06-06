"""
Candidate Filter — Hybrid Python+LLM Architecture

Enforces Rule 3 (history override) deterministically in Python before the
LLM sees the candidate list.  This eliminates two systematic failure patterns
observed in prompt-only experiments:

  1. Rule 3 over-application: LLM blocks an entire service after one action
     type fails, even though a different action type on that service (e.g.
     add_replica after increase_cpu failed) is still valid and untried.

  2. Rule 3 inconsistency: LLM applies the rule strictly to one service but
     ignores repeated failures on another.

Python enforces the rule uniformly as: block the exact (action_type, target)
pair that appeared in history with result="violation_persisted".  Nothing else
on that service is touched.

A post-selection validator catches position-bias cases where the LLM's
reasoning names one service but the selected letter maps to a different one.
"""

import logging
import random
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Maps executor-level action names (returned by LLM / action_executor) back to
# the candidate-level type names used by candidate_generator and history.
_EXECUTOR_TO_CANDIDATE_TYPES: Dict[str, set] = {
    "vertical_scaling":   {"increase_cpu", "reduce_cpu"},
    "horizontal_scaling": {"add_replica",  "remove_replica"},
    "service_placement":  {"service_placement"},
    "flow_scheduling":    {"flow_scheduling"},
}

ACTION_IDS = ["A", "B", "C", "D"]


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
    Deterministic pre-filter for candidate actions.

    Call filter() before passing candidates to the LLM, and
    validate_selection() after the LLM returns its recommendation.
    """

    def __init__(self, config: dict):
        self.config = config

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def filter(
        self,
        candidates: List[Dict[str, Any]],
        history_entries: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Remove candidates that are blocked by Rule 3 (repeated failure in
        history) and re-assign sequential letter IDs to the remainder.

        Args:
            candidates:      Raw candidates from CandidateActionGenerator
                             (already lettered A/B/C/D).
            history_entries: Structured history from DecisionHistory
                             .get_structured_history().

        Returns:
            Filtered + re-lettered candidate list.  Never empty — if all
            candidates are blocked the full original list is returned as a
            fallback (logged as a warning).
        """
        blocked = self._build_blocked_set(history_entries)

        if not blocked:
            return candidates  # Nothing to filter

        valid   = [c for c in candidates
                   if (c.get("type", ""), c.get("target", "")) not in blocked]
        removed = [c for c in candidates
                   if (c.get("type", ""), c.get("target", "")) in blocked]

        if removed:
            for c in removed:
                logger.info(
                    f"  [RULE-3 FILTER] removed [{c['id']}] "
                    f"{c['type']} → {c.get('target', '?')} "
                    f"(repeated failure in history)"
                )

        if not valid:
            logger.warning(
                "CandidateFilter: all candidates blocked by Rule 3 — "
                "returning full list as fallback to avoid empty decision"
            )
            return candidates

        # Shuffle before re-lettering to eliminate residual position bias
        random.shuffle(valid)

        # Re-assign sequential IDs so the LLM sees a compact A/B/C list
        filtered = []
        for i, c in enumerate(valid[:4]):
            c_copy = dict(c)
            c_copy["id"] = ACTION_IDS[i]
            filtered.append(c_copy)

        logger.info(
            f"CandidateFilter: {len(candidates)} → {len(filtered)} candidates "
            f"({len(removed)} removed by Rule 3)"
        )
        return filtered

    def validate_selection(
        self,
        recommendation: dict,
        valid_candidates: List[Dict[str, Any]],
    ) -> Tuple[dict, bool]:
        """
        Confirm the LLM's selected action corresponds to one of the valid
        (filtered) candidates.  If not, this is a position-bias event: the
        LLM's reasoning pointed to a valid service but its selected letter
        resolved to a blocked/absent candidate.

        In that case the first valid candidate is substituted and the
        override is logged.

        Args:
            recommendation:   Raw recommendation from DecisionMaker.
            valid_candidates: Filtered candidate list that was shown to LLM.

        Returns:
            (final_recommendation, was_overridden) tuple.
            was_overridden=True means a position-bias correction was applied.
        """
        if not valid_candidates:
            return recommendation, False

        action      = recommendation.get("action", "")
        target_dep  = recommendation.get("parameters", {}).get("deployment_name", "")

        # Determine which candidate types correspond to the LLM's action
        valid_ctypes = _EXECUTOR_TO_CANDIDATE_TYPES.get(action, set())

        for c in valid_candidates:
            if c.get("target") == target_dep and c.get("type") in valid_ctypes:
                return recommendation, False  # Selection is valid

        # Position bias: LLM selected something outside the valid set
        first_valid_action = _candidate_to_executor_action(valid_candidates[0])
        logger.warning(
            f"CandidateFilter: position bias detected — "
            f"LLM selected '{action}' on '{target_dep}' which is not in the "
            f"filtered candidate set.  Overriding with first valid candidate: "
            f"[{valid_candidates[0]['id']}] {valid_candidates[0]['type']} → "
            f"{valid_candidates[0].get('target', '?')}"
        )
        return first_valid_action, True

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_blocked_set(self, history_entries: list) -> set:
        """
        Build the set of (action_type, target) pairs blocked by Rule 3.

        Rule 3: if the same (action_type, target) pair appears in history
        with result="violation_persisted", do not offer it again.

        Only the EXACT pair is blocked.  Other action types on the same
        service, and the same action type on other services, remain valid.
        """
        blocked = set()
        for entry in history_entries:
            action_taken = entry.get("action_taken", {})
            if action_taken.get("result") == "violation_persisted":
                atype  = action_taken.get("type",   "")
                target = action_taken.get("target", "")
                if atype and target:
                    blocked.add((atype, target))
        return blocked
