"""
Decision History Module

This module manages the history of past violations and LLM decisions.
It provides context to the LLM so it can make smarter decisions
based on what actions were taken previously in the same session.

The history now includes OUTCOME tracking - showing whether previous
actions improved or worsened the response time, giving the LLM
feedback on its decisions.

The history resets at the start of each experiment time window.
"""

import logging
from datetime import datetime
from typing import Optional
from collections import deque

logger = logging.getLogger(__name__)


class DecisionHistory:
    """
    Manages history of violations and decisions for LLM context.
    
    Stores the last N violations with their monitoring data,
    the decisions that were made, AND the outcomes of those decisions
    so the LLM can learn from past actions in the current session.
    
    Outcome tracking:
    - Records response time BEFORE action
    - Records response time AFTER action (on next check)
    - Calculates if action IMPROVED, WORSENED, or had NO_CHANGE
    - Includes this feedback in prompts to LLM
    """
    
    def __init__(self, max_entries: int = 3):
        """
        Initialize Decision History.
        
        Args:
            max_entries: Maximum number of past violations to store (default: 3)
        """
        self.max_entries = max_entries
        self.history = deque(maxlen=max_entries)
        self.session_start: Optional[datetime] = None
        self.violation_counter = 0
        
        # Pending outcome tracking - stores data needed to evaluate last decision
        self.pending_outcome: Optional[dict] = None
        
        logger.info(f"DecisionHistory initialized with max_entries={max_entries}")
    
    def reset(self):
        """
        Reset history for a new experiment session.
        
        Called at the start of each time window experiment.
        """
        self.history.clear()
        self.session_start = datetime.now()
        self.violation_counter = 0
        self.pending_outcome = None
        logger.info("Decision history reset for new session")
    
    def add_entry(
        self,
        violation_type: str,
        response_time: float,
        ema_response_time: float,
        monitoring_summary: str,
        deployments_summary: str,
        decision: dict
    ):
        """
        Add a new entry to the history.
        
        Called after the LLM makes a decision and action is executed.
        
        Args:
            violation_type: Type of violation (UPPER_THRESHOLD_EXCEEDED or LOWER_THRESHOLD_EXCEEDED)
            response_time: Current response time when violation occurred
            ema_response_time: EMA response time when violation occurred
            monitoring_summary: Compact monitoring data string
            deployments_summary: Compact deployments data string
            decision: The decision made by LLM {"action": "...", "parameters": {...}}
        """
        # First, update the outcome of the PREVIOUS decision if there is one
        self._update_pending_outcome(response_time, ema_response_time, violation_type)
        
        self.violation_counter += 1
        entry = {
            "timestamp": datetime.now().isoformat(),
            "violation_number": self.violation_counter,
            "violation_type": violation_type,
            "response_time": round(response_time, 3),
            "ema_response_time": round(ema_response_time, 3),
            "monitoring_summary": monitoring_summary,
            "deployments_summary": deployments_summary,
            "decision": decision,
            # Outcome fields - will be populated when next violation occurs
            "outcome": None,  # "IMPROVED", "WORSENED", "NO_CHANGE", "RESOLVED"
            "outcome_details": None  # Human-readable outcome description
        }
        
        self.history.append(entry)
        
        # Store pending outcome data for this decision
        self.pending_outcome = {
            "violation_number": self.violation_counter,
            "before_rt": response_time,
            "before_ema": ema_response_time,
            "violation_type": violation_type,
            "action": decision.get("action"),
            "parameters": decision.get("parameters", {})
        }
        
        logger.info(f"Added decision to history (total: {len(self.history)})")
        logger.debug(f"Entry: {entry}")
    
    def _update_pending_outcome(
        self, 
        current_rt: float, 
        current_ema: float,
        current_violation_type: Optional[str]
    ):
        """
        Update the outcome of the previous decision based on current metrics.
        
        Args:
            current_rt: Current response time
            current_ema: Current EMA response time
            current_violation_type: Current violation type (None if no violation)
        """
        if self.pending_outcome is None:
            return
        
        if not self.history:
            self.pending_outcome = None
            return
        
        # Find the entry that matches the pending outcome
        last_entry = self.history[-1]
        if last_entry["violation_number"] != self.pending_outcome["violation_number"]:
            # Entry was rotated out of history
            self.pending_outcome = None
            return
        
        before_rt = self.pending_outcome["before_rt"]
        before_ema = self.pending_outcome["before_ema"]
        prev_violation_type = self.pending_outcome["violation_type"]
        action = self.pending_outcome["action"]
        
        # Calculate the change using EMA (more stable than instantaneous RT)
        ema_change = current_ema - before_ema
        ema_change_pct = (ema_change / before_ema * 100) if before_ema > 0 else 0
        
        # Determine outcome based on violation type and EMA change
        if current_violation_type is None:
            # No more violation - problem resolved!
            outcome = "RESOLVED"
            outcome_details = f"EMA changed {before_ema:.3f}s → {current_ema:.3f}s ({ema_change_pct:+.1f}%), now within target range"
        elif prev_violation_type == "LOWER_THRESHOLD_EXCEEDED":
            # For LOWER threshold, we want EMA to INCREASE
            if ema_change > 0.02:  # Significant increase (using smaller threshold for EMA)
                outcome = "IMPROVED"
                outcome_details = f"EMA increased {before_ema:.3f}s → {current_ema:.3f}s ({ema_change_pct:+.1f}%) - correct direction"
            elif ema_change < -0.02:  # Decreased (wrong direction)
                outcome = "WORSENED"
                outcome_details = f"EMA decreased {before_ema:.3f}s → {current_ema:.3f}s ({ema_change_pct:+.1f}%) - WRONG direction, should increase"
            else:
                outcome = "NO_CHANGE"
                outcome_details = f"EMA unchanged {before_ema:.3f}s → {current_ema:.3f}s ({ema_change_pct:+.1f}%)"
        elif prev_violation_type == "UPPER_THRESHOLD_EXCEEDED":
            # For UPPER threshold, we want EMA to DECREASE
            if ema_change < -0.02:  # Significant decrease
                outcome = "IMPROVED"
                outcome_details = f"EMA decreased {before_ema:.3f}s → {current_ema:.3f}s ({ema_change_pct:+.1f}%) - correct direction"
            elif ema_change > 0.02:  # Increased (wrong direction)
                outcome = "WORSENED"
                outcome_details = f"EMA increased {before_ema:.3f}s → {current_ema:.3f}s ({ema_change_pct:+.1f}%) - WRONG direction, should decrease"
            else:
                outcome = "NO_CHANGE"
                outcome_details = f"EMA unchanged {before_ema:.3f}s → {current_ema:.3f}s ({ema_change_pct:+.1f}%)"
        else:
            outcome = "UNKNOWN"
            outcome_details = f"EMA changed {before_ema:.3f}s → {current_ema:.3f}s"
        
        # Update the history entry
        last_entry["outcome"] = outcome
        last_entry["outcome_details"] = outcome_details
        last_entry["after_rt"] = round(current_rt, 3)
        last_entry["after_ema"] = round(current_ema, 3)
        
        logger.info(f"Updated outcome for violation #{last_entry['violation_number']}: {outcome}")
        logger.info(f"  {outcome_details}")
        
        self.pending_outcome = None
    
    def finalize_pending_outcome(self, current_rt: float, current_ema: float, is_violation: bool):
        """
        Finalize the outcome of the last decision when experiment ends or no violation.
        
        Call this when checking metrics but no violation occurred.
        
        Args:
            current_rt: Current response time
            current_ema: Current EMA response time
            is_violation: Whether current state is still a violation
        """
        violation_type = None if not is_violation else self.pending_outcome.get("violation_type") if self.pending_outcome else None
        self._update_pending_outcome(current_rt, current_ema, violation_type)
    
    def update_pending_outcome_before_prompt(self, current_rt: float, current_ema: float, violation_type: str):
        """
        Update the pending outcome BEFORE building a new prompt.
        
        This ensures that when we format history for the prompt, the outcome
        of the previous decision has already been evaluated.
        
        Args:
            current_rt: Current response time
            current_ema: Current EMA response time  
            violation_type: Current violation type
        """
        self._update_pending_outcome(current_rt, current_ema, violation_type)
    
    def get_history(self) -> list:
        """
        Get all history entries.
        
        Returns:
            List of history entries (oldest first)
        """
        return list(self.history)
    
    def get_history_count(self) -> int:
        """
        Get the number of entries in history.
        
        Returns:
            Number of stored entries
        """
        return len(self.history)
    
    def has_history(self) -> bool:
        """
        Check if there is any history.
        
        Returns:
            True if there is at least one entry
        """
        return len(self.history) > 0
    
    def format_for_prompt(self) -> str:
        """
        Format history for inclusion in LLM prompt.
        
        Now includes OUTCOME of each decision, showing whether the action
        improved or worsened the situation.
        
        Returns:
            Formatted string representation of history,
            or "(none)" if no history exists
        """
        if not self.history:
            return "(none)"
        
        lines = []
        
        for entry in self.history:
            lines.append(f"--- Violation #{entry['violation_number']} ---")
            lines.append(f"Type: {entry['violation_type']}")
            lines.append(f"Response Time: {entry['response_time']}s (EMA: {entry['ema_response_time']}s)")
            lines.append(f"Monitoring: {entry['monitoring_summary']}")
            lines.append(f"Decision: action={entry['decision'].get('action')}, params={entry['decision'].get('parameters')}")
            
            # Add outcome if available
            if entry.get("outcome"):
                outcome = entry["outcome"]
                outcome_details = entry.get("outcome_details", "")
                
                # Format outcome with emphasis
                if outcome == "IMPROVED":
                    lines.append(f"OUTCOME: ✓ {outcome} - {outcome_details}")
                elif outcome == "WORSENED":
                    lines.append(f"OUTCOME: ✗ {outcome} - {outcome_details}")
                elif outcome == "RESOLVED":
                    lines.append(f"OUTCOME: ✓✓ {outcome} - {outcome_details}")
                else:
                    lines.append(f"OUTCOME: {outcome} - {outcome_details}")
            else:
                lines.append("OUTCOME: (pending - awaiting next measurement)")
            
            lines.append("")
        
        return "\n".join(lines).strip()
    
    def get_last_decision(self) -> Optional[dict]:
        """
        Get the most recent decision.
        
        Returns:
            Last decision dict or None if no history
        """
        if not self.history:
            return None
        return self.history[-1]["decision"]
    
    def get_session_duration_seconds(self) -> Optional[float]:
        """
        Get the duration of the current session in seconds.
        
        Returns:
            Duration in seconds, or None if session not started
        """
        if self.session_start is None:
            return None
        return (datetime.now() - self.session_start).total_seconds()
    
    def __len__(self) -> int:
        """Return number of entries in history."""
        return len(self.history)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"DecisionHistory(entries={len(self.history)}, max={self.max_entries})"