"""
Decision History Module

This module manages the history of past violations and LLM decisions.
It provides context to TinyLlama so it can make smarter decisions
based on what actions were taken previously in the same session.

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
    
    Stores the last N violations with their monitoring data and
    the decisions that were made, so TinyLlama can learn from
    past actions in the current session.
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
        
        logger.info(f"DecisionHistory initialized with max_entries={max_entries}")
    
    def reset(self):
        """
        Reset history for a new experiment session.
        
        Called at the start of each time window experiment.
        """
        self.history.clear()
        self.session_start = datetime.now()
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
        entry = {
            "timestamp": datetime.now().isoformat(),
            "violation_number": len(self.history) + 1,
            "violation_type": violation_type,
            "response_time": round(response_time, 3),
            "ema_response_time": round(ema_response_time, 3),
            "monitoring_summary": monitoring_summary,
            "deployments_summary": deployments_summary,
            "decision": decision
        }
        
        self.history.append(entry)
        logger.info(f"Added decision to history (total: {len(self.history)})")
        logger.debug(f"Entry: {entry}")
    
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
