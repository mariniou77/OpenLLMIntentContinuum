"""
Experiment Telemetry Module

Aggregates and exports all per-experiment metrics into the evaluation_results/
directory structure.  Called at the end of each ablation run by run_ablation.py.

Outputs per experiment:
  llm_interactions.csv   — one row per LLM call (latency, tokens, action selected)
  actions_log.csv        — one row per executed action (type, target, success, outcome)
  summary.json           — aggregated stats + intent satisfaction rate
"""

import csv
import json
import os
import statistics
from datetime import datetime
from typing import Optional


class ExperimentTelemetry:
    """Collects and exports telemetry for a single ablation experiment run."""

    def __init__(
        self,
        experiment_name: str,
        experiment_label: str,
        enabled_actions: list,
        output_dir: str,
    ):
        self.experiment_name = experiment_name
        self.experiment_label = experiment_label
        self.enabled_actions = enabled_actions
        self.output_dir = output_dir

        # References to live objects — set via attach_*()
        self._watch_loop = None
        self._decision_maker = None
        self._locust_csv_prefix = None  # path prefix used with --csv flag

        # EMA timeline + SLO thresholds for time-in-band metric (set via set_ema_data)
        self._ema_timeline: list = []
        self._lower_threshold: float = 0.5
        self._upper_threshold: float = 3.0

    # ── Attach live objects ──────────────────────────────────────────────────

    def attach_watch_loop(self, watch_loop) -> None:
        self._watch_loop = watch_loop

    def attach_decision_maker(self, decision_maker) -> None:
        self._decision_maker = decision_maker

    def set_locust_csv_prefix(self, prefix: str) -> None:
        self._locust_csv_prefix = prefix

    def set_ema_data(self, ema_timeline: list, lower_threshold: float, upper_threshold: float) -> None:
        """Supply EMA timeline and SLO thresholds for the time-in-band metric."""
        self._ema_timeline = ema_timeline or []
        self._lower_threshold = lower_threshold
        self._upper_threshold = upper_threshold

    # ── Export ───────────────────────────────────────────────────────────────

    def export_all(self) -> str:
        """Write all output files and return the path to summary.json."""
        os.makedirs(self.output_dir, exist_ok=True)

        llm_calls = self._decision_maker.get_llm_calls_log() if self._decision_maker else []
        all_history = (
            self._watch_loop.decision_history.get_all_structured_history()
            if self._watch_loop else []
        )
        stats = self._watch_loop.stats.copy() if self._watch_loop else {}

        self._export_llm_interactions_csv(llm_calls)
        self._export_actions_log_csv(all_history)
        summary_path = self._export_summary_json(stats, llm_calls, all_history)
        return summary_path

    # ── CSV writers ──────────────────────────────────────────────────────────

    def _export_llm_interactions_csv(self, llm_calls: list) -> None:
        path = os.path.join(self.output_dir, "llm_interactions.csv")
        fieldnames = [
            "timestamp", "prompt_tokens", "completion_tokens",
            "latency_ms", "cost_usd", "action_selected"
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for call in llm_calls:
                writer.writerow({k: call.get(k, "") for k in fieldnames})

    def _export_actions_log_csv(self, all_history: list) -> None:
        """
        Write one row per executed action derived from the full structured history.

        Each structured history entry has the shape:
          { "violation": {...}, "metrics": {...}, "action_taken": {"type", "target", "result"} }
        """
        path = os.path.join(self.output_dir, "actions_log.csv")
        fieldnames = [
            "violation_status", "observed_rt", "threshold",
            "action_type", "target", "outcome"
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in all_history:
                v = entry.get("violation", {})
                at = entry.get("action_taken", {})
                writer.writerow({
                    "violation_status": v.get("status", ""),
                    "observed_rt": v.get("observed", ""),
                    "threshold": v.get("threshold", ""),
                    "action_type": at.get("type", ""),
                    "target": at.get("target", ""),
                    "outcome": at.get("result", ""),
                })

    # ── Summary JSON ─────────────────────────────────────────────────────────

    def _export_summary_json(
        self,
        stats: dict,
        llm_calls: list,
        all_history: list,
    ) -> str:
        violations_detected = stats.get("violations_detected", 0)
        actions_taken = stats.get("actions_taken", 0)
        actions_succeeded = stats.get("actions_succeeded", 0)
        violations_resolved = stats.get("violations_resolved", 0)
        no_candidate_violations = stats.get("no_candidate_violations", 0)

        isr = (
            round(violations_resolved / violations_detected, 4)
            if violations_detected > 0 else None
        )

        # Time-normalised ISR: resolutions per minute (corrects for cooldown suppression)
        duration_s = self._compute_duration(stats) or 0
        duration_min = duration_s / 60 if duration_s > 0 else 0
        time_normalised_isr = (
            round(violations_resolved / duration_min, 4)
            if violations_resolved > 0 and duration_min > 0 else None
        )

        # EMA time-in-band: % of monitoring cycles with SLO-compliant EMA
        ema_time_in_band_pct = self._compute_ema_time_in_band()

        # Locust P95 response time from stats CSV
        locust_p95_rt_ms = self._read_locust_p95()

        # LLM latency stats
        latencies = [c["latency_ms"] for c in llm_calls if c.get("latency_ms")]
        prompt_tokens_list = [c["prompt_tokens"] for c in llm_calls if c.get("prompt_tokens")]
        completion_tokens_list = [c["completion_tokens"] for c in llm_calls if c.get("completion_tokens")]

        mean_latency = round(statistics.mean(latencies), 1) if latencies else None
        p95_latency = round(
            sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 2 else (latencies[0] if latencies else None), 1
        ) if latencies else None

        # Action type breakdown from structured history
        action_breakdown: dict = {}
        for entry in all_history:
            atype = entry.get("action_taken", {}).get("type", "unknown")
            action_breakdown[atype] = action_breakdown.get(atype, 0) + 1

        # Total API cost (non-null only for OpenAI experiments)
        costs = [c["cost_usd"] for c in llm_calls if c.get("cost_usd") is not None]
        total_api_cost_usd = round(sum(costs), 4) if costs else None

        # Locust aggregate from CSV if available (fail silently)
        locust_requests, locust_failures = self._read_locust_totals()

        summary = {
            "experiment_name": self.experiment_name,
            "experiment_label": self.experiment_label,
            "enabled_actions": self.enabled_actions,
            "exported_at": datetime.utcnow().isoformat(),
            "duration_seconds": self._compute_duration(stats),
            "total_locust_requests": locust_requests,
            "total_locust_failures": locust_failures,
            "violations_detected": violations_detected,
            "no_candidate_violations": no_candidate_violations,
            "actions_executed": actions_taken,
            "actions_succeeded": actions_succeeded,
            "violations_resolved": violations_resolved,
            "intent_satisfaction_rate": isr,
            "time_normalised_isr": time_normalised_isr,
            "ema_time_in_band_pct": ema_time_in_band_pct,
            "locust_p95_rt_ms": locust_p95_rt_ms,
            "action_type_breakdown": action_breakdown,
            "mean_inference_latency_ms": mean_latency,
            "p95_inference_latency_ms": p95_latency,
            "mean_prompt_tokens": round(statistics.mean(prompt_tokens_list), 0) if prompt_tokens_list else None,
            "mean_completion_tokens": round(statistics.mean(completion_tokens_list), 0) if completion_tokens_list else None,
            "total_prompt_tokens": sum(prompt_tokens_list),
            "total_completion_tokens": sum(completion_tokens_list),
            "total_api_cost_usd": total_api_cost_usd,
        }

        path = os.path.join(self.output_dir, "summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  📄 Summary saved to {path}")
        return path

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _compute_duration(self, stats: dict) -> Optional[float]:
        start = stats.get("start_time")
        end = stats.get("end_time")
        if not start or not end:
            return None
        try:
            fmt = "%Y-%m-%dT%H:%M:%S.%f"
            t0 = datetime.fromisoformat(start)
            t1 = datetime.fromisoformat(end)
            return round((t1 - t0).total_seconds(), 1)
        except Exception:
            return None

    def _read_locust_totals(self):
        """Read total requests and failures from Locust stats CSV."""
        if not self._locust_csv_prefix:
            return None, None
        stats_file = f"{self._locust_csv_prefix}_stats.csv"
        if not os.path.exists(stats_file):
            return None, None
        try:
            total_req, total_fail = 0, 0
            with open(stats_file, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("Name") == "Aggregated":
                        total_req = int(row.get("Request Count", 0))
                        total_fail = int(row.get("Failure Count", 0))
                        break
            return total_req, total_fail
        except Exception:
            return None, None

    def _read_locust_p95(self) -> Optional[int]:
        """Read the Locust P95 response time (ms) for the /resize endpoint."""
        if not self._locust_csv_prefix:
            return None
        stats_file = f"{self._locust_csv_prefix}_stats.csv"
        if not os.path.exists(stats_file):
            return None
        try:
            with open(stats_file, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("Name") in ("/resize", "Aggregated"):
                        val = row.get("95%", "") or row.get("95th Percentile Response Time", "")
                        if val and val.strip():
                            return int(float(val))
            return None
        except Exception:
            return None

    def _compute_ema_time_in_band(self) -> Optional[float]:
        """Fraction (%) of EMA timeline points within the SLO band [lower, upper]."""
        timeline = self._ema_timeline
        if not timeline:
            return None
        lo, hi = self._lower_threshold, self._upper_threshold
        in_band = sum(
            1 for e in timeline
            if isinstance(e, dict) and lo <= e.get("ema", -1) <= hi
        )
        return round(in_band / len(timeline) * 100, 1)
