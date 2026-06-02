"""
Intent Watch Loop Module

This module implements the core monitoring loop that:
1. Monitors application response times
2. Detects SLO violations using Exponential Moving Average (EMA)
3. Triggers the Decision Maker when violations occur
4. Executes recommended actions via Action Executor
5. Maintains decision history for LLM context

Updated to support:
- Decision history (last N violations)
- Time window experiments (run for X minutes)
- Cleaner integration with new prompt format
"""

import logging
import time
import subprocess
import uuid
from typing import Optional, Callable
from datetime import datetime, timedelta

from data_collector import DataCollector
from decision_maker import DecisionMaker
from decision_history import DecisionHistory
from action_executor import ActionExecutor
from candidate_generator import CandidateActionGenerator
from candidate_filter import CandidateFilter

logger = logging.getLogger(__name__)


class IntentWatchLoop:
    """
    Core monitoring loop for intent-based resource management.
    
    This class continuously monitors application response times and
    triggers corrective actions when SLO violations are detected.
    """
    
    def __init__(self, config: dict):
        """
        Initialize Intent Watch Loop with configuration.
        
        Args:
            config: Configuration dictionary containing thresholds and settings
        """
        self.config = config
        self.monitor_only = config.get("monitor_only", False)

        # Intent thresholds
        intent_config = config["intent"]
        self.upper_threshold = intent_config["upper_threshold"]
        self.lower_threshold = intent_config["lower_threshold"]
        self.ema_alpha = intent_config["ema_alpha"]
        self.check_interval = intent_config["check_interval"]
        self.wait_after_action = intent_config["wait_after_action"]

        # Application endpoint
        self.app_endpoint = config["application"]["entry_point"]
        self.test_image_path = config["application"].get("test_image")

        # Data collector always initialized (needed for K8s client references)
        self.data_collector = DataCollector(config)

        # LLM and action components — skipped in monitor-only mode
        if not self.monitor_only:
            self.decision_maker = DecisionMaker(config)
            self.candidate_generator = CandidateActionGenerator(config)
            self.action_executor = ActionExecutor(
                config,
                self.data_collector.k8s_client,
                self.data_collector.onos_client
            )
            self.candidate_filter = CandidateFilter(config)
        else:
            self.decision_maker = None
            self.candidate_generator = None
            self.action_executor = None
            self.candidate_filter = None

        # Decision history
        max_history = config.get("history", {}).get("max_entries", 5)
        self.decision_history = DecisionHistory(max_entries=max_history)

        # EMA state
        self.ema_rt: Optional[float] = None
        self.last_rt: Optional[float] = None

        # Loop control
        self.running = False
        self.paused = False

        # Tracks whether the previous violation is pending resolution
        self._pending_violation = False

        # Per-measurement EMA time-series (used for Figure 7 plots)
        self.ema_timeline: list = []

        # Statistics
        self.stats = {
            "total_requests": 0,
            "violations_detected": 0,
            "no_candidate_violations": 0,
            "actions_taken": 0,
            "actions_succeeded": 0,
            "violations_resolved": 0,
            "start_time": None,
            "end_time": None
        }
    
    def _measure_response_time(self) -> Optional[float]:
        """
        Measure response time by sending a request to the application via SSH.
        
        Since the SDN-Controller cannot directly reach the microservices via
        the SDN network, we execute curl on the Master node via SSH.
        
        Returns:
            Response time in seconds, or None if request failed
        """
        try:
            app_config = self.config["application"]
            k8s_config = self.config["endpoints"]
            
            # Build the curl command to run locally on sdn-controller
            request_id = str(uuid.uuid4())

            curl_command = [
                "curl", "-X", "POST",
                "-F", f"image=@{app_config.get('test_image', 'images/family.jpg')}",
                "-H", f"X-Request-ID: {request_id}",
                "-H", f"X-Webhooks: {app_config.get('webhooks', '')}",
                "-H", "X-Special-Object: person",
                "-H", f"X-Central-DB-URL: {app_config.get('db_url', '')}",
                "-H", f"X-Logs-URL: {app_config.get('logs_url', '')}",
                app_config.get("entry_point", "http://10.56.1.209:5001/resize"),
                "--max-time", "90",
                "-w", "%{time_total}",
                "-o", "/dev/null",
                "-s"
            ]

            result = subprocess.run(
                curl_command,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                try:
                    response_time = float(result.stdout.strip())
                    logger.debug(f"Request successful, response time: {response_time:.3f}s")
                    return response_time
                except ValueError:
                    logger.error(f"Could not parse response time: {result.stdout}")
                    return None
            else:
                logger.warning(f"Curl failed with return code {result.returncode}")
                logger.warning(f"Stderr: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("SSH request timed out after 90 seconds")
            return 90.0
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None
    
    def _update_ema(self, response_time: float):
        """
        Update the Exponential Moving Average of response time.
        
        EMA formula: EMA_t = (1 - α) × EMA_{t-1} + α × RT_t
        
        Args:
            response_time: Latest response time measurement
        """
        if self.ema_rt is None:
            self.ema_rt = response_time
        else:
            self.ema_rt = (1 - self.ema_alpha) * self.ema_rt + self.ema_alpha * response_time
    
    def _check_violation(self) -> Optional[str]:
        """
        Check if current EMA violates thresholds.
        
        Returns:
            "UPPER_THRESHOLD_EXCEEDED" if above upper threshold,
            "LOWER_THRESHOLD_EXCEEDED" if below lower threshold,
            None if within acceptable range
        """
        if self.ema_rt is None:
            return None
        
        if self.ema_rt > self.upper_threshold:
            return "UPPER_THRESHOLD_EXCEEDED"
        elif self.ema_rt < self.lower_threshold:
            return "LOWER_THRESHOLD_EXCEEDED"
        
        return None
    
    def _handle_violation(self, violation_type: str, current_rt: float):
        """
        Handle a detected violation using the 8-message LLM conversation format.
        
        Flow: collect data → build structured state → generate candidates →
              query LLM (8-msg) → execute action → record history
        """
        logger.warning("=" * 60)
        logger.warning(f"VIOLATION DETECTED: {violation_type}")
        logger.warning(f"Current RT: {current_rt:.3f}s | EMA: {self.ema_rt:.3f}s")
        logger.warning(f"Thresholds: [{self.lower_threshold}s, {self.upper_threshold}s]")
        logger.warning("=" * 60)

        self.stats["violations_detected"] += 1
        self._pending_violation = True

        # Monitor-only mode: record violation but skip all LLM and action logic
        if self.monitor_only:
            logger.info("[MONITOR-ONLY] Violation logged — no LLM query or action taken")
            return

        logger.warning(f"History: {self.decision_history.get_history_count()} previous decisions")
        
        # Step 1: Collect raw system data
        logger.info("Collecting system data...")
        system_state = self.data_collector.collect_all()
        
        # Get compact strings for logging and legacy history
        monitoring_str = self.data_collector.format_monitoring_compact(system_state)
        deployments_str = self.data_collector.format_deployments_compact(system_state)
        logger.info(f"Deployments: {deployments_str}")
        
        # Step 2: Build structured state for 8-message format
        threshold = (self.upper_threshold if violation_type == "UPPER_THRESHOLD_EXCEEDED"
                     else self.lower_threshold)
        structured_state = self.data_collector.build_structured_state(
            system_state=system_state,
            violation_type=violation_type,
            ema_rt=self.ema_rt,
            threshold=threshold
        )
        
        # Step 3: Generate candidate actions
        candidate_actions = self.candidate_generator.generate(
            structured_state=structured_state,
            violation_type=violation_type
        )
        logger.info(f"Generated {len(candidate_actions)} candidate actions:")
        for c in candidate_actions:
            logger.info(f"  [{c['id']}] {c['type']} → {c.get('target', c.get('description', 'n/a'))}")
        
        # Step 4: Update outcome of previous decision and get history
        self.decision_history.update_pending_outcome_before_prompt(
            current_rt=current_rt,
            current_ema=self.ema_rt,
            violation_type=violation_type
        )
        # Also update structured history outcome
        self.decision_history.update_last_structured_outcome("violation_persisted")

        history_entries = self.decision_history.get_structured_history()

        # Step 4b: Hybrid filter — enforce Rule 3 in Python before the LLM
        # sees the candidates.  Removes (action_type, target) pairs that
        # already appeared in history with result=violation_persisted.
        # Re-letters the remaining candidates A/B/C/... so the LLM sees a
        # compact list without gaps.
        filtered_candidates = self.candidate_filter.filter(
            candidate_actions, history_entries
        )

        # No candidates at all (e.g. single-action ablation with trigger condition unmet)
        if not filtered_candidates:
            self.stats["no_candidate_violations"] += 1
            logger.info("No valid candidates after filter — recording as 0 actions taken for this violation")
            return

        if len(filtered_candidates) != len(candidate_actions):
            logger.info("Filtered candidate list (shown to LLM):")
            for c in filtered_candidates:
                logger.info(
                    f"  [{c['id']}] {c['type']} → "
                    f"{c.get('target', c.get('description', 'n/a'))}"
                )

        # Step 5: Query LLM with 8-message format (filtered candidates)
        logger.info("Querying LLM for recommendation (8-msg format)...")
        recommendation = self.decision_maker.analyze_and_recommend(
            violation_type=violation_type,
            current_rt=current_rt,
            ema_rt=self.ema_rt,
            structured_state=structured_state,
            candidate_actions=filtered_candidates,
            history_entries=history_entries
        )

        # Step 5b: Post-selection validation — catch position bias where the
        # LLM's reasoning identified a valid service but the selected letter
        # resolved to a candidate outside the filtered set.
        recommendation, was_overridden = self.candidate_filter.validate_selection(
            recommendation, filtered_candidates
        )
        if was_overridden:
            logger.warning("Hybrid filter: position-bias correction applied")

        logger.info(f"Recommended Action: {recommendation.get('action', 'none')}")
        if recommendation.get('parameters'):
            logger.info(f"Parameters: {recommendation.get('parameters')}")
        
        # Step 6: Execute the action
        action_executed = False
        if recommendation.get("action") != "none":
            logger.info("Executing recommended action...")
            result = self.action_executor.execute(
                action=recommendation.get("action"),
                parameters=recommendation.get("parameters", {}),
                analysis=""
            )
            
            if result["success"]:
                logger.info(f"Action successful: {result['message']}")
                self.stats["actions_taken"] += 1
                self.stats["actions_succeeded"] += 1
                action_executed = True

                # Clear CLOSE_WAIT state on ms3 after any action that restarted another
                # service pod (ms3 hangs 90s when a downstream pod is briefly unavailable,
                # accumulating CLOSE_WAIT sockets that block new requests).
                # Exception: when the action scaled ms3 itself, its pods are already fresh
                # — force-deleting them immediately would kill the just-scheduled replicas
                # and double the time waiting for the rollout to complete.
                action_target = recommendation.get("parameters", {}).get("deployment_name", "")
                ms3_was_scaled = "microservice3" in action_target

                if ms3_was_scaled:
                    # Just wait for the rollout K8s already started (no delete needed)
                    logger.info("Waiting for ms3 rollout to complete...")
                    self.data_collector.k8s_client._run_kubectl(
                        "rollout status deployment/microservice3-deployment --timeout=180s"
                    )
                else:
                    # Another service's pod restarted → ms3 may have accumulated CLOSE_WAIT.
                    # Only force-delete if ALL ms3 replicas are currently ready.
                    # If ms3 is in a partial rollout (e.g. 1/2 ready after a previous
                    # add_replica), killing the running pod leaves ms3 at 0/N ready and
                    # causes a full service outage until new pods start (V8 failure pattern).
                    ready_str = self.data_collector.k8s_client._run_kubectl(
                        'get deployment microservice3-deployment -o jsonpath="{.status.readyReplicas}"'
                    ).strip().strip('"') or "0"
                    desired_str = self.data_collector.k8s_client._run_kubectl(
                        'get deployment microservice3-deployment -o jsonpath="{.spec.replicas}"'
                    ).strip().strip('"') or "1"
                    try:
                        ms3_ready = int(ready_str)
                        ms3_desired = int(desired_str)
                    except (ValueError, TypeError):
                        ms3_ready, ms3_desired = 0, 1

                    if ms3_ready >= ms3_desired:
                        logger.info("Restarting ms3 pod to clear connection state...")
                        self.data_collector.k8s_client._run_kubectl(
                            "delete pod -l app=microservice3 --force --grace-period=0"
                        )
                        self.data_collector.k8s_client._run_kubectl(
                            "rollout status deployment/microservice3-deployment --timeout=120s"
                        )
                    else:
                        logger.info(
                            f"ms3 in partial rollout ({ms3_ready}/{ms3_desired} ready) — "
                            "skipping force-delete to avoid outage; waiting for rollout..."
                        )
                        self.data_collector.k8s_client._run_kubectl(
                            "rollout status deployment/microservice3-deployment --timeout=180s"
                        )
                logger.info("ms3 pod ready.")

                # After any ms3 pod replacement (rollout or force-delete), the new pod
                # starts with a cold SSD model. The first real request will be slow
                # (~5-7s), pushing EMA over the upper threshold immediately and causing
                # a spurious violation on the next check. Send one warm-up request now
                # so the SSD model is loaded before the stabilization wait ends.
                logger.info("Sending ms3 warm-up request to pre-load SSD model...")
                warmup_rt = self._measure_response_time()
                if warmup_rt is not None:
                    logger.info(f"ms3 warm-up complete (RT={warmup_rt:.3f}s, not counted in metrics)")
                else:
                    logger.warning("ms3 warm-up request failed — SSD model may still be cold")

                logger.info(f"Waiting {self.wait_after_action}s for system to stabilize...")
                time.sleep(self.wait_after_action)
            else:
                logger.error(f"Action failed: {result['message']}")
        else:
            logger.info("No action recommended by LLM")
        
        # Step 7: Save to both legacy and structured history
        self.decision_history.add_entry(
            violation_type=violation_type,
            response_time=current_rt,
            ema_response_time=self.ema_rt if self.ema_rt else current_rt,
            monitoring_summary=monitoring_str,
            deployments_summary=deployments_str,
            decision={
                "action": recommendation.get("action", "none"),
                "parameters": recommendation.get("parameters", {})
            }
        )
        
        # Build and save structured history entry for 8-msg format.
        # Match against filtered_candidates (what the LLM actually saw).
        selected_candidate = filtered_candidates[0] if filtered_candidates else {}
        for c in filtered_candidates:
            executor_action = self.decision_maker._candidate_to_executor_action(c)
            if (executor_action.get("action") == recommendation.get("action") and
                executor_action.get("parameters", {}).get("deployment_name") ==
                recommendation.get("parameters", {}).get("deployment_name")):
                selected_candidate = c
                break
        
        structured_entry = self.decision_history.build_structured_entry(
            structured_state=structured_state,
            selected_candidate=selected_candidate,
            candidate_actions=candidate_actions
        )
        self.decision_history.add_structured_entry(structured_entry)
    
    def run_once(self) -> dict:
        """
        Run a single iteration of the watch loop.
        
        Useful for testing or manual control.
        
        Returns:
            Dictionary with iteration results
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "response_time": None,
            "ema_rt": None,
            "violation": None,
            "action_taken": False
        }
        
        # Measure response time
        rt = self._measure_response_time()
        
        if rt is None:
            logger.error("Failed to measure response time")
            return result
        
        result["response_time"] = rt
        self.last_rt = rt
        self.stats["total_requests"] += 1

        # Update EMA
        self._update_ema(rt)
        result["ema_rt"] = self.ema_rt

        # Record EMA timeline point (used for Figure 7 EMA response-time plots)
        timeline_entry = {
            "timestamp": datetime.now().isoformat(),
            "rt": round(rt, 3),
            "ema": round(self.ema_rt, 3),
            "violation": None
        }

        # Add to data collector history
        self.data_collector.add_response_time(rt)

        # Log current state
        logger.info(f"RT: {rt:.3f}s | EMA: {self.ema_rt:.3f}s | Thresholds: [{self.lower_threshold}, {self.upper_threshold}]")

        # Check for violations
        violation = self._check_violation()

        if violation:
            timeline_entry["violation"] = violation
            result["violation"] = violation
            self._handle_violation(violation, rt)
            result["action_taken"] = True
        else:
            # No violation — if a previous violation was pending resolution, count it
            had_pending = self._pending_violation
            self.decision_history.finalize_pending_outcome(rt, self.ema_rt, is_violation=False)
            self.decision_history.update_last_structured_outcome("violation_resolved")
            if had_pending:
                self.stats["violations_resolved"] += 1
                self._pending_violation = False

        self.ema_timeline.append(timeline_entry)
        
        return result
    
    def run_time_window(
        self,
        duration_minutes: int,
        callback: Optional[Callable] = None
    ) -> dict:
        """
        Run the monitoring loop for a fixed time window.
        
        This is the main method for running experiments.
        
        Args:
            duration_minutes: How long to run the experiment (in minutes)
            callback: Optional callback function called after each iteration
            
        Returns:
            Dictionary with experiment results and statistics
        """
        logger.info("=" * 60)
        logger.info(f"STARTING TIME WINDOW EXPERIMENT")
        logger.info(f"Duration: {duration_minutes} minutes")
        logger.info(f"Upper threshold: {self.upper_threshold}s")
        logger.info(f"Lower threshold: {self.lower_threshold}s")
        logger.info(f"Check interval: {self.check_interval}s")
        logger.info(f"EMA alpha: {self.ema_alpha}")
        logger.info(f"Max history entries: {self.decision_history.max_entries}")
        logger.info("=" * 60)
        
        # Warm-up request to avoid cold start affecting EMA
        logger.info("Sending warm-up request (not counted in metrics)...")
        warmup_rt = self._measure_response_time()
        if warmup_rt is not None:
            logger.info(f"Warm-up complete. Response time: {warmup_rt:.3f}s (ignored)")
        else:
            logger.warning("Warm-up request failed, continuing anyway...")
        
        # Reset state for new experiment
        self.decision_history.reset()
        self.ema_rt = None
        self.last_rt = None
        self._pending_violation = False
        self.ema_timeline = []
        self.stats = {
            "total_requests": 0,
            "violations_detected": 0,
            "no_candidate_violations": 0,
            "actions_taken": 0,
            "actions_succeeded": 0,
            "violations_resolved": 0,
            "start_time": datetime.now().isoformat(),
            "end_time": None
        }
        
        # Calculate end time
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        
        self.running = True
        iteration = 0
        
        try:
            while self.running and datetime.now() < end_time:
                if not self.paused:
                    result = self.run_once()
                    
                    if callback:
                        callback(result)
                    
                    iteration += 1
                    
                    # Log progress every 10 iterations
                    if iteration % 10 == 0:
                        remaining = (end_time - datetime.now()).total_seconds() / 60
                        logger.info(f"Progress: {iteration} iterations, {remaining:.1f} minutes remaining")
                
                # Wait before next check
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping experiment...")
        finally:
            self.running = False
            self.stats["end_time"] = datetime.now().isoformat()
        
        # Print final statistics
        self._print_stats()
        
        # Return experiment results
        return {
            "stats": self.stats.copy(),
            "history": self.decision_history.get_history(),
            "all_structured_history": self.decision_history.get_all_structured_history(),
            "llm_calls_log": self.decision_maker.get_llm_calls_log() if self.decision_maker else [],
            "ema_timeline": self.ema_timeline,
            "duration_minutes": duration_minutes,
            "iterations": iteration
        }
    
    def run(self, max_iterations: Optional[int] = None, callback: Optional[Callable] = None):
        """
        Run the continuous monitoring loop (legacy method).
        
        Args:
            max_iterations: Optional limit on number of iterations (None = run forever)
            callback: Optional callback function called after each iteration
        """
        logger.info("=" * 60)
        logger.info("Starting Intent Watch Loop (continuous mode)")
        logger.info(f"Upper threshold: {self.upper_threshold}s")
        logger.info(f"Lower threshold: {self.lower_threshold}s")
        logger.info(f"Check interval: {self.check_interval}s")
        logger.info(f"EMA alpha: {self.ema_alpha}")
        logger.info("=" * 60)
        
        # Reset history for new run
        self.decision_history.reset()
        
        self.running = True
        self.stats["start_time"] = datetime.now().isoformat()
        iteration = 0
        
        try:
            while self.running:
                if max_iterations and iteration >= max_iterations:
                    logger.info(f"Reached max iterations ({max_iterations})")
                    break
                
                if not self.paused:
                    result = self.run_once()
                    
                    if callback:
                        callback(result)
                    
                    iteration += 1
                
                # Wait before next check
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping...")
        finally:
            self.running = False
            self.stats["end_time"] = datetime.now().isoformat()
            logger.info("Intent Watch Loop stopped")
            self._print_stats()
    
    def stop(self):
        """Stop the monitoring loop."""
        self.running = False
    
    def pause(self):
        """Pause the monitoring loop."""
        self.paused = True
        logger.info("Watch loop paused")
    
    def resume(self):
        """Resume the monitoring loop."""
        self.paused = False
        logger.info("Watch loop resumed")
    
    def _print_stats(self):
        """Print statistics summary."""
        logger.info("=" * 60)
        logger.info("EXPERIMENT STATISTICS")
        logger.info(f"  Total requests: {self.stats['total_requests']}")
        logger.info(f"  Violations detected: {self.stats['violations_detected']}")
        logger.info(f"  No-candidate violations: {self.stats.get('no_candidate_violations', 0)}")
        logger.info(f"  Actions taken: {self.stats['actions_taken']}")
        logger.info(f"  Actions succeeded: {self.stats.get('actions_succeeded', 0)}")
        logger.info(f"  Violations resolved: {self.stats.get('violations_resolved', 0)}")
        logger.info(f"  Start time: {self.stats['start_time']}")
        logger.info(f"  End time: {self.stats['end_time']}")
        
        # Print decision history summary
        if self.decision_history.has_history():
            logger.info("  Decision History:")
            for entry in self.decision_history.get_history():
                logger.info(f"    - Violation #{entry['violation_number']}: {entry['decision']['action']}")
        
        logger.info("=" * 60)
    
    def get_status(self) -> dict:
        """
        Get current status of the watch loop.
        
        Returns:
            Dictionary with current status information
        """
        return {
            "running": self.running,
            "paused": self.paused,
            "ema_rt": self.ema_rt,
            "last_rt": self.last_rt,
            "upper_threshold": self.upper_threshold,
            "lower_threshold": self.lower_threshold,
            "stats": self.stats.copy(),
            "history_count": self.decision_history.get_history_count()
        }
    
    def get_decision_history(self) -> list:
        """
        Get the current decision history.
        
        Returns:
            List of decision history entries
        """
        return self.decision_history.get_history()