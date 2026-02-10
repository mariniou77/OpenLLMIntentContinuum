"""
Intent Watch Loop Module

This module implements the core monitoring loop that:
1. Monitors application response times
2. Detects SLO violations using Exponential Moving Average (EMA)
3. Triggers the Decision Maker when violations occur
4. Executes recommended actions via Action Executor

This is the main control loop of the IntentContinuum system.
"""

import logging
import time
import requests
from typing import Optional, Callable
from datetime import datetime

from data_collector import DataCollector
from decision_maker import DecisionMaker
from action_executor import ActionExecutor

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
        
        # Initialize components
        self.data_collector = DataCollector(config)
        self.decision_maker = DecisionMaker(config)
        self.action_executor = ActionExecutor(
            config, 
            self.data_collector.k8s_client,
            self.data_collector.onos_client
        )
        
        # EMA state
        self.ema_rt: Optional[float] = None
        
        # Loop control
        self.running = False
        self.paused = False
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "violations_detected": 0,
            "actions_taken": 0,
            "start_time": None
        }
    
    def _measure_response_time(self) -> Optional[float]:
        """
        Measure response time by sending a request to the application via SSH.
        
        Since the SDN-Controller cannot directly reach the microservices via
        the SDN network, we execute curl on the Master node via SSH.
        
        Returns:
            Response time in seconds, or None if request failed
        """
        import uuid
        import subprocess
        
        try:
            app_config = self.config["application"]
            k8s_config = self.config["endpoints"]
            
            # Build the curl command to run on the Master node
            request_id = str(uuid.uuid4())
            
            # Use the SDN IP (192.168.100.100) since we're running from Master
            # The test image must exist on the Master node
            curl_command = (
                f'curl -X POST '
                f'-F "image=@{app_config.get("remote_test_image", "/home/antonios-icontinuum/test_converted.jpg")}" '
                f'-H "X-Request-ID: {request_id}" '
                f'-H "X-Webhooks: {app_config.get("webhooks", "")}" '
                f'-H "X-Special-Object: person" '
                f'-H "X-Central-DB-URL: {app_config.get("db_url", "")}" '
                f'-H "X-Logs-URL: {app_config.get("logs_url", "")}" '
                f'{app_config.get("sdn_entry_point", "http://192.168.100.100:5001/resize")} '
                f'--max-time 60 '
                f'-w "%{{time_total}}" '
                f'-o /dev/null -s'
            )
            
            # Execute curl via SSH on the Master node
            master_host = k8s_config.get("kubernetes_master", "10.132.0.14")
            master_user = k8s_config.get("kubernetes_user", "antonios-icontinuum")
            
            ssh_command = [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=10",
                f"{master_user}@{master_host}",
                curl_command
            ]
            
            result = subprocess.run(
                ssh_command,
                capture_output=True,
                text=True,
                timeout=90
            )
            
            if result.returncode == 0:
                # Parse the response time from curl output
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
            "UPPER" if above upper threshold,
            "LOWER" if below lower threshold,
            None if within acceptable range
        """
        if self.ema_rt is None:
            return None
        
        if self.ema_rt > self.upper_threshold:
            return "UPPER_THRESHOLD_EXCEEDED"
        elif self.ema_rt < self.lower_threshold:
            return "LOWER_THRESHOLD_EXCEEDED"
        
        return None
    
    def _handle_violation(self, violation_type: str):
        """
        Handle a detected violation by invoking Decision Maker and Action Executor.
        
        Args:
            violation_type: Type of violation detected
        """
        logger.warning(f"=" * 60)
        logger.warning(f"VIOLATION DETECTED: {violation_type}")
        logger.warning(f"Current EMA: {self.ema_rt:.3f}s")
        logger.warning(f"Thresholds: [{self.lower_threshold}s, {self.upper_threshold}s]")
        logger.warning(f"=" * 60)
        
        self.stats["violations_detected"] += 1
        
        # Step 1: Collect system data
        logger.info("Collecting system data...")
        system_state = self.data_collector.collect_all()
        formatted_state = self.data_collector.format_for_llm(system_state, violation_type)
        
        # Step 2: Get LLM recommendation
        logger.info("Querying LLM for recommendation...")
        recommendation = self.decision_maker.analyze_and_recommend(
            formatted_state,
            violation_type
        )
        
        logger.info(f"LLM Analysis: {recommendation.get('analysis', 'N/A')}")
        logger.info(f"Recommended Action: {recommendation.get('action', 'none')}")
        
        # Step 3: Execute the action
        if recommendation.get("action") != "none":
            logger.info("Executing recommended action...")
            result = self.action_executor.execute(
                action=recommendation.get("action"),
                parameters=recommendation.get("parameters", {}),
                analysis=recommendation.get("analysis", "")
            )
            
            if result["success"]:
                logger.info(f"Action successful: {result['message']}")
                self.stats["actions_taken"] += 1
                
                # Wait for system to stabilize
                logger.info(f"Waiting {self.wait_after_action}s for system to stabilize...")
                time.sleep(self.wait_after_action)
                
                # Reset EMA after action to avoid immediate re-triggering
                self.ema_rt = None
            else:
                logger.error(f"Action failed: {result['message']}")
        else:
            logger.info("No action recommended by LLM")
    
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
        self.stats["total_requests"] += 1
        
        # Update EMA
        self._update_ema(rt)
        result["ema_rt"] = self.ema_rt
        
        # Add to data collector history
        self.data_collector.add_response_time(rt)
        
        # Log current state
        logger.info(f"RT: {rt:.3f}s | EMA: {self.ema_rt:.3f}s | Thresholds: [{self.lower_threshold}, {self.upper_threshold}]")
        
        # Check for violations
        violation = self._check_violation()
        
        if violation:
            result["violation"] = violation
            self._handle_violation(violation)
            result["action_taken"] = True
        
        return result
    
    def run(self, max_iterations: Optional[int] = None, callback: Optional[Callable] = None):
        """
        Run the continuous monitoring loop.
        
        Args:
            max_iterations: Optional limit on number of iterations (None = run forever)
            callback: Optional callback function called after each iteration
        """
        logger.info("=" * 60)
        logger.info("Starting Intent Watch Loop")
        logger.info(f"Upper threshold: {self.upper_threshold}s")
        logger.info(f"Lower threshold: {self.lower_threshold}s")
        logger.info(f"Check interval: {self.check_interval}s")
        logger.info(f"EMA alpha: {self.ema_alpha}")
        logger.info("=" * 60)
        
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
        logger.info("Intent Watch Loop Statistics")
        logger.info(f"  Total requests: {self.stats['total_requests']}")
        logger.info(f"  Violations detected: {self.stats['violations_detected']}")
        logger.info(f"  Actions taken: {self.stats['actions_taken']}")
        logger.info(f"  Start time: {self.stats['start_time']}")
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
            "upper_threshold": self.upper_threshold,
            "lower_threshold": self.lower_threshold,
            "stats": self.stats.copy()
        }