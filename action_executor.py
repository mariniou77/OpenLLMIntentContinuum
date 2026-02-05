"""
Action Executor Module

This module executes the actions recommended by the Decision Maker.
It interfaces with Kubernetes and ONOS to implement changes.

Supported actions:
- horizontal_scaling: Scale deployment replicas up or down
- vertical_scaling: Adjust CPU/memory limits (future)
- service_placement: Migrate pods to different nodes (future)
- flow_scheduling: Update network flow rules (future)
"""

import logging
import time
from typing import Optional

from utils.kubernetes_client import KubernetesClient
from utils.onos_client import ONOSClient

logger = logging.getLogger(__name__)


class ActionExecutor:
    """
    Executes recommended actions on the infrastructure.
    
    This class is responsible for translating high-level actions
    (like "scale deployment X to Y replicas") into actual API calls
    to Kubernetes and ONOS.
    """
    
    def __init__(self, config: dict):
        """
        Initialize Action Executor with configuration.
        
        Args:
            config: Configuration dictionary containing endpoints and settings
        """
        self.config = config
        
        # Initialize clients
        self.k8s_client = KubernetesClient(
            master_ip=config["endpoints"]["kubernetes_master"]
        )
        
        self.onos_client = ONOSClient(
            base_url=config["endpoints"]["onos"],
            username=config["endpoints"]["onos_user"],
            password=config["endpoints"]["onos_password"]
        )
        
        # Get enabled actions from config
        self.enabled_actions = config.get("actions", {})
        
        # Get deployment constraints from config
        self.deployment_config = {}
        for dep in config.get("kubernetes", {}).get("deployments", []):
            self.deployment_config[dep["name"]] = {
                "min_replicas": dep.get("min_replicas", 1),
                "max_replicas": dep.get("max_replicas", 5)
            }
        
        # Track action history
        self.action_history = []
    
    def execute(self, action: dict) -> dict:
        """
        Execute a recommended action.
        
        This is the main entry point called after the Decision Maker
        provides a recommendation.
        
        Args:
            action: Action dictionary from Decision Maker containing:
                - action: Action type (horizontal_scaling, etc.)
                - parameters: Action-specific parameters
                - analysis: LLM's analysis (for logging)
                
        Returns:
            Result dictionary with:
                - success: Boolean indicating if action succeeded
                - message: Description of what happened
                - details: Additional information
        """
        action_type = action.get("action", "none")
        parameters = action.get("parameters", {})
        analysis = action.get("analysis", "No analysis")
        
        logger.info(f"Executing action: {action_type}")
        logger.info(f"Analysis: {analysis}")
        
        # Check if action type is "none"
        if action_type == "none":
            return {
                "success": True,
                "message": "No action required",
                "details": {"analysis": analysis}
            }
        
        # Check if action is enabled
        if not self.enabled_actions.get(action_type, False):
            logger.warning(f"Action '{action_type}' is not enabled in config")
            return {
                "success": False,
                "message": f"Action '{action_type}' is disabled",
                "details": {"enabled_actions": self.enabled_actions}
            }
        
        # Route to appropriate handler
        if action_type == "horizontal_scaling":
            result = self._execute_horizontal_scaling(parameters)
        elif action_type == "vertical_scaling":
            result = self._execute_vertical_scaling(parameters)
        elif action_type == "service_placement":
            result = self._execute_service_placement(parameters)
        elif action_type == "flow_scheduling":
            result = self._execute_flow_scheduling(parameters)
        else:
            logger.error(f"Unknown action type: {action_type}")
            result = {
                "success": False,
                "message": f"Unknown action type: {action_type}",
                "details": {}
            }
        
        # Record action in history
        self._record_action(action_type, parameters, result)
        
        return result
    
    def _execute_horizontal_scaling(self, parameters: dict) -> dict:
        """
        Execute horizontal scaling action.
        
        Args:
            parameters: Dictionary containing:
                - deployment_name: Name of deployment to scale
                - replicas: Target number of replicas
                
        Returns:
            Result dictionary
        """
        deployment_name = parameters.get("deployment_name")
        target_replicas = parameters.get("replicas")
        
        if not deployment_name or target_replicas is None:
            return {
                "success": False,
                "message": "Missing required parameters (deployment_name or replicas)",
                "details": parameters
            }
        
        # Get deployment constraints
        constraints = self.deployment_config.get(deployment_name, {
            "min_replicas": 1,
            "max_replicas": 5
        })
        
        # Enforce constraints
        min_replicas = constraints["min_replicas"]
        max_replicas = constraints["max_replicas"]
        
        original_target = target_replicas
        target_replicas = max(min_replicas, min(max_replicas, int(target_replicas)))
        
        if target_replicas != original_target:
            logger.info(f"Adjusted replicas from {original_target} to {target_replicas} (constraints: {min_replicas}-{max_replicas})")
        
        # Get current replica count (case-insensitive match)
        deployments = self.k8s_client.get_deployments()
        current_replicas = None
        actual_deployment_name = None
        
        for dep in deployments:
            if dep["name"].lower() == deployment_name.lower():
                current_replicas = dep["replicas_desired"]
                actual_deployment_name = dep["name"]  # Use actual name from cluster
                break
        
        # Use the actual deployment name from the cluster
        if actual_deployment_name:
            deployment_name = actual_deployment_name
        
        if current_replicas is None:
            return {
                "success": False,
                "message": f"Deployment '{deployment_name}' not found",
                "details": {"available_deployments": [d["name"] for d in deployments]}
            }
        
        # Check if scaling is needed
        if current_replicas == target_replicas:
            return {
                "success": True,
                "message": f"Deployment '{deployment_name}' already has {target_replicas} replicas",
                "details": {
                    "deployment": deployment_name,
                    "replicas": target_replicas,
                    "action_taken": False
                }
            }
        
        # Execute scaling
        logger.info(f"Scaling {deployment_name}: {current_replicas} -> {target_replicas} replicas")
        
        success = self.k8s_client.scale_deployment(
            deployment_name=deployment_name,
            replicas=target_replicas
        )
        
        if success:
            return {
                "success": True,
                "message": f"Scaled '{deployment_name}' from {current_replicas} to {target_replicas} replicas",
                "details": {
                    "deployment": deployment_name,
                    "previous_replicas": current_replicas,
                    "new_replicas": target_replicas,
                    "action_taken": True
                }
            }
        else:
            return {
                "success": False,
                "message": f"Failed to scale '{deployment_name}'",
                "details": {
                    "deployment": deployment_name,
                    "target_replicas": target_replicas
                }
            }
    
    def _execute_vertical_scaling(self, parameters: dict) -> dict:
        """
        Execute vertical scaling action (adjust CPU/memory limits).
        
        NOT YET IMPLEMENTED.
        
        Args:
            parameters: Dictionary containing scaling parameters
            
        Returns:
            Result dictionary
        """
        logger.warning("Vertical scaling is not yet implemented")
        return {
            "success": False,
            "message": "Vertical scaling is not yet implemented",
            "details": parameters
        }
    
    def _execute_service_placement(self, parameters: dict) -> dict:
        """
        Execute service placement action (migrate pod to different node).
        
        NOT YET IMPLEMENTED.
        
        Args:
            parameters: Dictionary containing placement parameters
            
        Returns:
            Result dictionary
        """
        logger.warning("Service placement is not yet implemented")
        return {
            "success": False,
            "message": "Service placement is not yet implemented",
            "details": parameters
        }
    
    def _execute_flow_scheduling(self, parameters: dict) -> dict:
        """
        Execute flow scheduling action (update network routes).
        
        NOT YET IMPLEMENTED.
        
        Args:
            parameters: Dictionary containing flow parameters
            
        Returns:
            Result dictionary
        """
        logger.warning("Flow scheduling is not yet implemented")
        return {
            "success": False,
            "message": "Flow scheduling is not yet implemented",
            "details": parameters
        }
    
    def _record_action(self, action_type: str, parameters: dict, result: dict):
        """
        Record an action in the history for tracking.
        
        Args:
            action_type: Type of action executed
            parameters: Parameters used
            result: Result of the action
        """
        record = {
            "timestamp": time.time(),
            "action_type": action_type,
            "parameters": parameters,
            "success": result.get("success", False),
            "message": result.get("message", "")
        }
        
        self.action_history.append(record)
        
        # Keep only last 100 actions
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-100:]
    
    def get_action_history(self, limit: int = 10) -> list:
        """
        Get recent action history.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of recent action records
        """
        return self.action_history[-limit:]
    
    def get_current_state(self) -> dict:
        """
        Get current state of all managed deployments.
        
        Returns:
            Dictionary with deployment states
        """
        deployments = self.k8s_client.get_deployments()
        
        state = {}
        for dep in deployments:
            name = dep["name"]
            if name in self.deployment_config or "microservice" in name or "db" in name:
                state[name] = {
                    "replicas_desired": dep["replicas_desired"],
                    "replicas_ready": dep["replicas_ready"],
                    "replicas_available": dep["replicas_available"]
                }
        
        return state