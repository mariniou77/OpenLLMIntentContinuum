"""
Action Executor Module

This module executes the actions recommended by the Decision Maker.
Supported actions:
- horizontal_scaling: Change the number of pod replicas
- vertical_scaling: Change CPU/memory limits for a deployment
- service_placement: Move a pod to a different node
- flow_scheduling: Change network traffic path via ONOS
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ActionExecutor:
    """
    Executes recommended actions on the infrastructure.
    """
    
    def __init__(self, config: dict, kubernetes_client, onos_client=None):
        """
        Initialize the Action Executor.
        
        Args:
            config: Configuration dictionary
            kubernetes_client: Kubernetes client instance
            onos_client: ONOS client instance (optional, for flow scheduling)
        """
        self.config = config
        self.k8s_client = kubernetes_client
        self.onos_client = onos_client
        self.action_history = []
        self.max_history = 100
        
    def execute(self, action: str, parameters: dict, analysis: str = "") -> Dict[str, Any]:
        """
        Execute the recommended action.
        
        Args:
            action: Action type (horizontal_scaling, vertical_scaling, etc.)
            parameters: Action parameters
            analysis: LLM analysis text for logging
            
        Returns:
            Dictionary with success status and message
        """
        logger.info(f"Executing action: {action}")
        logger.info(f"Analysis: {analysis}")
        
        # Check if action is enabled
        if not self.config.get("actions", {}).get(action, False):
            return {
                "success": False,
                "message": f"Action '{action}' is not enabled in configuration"
            }
        
        # Route to appropriate handler
        handlers = {
            "horizontal_scaling": self._execute_horizontal_scaling,
            "vertical_scaling": self._execute_vertical_scaling,
            "service_placement": self._execute_service_placement,
            "flow_scheduling": self._execute_flow_scheduling
        }
        
        handler = handlers.get(action)
        if not handler:
            return {
                "success": False,
                "message": f"Unknown action type: {action}"
            }
        
        result = handler(parameters)
        
        # Record action in history
        self._record_action(action, parameters, result, analysis)
        
        return result
    
    def _execute_horizontal_scaling(self, parameters: dict) -> Dict[str, Any]:
        """
        Execute horizontal scaling (change replica count).
        
        Args:
            parameters: Must contain 'deployment_name' and 'replicas'
            
        Returns:
            Result dictionary
        """
        deployment_name = parameters.get("deployment_name", "").lower()
        target_replicas = parameters.get("replicas", 2)
        
        if not deployment_name:
            return {"success": False, "message": "No deployment name provided"}
        
        # Validate against constraints
        constraints = self._get_deployment_constraints(deployment_name)
        min_replicas = constraints.get("min_replicas", 1)
        max_replicas = constraints.get("max_replicas", 5)
        
        # Clamp replicas to valid range
        target_replicas = max(min_replicas, min(max_replicas, int(target_replicas)))
        
        # Get current replica count
        current_state = self.k8s_client.get_deployments()
        current_replicas = None
        actual_deployment_name = None
        
        for dep in current_state:
            if dep.get("name", "").lower() == deployment_name:
                current_replicas = dep.get("replicas_desired") or dep.get("replicas_ready") or 0
                actual_deployment_name = dep.get("name")
                break
        
        if actual_deployment_name is None:
            return {
                "success": False,
                "message": f"Deployment '{deployment_name}' not found"
            }
        
        if current_replicas == target_replicas:
            return {
                "success": True,
                "message": f"Deployment '{actual_deployment_name}' already has {target_replicas} replicas"
            }
        
        # Execute scaling
        logger.info(f"Scaling {actual_deployment_name}: {current_replicas} -> {target_replicas} replicas")
        result = self.k8s_client.scale_deployment(actual_deployment_name, target_replicas)
        
        if result.get("success"):
            return {
                "success": True,
                "message": f"Scaled '{actual_deployment_name}' from {current_replicas} to {target_replicas} replicas"
            }
        else:
            return {
                "success": False,
                "message": f"Failed to scale: {result.get('error', 'Unknown error')}"
            }
    
    def _execute_vertical_scaling(self, parameters: dict) -> Dict[str, Any]:
        """
        Execute vertical scaling (change CPU/memory limits).
        
        This uses 'kubectl set resources' to update resource limits.
        The pods will be restarted with the new limits.
        
        Args:
            parameters: Must contain 'deployment_name', 'cpu_limit', 'memory_limit'
            
        Returns:
            Result dictionary
        """
        deployment_name = parameters.get("deployment_name", "").lower()
        cpu_limit = parameters.get("cpu_limit", "500m")
        memory_limit = parameters.get("memory_limit", "512Mi")
        
        if not deployment_name:
            return {"success": False, "message": "No deployment name provided"}
        
        # Find the actual deployment name (case-insensitive)
        current_state = self.k8s_client.get_deployments()
        actual_deployment_name = None
        
        for dep in current_state:
            if dep.get("name", "").lower() == deployment_name:
                actual_deployment_name = dep.get("name")
                break
        
        if actual_deployment_name is None:
            return {
                "success": False,
                "message": f"Deployment '{deployment_name}' not found"
            }
        
        logger.info(f"Vertical scaling {actual_deployment_name}: CPU={cpu_limit}, Memory={memory_limit}")
        
        # Use kubectl set resources command (more reliable than patch for resources)
        result = self.k8s_client.set_resources(
            actual_deployment_name, 
            cpu_limit, 
            memory_limit
        )
        
        if result.get("success"):
            return {
                "success": True,
                "message": f"Updated '{actual_deployment_name}' resources: CPU={cpu_limit}, Memory={memory_limit}"
            }
        else:
            return {
                "success": False,
                "message": f"Failed to update resources: {result.get('error', 'Unknown error')}"
            }
    
    def _execute_service_placement(self, parameters: dict) -> Dict[str, Any]:
        """
        Execute service placement (move pod to different node).
        
        This adds a nodeSelector to the deployment, causing pods to be
        rescheduled on the target node.
        
        Args:
            parameters: Must contain 'deployment_name' and 'target_node'
            
        Returns:
            Result dictionary
        """
        deployment_name = parameters.get("deployment_name", "").lower()
        target_node = parameters.get("target_node", "")
        
        if not deployment_name:
            return {"success": False, "message": "No deployment name provided"}
        
        if not target_node:
            return {"success": False, "message": "No target node provided"}
        
        # Find the actual deployment name
        current_state = self.k8s_client.get_deployments()
        actual_deployment_name = None
        
        for dep in current_state:
            if dep.get("name", "").lower() == deployment_name:
                actual_deployment_name = dep.get("name")
                break
        
        if actual_deployment_name is None:
            return {
                "success": False,
                "message": f"Deployment '{deployment_name}' not found"
            }
        
        # Verify target node exists
        nodes = self.k8s_client.get_nodes()
        node_names = [n.get("name", "").lower() for n in nodes]
        
        if target_node.lower() not in node_names:
            return {
                "success": False,
                "message": f"Target node '{target_node}' not found. Available: {node_names}"
            }
        
        # Build the patch to add nodeSelector
        patch_json = f'{{"spec":{{"template":{{"spec":{{"nodeSelector":{{"kubernetes.io/hostname":"{target_node}"}}}}}}}}}}'
        
        logger.info(f"Moving {actual_deployment_name} to node {target_node}")
        
        result = self.k8s_client.patch_deployment(actual_deployment_name, patch_json)
        
        if result.get("success"):
            return {
                "success": True,
                "message": f"Scheduled '{actual_deployment_name}' to run on node '{target_node}'"
            }
        else:
            return {
                "success": False,
                "message": f"Failed to update placement: {result.get('error', 'Unknown error')}"
            }
    
    def _execute_flow_scheduling(self, parameters: dict) -> Dict[str, Any]:
        """
        Execute flow scheduling (change network path via ONOS).
        
        This installs flow rules in ONOS to route traffic through
        the specified path.
        
        Args:
            parameters: Must contain 'source_switch', 'destination_switch', 'new_path'
            
        Returns:
            Result dictionary
        """
        if self.onos_client is None:
            return {
                "success": False,
                "message": "ONOS client not configured for flow scheduling"
            }
        
        source = parameters.get("source_switch", "")
        destination = parameters.get("destination_switch", "")
        new_path = parameters.get("new_path", [])
        
        if not source or not destination:
            return {
                "success": False,
                "message": "Source and destination switches are required"
            }
        
        if not new_path:
            return {
                "success": False,
                "message": "New path (list of switches) is required"
            }
        
        logger.info(f"Configuring flow path: {source} -> {destination} via {new_path}")
        
        # Install intent in ONOS for the new path
        result = self.onos_client.add_point_to_point_intent(source, destination, new_path)
        
        if result.get("success"):
            return {
                "success": True,
                "message": f"Installed flow path from {source} to {destination} via {len(new_path)} switches"
            }
        else:
            return {
                "success": False,
                "message": f"Failed to install flow: {result.get('error', 'Unknown error')}"
            }
    
    def _get_deployment_constraints(self, deployment_name: str) -> dict:
        """
        Get constraints for a deployment from configuration.
        
        Args:
            deployment_name: Name of the deployment
            
        Returns:
            Dictionary with min_replicas, max_replicas, etc.
        """
        deployments = self.config.get("kubernetes", {}).get("deployments", [])
        
        for dep in deployments:
            if dep.get("name", "").lower() == deployment_name.lower():
                return dep
        
        # Return defaults if not found
        return {
            "min_replicas": 1,
            "max_replicas": 5,
            "default_cpu": "300m",
            "default_memory": "312Mi"
        }
    
    def _record_action(self, action: str, parameters: dict, result: dict, analysis: str):
        """
        Record an action in the history.
        
        Args:
            action: Action type
            parameters: Action parameters
            result: Execution result
            analysis: LLM analysis
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "parameters": parameters,
            "success": result.get("success", False),
            "message": result.get("message", ""),
            "analysis": analysis
        }
        
        self.action_history.append(record)
        
        # Trim history if too long
        if len(self.action_history) > self.max_history:
            self.action_history = self.action_history[-self.max_history:]
    
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
        Get current state of deployments.
        
        Returns:
            Dictionary with deployment states
        """
        deployments = self.k8s_client.get_deployments()
        return {
            "deployments": deployments
        }