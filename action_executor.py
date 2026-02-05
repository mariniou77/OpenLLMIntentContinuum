"""
Action Executor Module for IntentContinuum

This module executes the corrective actions recommended by the LLM Decision Maker.
It interfaces with:
- Kubernetes via SSH + kubectl on master node
- ONOS API for network actions (flow scheduling)

Actions supported:
- horizontal_scaling: Adjust replica count
- vertical_scaling: Adjust CPU/memory limits
- service_placement: Move pod to different node
- flow_scheduling: Reroute network traffic
"""

import subprocess
import json
import logging
import time
import requests
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActionExecutor:
    """
    Executes corrective actions on the Kubernetes cluster and SDN network.
    
    Uses SSH + kubectl to communicate with Kubernetes (same pattern as KubernetesClient).
    """
    
    def __init__(self, master_ip: str, username: str = "antonios-icontinuum",
                 namespace: str = "default", onos_url: str = "http://localhost:8181"):
        """
        Initialize the Action Executor.
        
        Args:
            master_ip: IP address of the Kubernetes master node
            username: SSH username for the master node
            namespace: Kubernetes namespace where app is deployed
            onos_url: ONOS controller REST API URL
        """
        self.master_ip = master_ip
        self.username = username
        self.namespace = namespace
        self.onos_url = onos_url
        self.onos_auth = ("onos", "rocks")  # Default ONOS credentials
        
        logger.info(f"ActionExecutor initialized - Master: {master_ip}, Namespace: {namespace}")
    
    def _run_kubectl(self, command: str, timeout: int = 30) -> tuple[bool, str]:
        """
        Run kubectl command on master node via SSH.
        
        Args:
            command: kubectl command to run (without 'kubectl' prefix)
            timeout: Command timeout in seconds
            
        Returns:
            Tuple of (success: bool, output: str)
        """
        ssh_command = f"ssh -o StrictHostKeyChecking=no {self.username}@{self.master_ip} 'sudo kubectl {command}'"
        
        try:
            result = subprocess.run(
                ssh_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode != 0:
                logger.error(f"kubectl error: {result.stderr}")
                return False, result.stderr
            
            return True, result.stdout
            
        except subprocess.TimeoutExpired:
            logger.error(f"kubectl command timed out: {command}")
            return False, "Command timed out"
        except Exception as e:
            logger.error(f"SSH/kubectl error: {e}")
            return False, str(e)
    
    def execute(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the recommended action.
        
        Args:
            action: Action dictionary from Decision Maker containing:
                - action: Action type (horizontal_scaling, vertical_scaling, etc.)
                - parameters: Action-specific parameters
                - analysis: LLM's analysis
                - source: Source of violation (compute/network)
        
        Returns:
            Result dictionary with success status and details
        """
        action_type = action.get("action", "none")
        parameters = action.get("parameters", {})
        
        logger.info(f"Executing action: {action_type}")
        logger.info(f"Parameters: {parameters}")
        
        # Route to appropriate handler
        if action_type == "horizontal_scaling":
            return self._execute_horizontal_scaling(parameters)
        elif action_type == "vertical_scaling":
            return self._execute_vertical_scaling(parameters)
        elif action_type == "service_placement":
            return self._execute_service_placement(parameters)
        elif action_type == "flow_scheduling":
            return self._execute_flow_scheduling(parameters)
        elif action_type == "none":
            logger.info("No action required")
            return {"success": True, "message": "No action taken", "action": "none"}
        else:
            logger.warning(f"Unknown action type: {action_type}")
            return {"success": False, "error": f"Unknown action: {action_type}"}
    
    def _execute_horizontal_scaling(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scale deployment horizontally by adjusting replica count.
        
        Args:
            params: Must contain:
                - deployment_name: Name of the deployment to scale
                - replicas: Target number of replicas
        
        Returns:
            Result dictionary
        """
        deployment_name = params.get("deployment_name")
        replicas = params.get("replicas")
        
        if not deployment_name or replicas is None:
            return {
                "success": False,
                "error": "Missing deployment_name or replicas parameter"
            }
        
        try:
            replicas = int(replicas)
        except (ValueError, TypeError):
            return {"success": False, "error": f"Invalid replicas value: {replicas}"}
        
        # Clamp replicas to reasonable range
        replicas = max(1, min(10, replicas))
        
        # Get current replica count
        current_replicas = self._get_deployment_replicas(deployment_name)
        
        if current_replicas is not None and current_replicas == replicas:
            return {
                "success": True,
                "message": f"Already at {replicas} replicas",
                "action": "horizontal_scaling",
                "changed": False
            }
        
        logger.info(f"Scaling {deployment_name} to {replicas} replicas")
        
        # Execute scale command
        command = f"scale deployment {deployment_name} --replicas={replicas} -n {self.namespace}"
        success, output = self._run_kubectl(command)
        
        if success:
            logger.info(f"Successfully scaled {deployment_name}: {current_replicas} -> {replicas}")
            return {
                "success": True,
                "message": f"Scaled {deployment_name} from {current_replicas} to {replicas} replicas",
                "action": "horizontal_scaling",
                "changed": True,
                "previous_replicas": current_replicas,
                "new_replicas": replicas
            }
        else:
            return {"success": False, "error": output}
    
    def _execute_vertical_scaling(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scale deployment vertically by adjusting CPU/memory limits.
        
        Args:
            params: Must contain:
                - deployment_name: Name of the deployment
                - cpu_limit: New CPU limit (e.g., "500m")
                - memory_limit: New memory limit (e.g., "512Mi")
        
        Returns:
            Result dictionary
        """
        deployment_name = params.get("deployment_name")
        cpu_limit = params.get("cpu_limit")
        memory_limit = params.get("memory_limit")
        
        if not deployment_name:
            return {"success": False, "error": "Missing deployment_name"}
        
        if not cpu_limit and not memory_limit:
            return {"success": False, "error": "Must specify cpu_limit or memory_limit"}
        
        logger.info(f"Vertical scaling {deployment_name}: CPU={cpu_limit}, Mem={memory_limit}")
        
        # Build resource string
        resources = []
        if cpu_limit:
            resources.append(f"cpu={cpu_limit}")
        if memory_limit:
            resources.append(f"memory={memory_limit}")
        
        resource_str = ",".join(resources)
        
        # Use kubectl set resources command
        command = f"set resources deployment {deployment_name} --limits={resource_str} --requests={resource_str} -n {self.namespace}"
        success, output = self._run_kubectl(command)
        
        if success:
            logger.info(f"Successfully updated resources for {deployment_name}")
            return {
                "success": True,
                "message": f"Updated {deployment_name} resources: CPU={cpu_limit}, Mem={memory_limit}",
                "action": "vertical_scaling",
                "changed": True
            }
        else:
            return {"success": False, "error": output}
    
    def _execute_service_placement(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move a pod to a different node using nodeSelector patch.
        
        Args:
            params: Must contain:
                - deployment_name: Name of the deployment
                - target_node: Node to place the pod on
        
        Returns:
            Result dictionary
        """
        deployment_name = params.get("deployment_name")
        target_node = params.get("target_node")
        
        if not deployment_name or not target_node:
            return {"success": False, "error": "Missing deployment_name or target_node"}
        
        logger.info(f"Placing {deployment_name} on node {target_node}")
        
        # Verify target node exists
        success, output = self._run_kubectl(f"get node {target_node}")
        if not success:
            return {"success": False, "error": f"Node {target_node} not found"}
        
        # Patch deployment with nodeSelector
        patch_json = json.dumps({
            "spec": {
                "template": {
                    "spec": {
                        "nodeSelector": {
                            "kubernetes.io/hostname": target_node
                        }
                    }
                }
            }
        })
        
        # Escape quotes for shell
        patch_json_escaped = patch_json.replace('"', '\\"')
        
        command = f'patch deployment {deployment_name} -n {self.namespace} -p "{patch_json_escaped}"'
        success, output = self._run_kubectl(command)
        
        if success:
            logger.info(f"Successfully set nodeSelector for {deployment_name} to {target_node}")
            return {
                "success": True,
                "message": f"Configured {deployment_name} to run on {target_node}",
                "action": "service_placement",
                "changed": True,
                "target_node": target_node
            }
        else:
            return {"success": False, "error": output}
    
    def _execute_flow_scheduling(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reroute network traffic through ONOS SDN controller.
        
        Args:
            params: Must contain:
                - src_ip: Source IP address
                - dst_ip: Destination IP address
                - path: List of switch IDs for the new path (optional)
        
        Returns:
            Result dictionary
        """
        src_ip = params.get("src_ip")
        dst_ip = params.get("dst_ip")
        path = params.get("path", [])
        
        if not src_ip or not dst_ip:
            return {"success": False, "error": "Missing src_ip or dst_ip"}
        
        logger.info(f"Rerouting traffic {src_ip} -> {dst_ip} via {path}")
        
        try:
            # Create an ONOS intent for the new path
            intent = {
                "type": "HostToHostIntent",
                "appId": "org.onosproject.cli",
                "priority": 100,
                "one": f"{src_ip}/-1",
                "two": f"{dst_ip}/-1"
            }
            
            # Submit intent to ONOS
            url = f"{self.onos_url}/onos/v1/intents"
            response = requests.post(
                url,
                json=intent,
                auth=self.onos_auth,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code in [200, 201]:
                logger.info("Successfully submitted intent to ONOS")
                return {
                    "success": True,
                    "message": f"Rerouted traffic between {src_ip} and {dst_ip}",
                    "action": "flow_scheduling",
                    "changed": True
                }
            else:
                error_msg = f"ONOS API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Network error communicating with ONOS: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def _get_deployment_replicas(self, deployment_name: str) -> Optional[int]:
        """
        Get current replica count for a deployment.
        
        Args:
            deployment_name: Name of the deployment
            
        Returns:
            Number of replicas or None if not found
        """
        command = f"get deployment {deployment_name} -n {self.namespace} -o jsonpath='{{.spec.replicas}}'"
        success, output = self._run_kubectl(command)
        
        if success and output.strip():
            try:
                return int(output.strip())
            except ValueError:
                return None
        return None
    
    def get_deployment_status(self, deployment_name: str) -> Optional[Dict[str, Any]]:
        """
        Get current status of a deployment.
        
        Args:
            deployment_name: Name of the deployment
            
        Returns:
            Status dictionary or None if not found
        """
        command = f"get deployment {deployment_name} -n {self.namespace} -o json"
        success, output = self._run_kubectl(command)
        
        if not success:
            return None
        
        try:
            deployment = json.loads(output)
            return {
                "name": deployment_name,
                "replicas": deployment.get("spec", {}).get("replicas", 0),
                "ready_replicas": deployment.get("status", {}).get("readyReplicas", 0),
                "available_replicas": deployment.get("status", {}).get("availableReplicas", 0)
            }
        except json.JSONDecodeError:
            return None
    
    def list_deployments(self) -> list:
        """
        List all deployments in the namespace.
        
        Returns:
            List of deployment names
        """
        command = f"get deployments -n {self.namespace} -o jsonpath='{{.items[*].metadata.name}}'"
        success, output = self._run_kubectl(command)
        
        if success and output.strip():
            return output.strip().split()
        return []
    
    def wait_for_rollout(self, deployment_name: str, timeout: int = 120) -> bool:
        """
        Wait for a deployment rollout to complete.
        
        Args:
            deployment_name: Name of the deployment
            timeout: Maximum seconds to wait
            
        Returns:
            True if rollout completed, False if timeout
        """
        logger.info(f"Waiting for {deployment_name} rollout (timeout: {timeout}s)")
        
        command = f"rollout status deployment/{deployment_name} -n {self.namespace} --timeout={timeout}s"
        success, output = self._run_kubectl(command, timeout=timeout + 10)
        
        if success:
            logger.info(f"Rollout complete for {deployment_name}")
            return True
        else:
            logger.warning(f"Rollout timeout or failed: {output}")
            return False