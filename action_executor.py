"""
Action Executor Module for IntentContinuum

This module executes the corrective actions recommended by the LLM Decision Maker.
It interfaces with:
- Kubernetes API for compute actions (scaling, placement)
- ONOS API for network actions (flow scheduling)

Actions supported:
- horizontal_scaling: Adjust replica count
- vertical_scaling: Adjust CPU/memory limits
- service_placement: Move pod to different node
- flow_scheduling: Reroute network traffic
"""

import logging
import time
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import requests
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActionExecutor:
    """
    Executes corrective actions on the Kubernetes cluster and SDN network.
    
    This is called after the Decision Maker recommends an action.
    """
    
    def __init__(self, namespace: str = "default", onos_url: str = "http://localhost:8181"):
        """
        Initialize the Action Executor.
        
        Args:
            namespace: Kubernetes namespace where app is deployed
            onos_url: ONOS controller REST API URL
        """
        self.namespace = namespace
        self.onos_url = onos_url
        self.onos_auth = ("onos", "rocks")  # Default ONOS credentials
        
        # Load Kubernetes configuration
        try:
            # Try in-cluster config first (when running inside K8s)
            config.load_incluster_config()
            logger.info("Loaded in-cluster Kubernetes config")
        except config.ConfigException:
            # Fall back to kubeconfig file
            try:
                config.load_kube_config()
                logger.info("Loaded kubeconfig from file")
            except config.ConfigException as e:
                logger.error(f"Could not load Kubernetes config: {e}")
                raise
        
        # Initialize Kubernetes API clients
        self.apps_v1 = client.AppsV1Api()
        self.core_v1 = client.CoreV1Api()
        
        logger.info(f"ActionExecutor initialized for namespace: {namespace}")
    
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
        
        logger.info(f"Scaling {deployment_name} to {replicas} replicas")
        
        try:
            # Get current deployment
            deployment = self.apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace
            )
            
            current_replicas = deployment.spec.replicas
            logger.info(f"Current replicas: {current_replicas}")
            
            if current_replicas == replicas:
                return {
                    "success": True,
                    "message": f"Already at {replicas} replicas",
                    "action": "horizontal_scaling",
                    "changed": False
                }
            
            # Patch the deployment
            patch = {"spec": {"replicas": replicas}}
            self.apps_v1.patch_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace,
                body=patch
            )
            
            logger.info(f"Successfully scaled {deployment_name}: {current_replicas} -> {replicas}")
            
            return {
                "success": True,
                "message": f"Scaled {deployment_name} from {current_replicas} to {replicas} replicas",
                "action": "horizontal_scaling",
                "changed": True,
                "previous_replicas": current_replicas,
                "new_replicas": replicas
            }
            
        except ApiException as e:
            error_msg = f"Kubernetes API error: {e.status} - {e.reason}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
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
        
        try:
            # Get current deployment
            deployment = self.apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace
            )
            
            # Build the resource limits patch
            resources = {}
            if cpu_limit:
                resources["cpu"] = cpu_limit
            if memory_limit:
                resources["memory"] = memory_limit
            
            # Patch the first container's resources
            patch = {
                "spec": {
                    "template": {
                        "spec": {
                            "containers": [{
                                "name": deployment.spec.template.spec.containers[0].name,
                                "resources": {
                                    "limits": resources,
                                    "requests": resources
                                }
                            }]
                        }
                    }
                }
            }
            
            self.apps_v1.patch_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace,
                body=patch
            )
            
            logger.info(f"Successfully updated resources for {deployment_name}")
            
            return {
                "success": True,
                "message": f"Updated {deployment_name} resources: CPU={cpu_limit}, Mem={memory_limit}",
                "action": "vertical_scaling",
                "changed": True
            }
            
        except ApiException as e:
            error_msg = f"Kubernetes API error: {e.status} - {e.reason}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def _execute_service_placement(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move a pod to a different node using nodeSelector.
        
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
        
        try:
            # Verify target node exists
            try:
                self.core_v1.read_node(name=target_node)
            except ApiException:
                return {"success": False, "error": f"Node {target_node} not found"}
            
            # Patch deployment with nodeSelector
            patch = {
                "spec": {
                    "template": {
                        "spec": {
                            "nodeSelector": {
                                "kubernetes.io/hostname": target_node
                            }
                        }
                    }
                }
            }
            
            self.apps_v1.patch_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace,
                body=patch
            )
            
            logger.info(f"Successfully set nodeSelector for {deployment_name} to {target_node}")
            
            return {
                "success": True,
                "message": f"Configured {deployment_name} to run on {target_node}",
                "action": "service_placement",
                "changed": True,
                "target_node": target_node
            }
            
        except ApiException as e:
            error_msg = f"Kubernetes API error: {e.status} - {e.reason}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def _execute_flow_scheduling(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reroute network traffic through ONOS SDN controller.
        
        Args:
            params: Must contain:
                - src_ip: Source IP address
                - dst_ip: Destination IP address
                - path: List of switch IDs for the new path
        
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
            # Using HostToHostIntent for simplicity
            intent = {
                "type": "HostToHostIntent",
                "appId": "org.onosproject.cli",
                "priority": 100,
                "one": f"{src_ip}/-1",  # Host ID format
                "two": f"{dst_ip}/-1"
            }
            
            # If specific path is provided, use PointToPointIntent with waypoints
            if path:
                logger.info(f"Path specified: {path}")
                # For now, we use HostToHostIntent and let ONOS find the path
                # Full path control would require more complex intent configuration
            
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
    
    def get_deployment_status(self, deployment_name: str) -> Optional[Dict[str, Any]]:
        """
        Get current status of a deployment.
        
        Args:
            deployment_name: Name of the deployment
            
        Returns:
            Status dictionary or None if not found
        """
        try:
            deployment = self.apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace
            )
            
            return {
                "name": deployment_name,
                "replicas": deployment.spec.replicas,
                "ready_replicas": deployment.status.ready_replicas or 0,
                "available_replicas": deployment.status.available_replicas or 0
            }
        except ApiException:
            return None
    
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
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_deployment_status(deployment_name)
            if status:
                if status["ready_replicas"] == status["replicas"]:
                    logger.info(f"Rollout complete: {status['ready_replicas']}/{status['replicas']} ready")
                    return True
                logger.debug(f"Rollout in progress: {status['ready_replicas']}/{status['replicas']} ready")
            time.sleep(5)
        
        logger.warning(f"Rollout timeout after {timeout}s")
        return False