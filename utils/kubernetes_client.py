"""
Kubernetes Client via SSH

This module provides functions to interact with Kubernetes via kubectl commands
executed over SSH on the master node.

Note: We use SSH + kubectl instead of the kubernetes Python client to keep
dependencies minimal and because the SDN-Controller is not part of the K8s cluster.
"""

import subprocess
import json
import logging

logger = logging.getLogger(__name__)


class KubernetesClient:
    """Client for interacting with Kubernetes via SSH to master node."""
    
    def __init__(self, master_ip: str, username: str = "antonios-icontinuum"):
        """
        Initialize Kubernetes client.
        
        Args:
            master_ip: IP address of the Kubernetes master node
            username: SSH username for the master node
        """
        self.master_ip = master_ip
        self.username = username
    
    def _run_kubectl(self, command: str) -> str:
        """
        Run kubectl command on master node via SSH.
        
        Args:
            command: kubectl command to run (without 'kubectl' prefix)
            
        Returns:
            Command output as string
        """
        ssh_command = f"ssh -o StrictHostKeyChecking=no {self.username}@{self.master_ip} 'sudo kubectl {command}'"
        try:
            result = subprocess.run(
                ssh_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode != 0:
                logger.error(f"kubectl error: {result.stderr}")
                return ""
            return result.stdout
        except subprocess.TimeoutExpired:
            logger.error(f"kubectl command timed out: {command}")
            return ""
        except Exception as e:
            logger.error(f"SSH/kubectl error: {e}")
            return ""
    
    def get_nodes(self) -> list:
        """
        Get all nodes in the cluster.
        
        Returns:
            List of node dictionaries with name, status, and resource info.
        """
        output = self._run_kubectl("get nodes -o json")
        if not output:
            return []
        
        try:
            data = json.loads(output)
            nodes = []
            for item in data.get("items", []):
                node = {
                    "name": item.get("metadata", {}).get("name"),
                    "status": self._get_node_status(item),
                    "cpu_capacity": item.get("status", {}).get("capacity", {}).get("cpu"),
                    "memory_capacity": item.get("status", {}).get("capacity", {}).get("memory")
                }
                nodes.append(node)
            return nodes
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse nodes JSON: {e}")
            return []
    
    def _get_node_status(self, node_data: dict) -> str:
        """Extract node status from node data."""
        conditions = node_data.get("status", {}).get("conditions", [])
        for condition in conditions:
            if condition.get("type") == "Ready":
                return "Ready" if condition.get("status") == "True" else "NotReady"
        return "Unknown"
    
    def get_pods(self, namespace: str = "default") -> list:
        """
        Get all pods in a namespace.
        
        Args:
            namespace: Kubernetes namespace
            
        Returns:
            List of pod dictionaries with name, status, node, and resource info.
        """
        output = self._run_kubectl(f"get pods -n {namespace} -o json")
        if not output:
            return []
        
        try:
            data = json.loads(output)
            pods = []
            for item in data.get("items", []):
                # Get resource requests/limits from first container
                containers = item.get("spec", {}).get("containers", [{}])
                resources = containers[0].get("resources", {}) if containers else {}
                
                pod = {
                    "name": item.get("metadata", {}).get("name"),
                    "namespace": item.get("metadata", {}).get("namespace"),
                    "node": item.get("spec", {}).get("nodeName"),
                    "status": item.get("status", {}).get("phase"),
                    "ip": item.get("status", {}).get("podIP"),
                    "labels": item.get("metadata", {}).get("labels", {}),
                    "cpu_request": resources.get("requests", {}).get("cpu"),
                    "memory_request": resources.get("requests", {}).get("memory"),
                    "cpu_limit": resources.get("limits", {}).get("cpu"),
                    "memory_limit": resources.get("limits", {}).get("memory")
                }
                pods.append(pod)
            return pods
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse pods JSON: {e}")
            return []
    
    def get_deployments(self, namespace: str = "default") -> list:
        """
        Get all deployments in a namespace.
        
        Args:
            namespace: Kubernetes namespace
            
        Returns:
            List of deployment dictionaries with name and replica counts.
        """
        output = self._run_kubectl(f"get deployments -n {namespace} -o json")
        if not output:
            return []
        
        try:
            data = json.loads(output)
            deployments = []
            for item in data.get("items", []):
                deployment = {
                    "name": item.get("metadata", {}).get("name"),
                    "replicas_desired": item.get("spec", {}).get("replicas", 0),
                    "replicas_ready": item.get("status", {}).get("readyReplicas", 0),
                    "replicas_available": item.get("status", {}).get("availableReplicas", 0)
                }
                deployments.append(deployment)
            return deployments
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse deployments JSON: {e}")
            return []
    
    def scale_deployment(self, deployment_name: str, replicas: int, namespace: str = "default") -> bool:
        """
        Scale a deployment to a specific number of replicas.
        
        Args:
            deployment_name: Name of the deployment to scale
            replicas: Desired number of replicas
            namespace: Kubernetes namespace
            
        Returns:
            True if scaling was successful, False otherwise
        """
        logger.info(f"Scaling {deployment_name} to {replicas} replicas")
        output = self._run_kubectl(f"scale deployment {deployment_name} --replicas={replicas} -n {namespace}")
        
        # kubectl scale returns something like "deployment.apps/xxx scaled"
        if "scaled" in output.lower():
            logger.info(f"Successfully scaled {deployment_name} to {replicas} replicas")
            return True
        else:
            logger.error(f"Failed to scale {deployment_name}: {output}")
            return False
        
    def patch_deployment(self, deployment_name: str, patch_json: str, namespace: str = "default") -> dict:
        """
        Patch a deployment with the given JSON patch.
        
        Args:
            deployment_name: Name of the deployment to patch
            patch_json: JSON string with the patch content
            namespace: Kubernetes namespace
            
        Returns:
            Dictionary with success status and message
        """
        command = (
            f"sudo kubectl patch deployment {deployment_name} "
            f"-n {namespace} "
            f"--type=strategic "
            f"-p '{patch_json}'"
        )
        
        result = self._run_kubectl(command)
        
        if result["success"]:
            logger.info(f"Successfully patched {deployment_name}")
            return {"success": True, "message": f"Patched {deployment_name}"}
        else:
            logger.error(f"Failed to patch {deployment_name}: {result.get('error')}")
            return {"success": False, "error": result.get("error", "Unknown error")}
    
    def get_cluster_summary(self) -> dict:
        """
        Get a summary of the cluster state for LLM analysis.
        
        Returns:
            Dictionary with nodes, pods, and deployments information.
        """
        nodes = self.get_nodes()
        pods = self.get_pods()
        deployments = self.get_deployments()
        
        # Filter to only show microservice pods
        microservice_pods = [p for p in pods if "microservice" in p.get("name", "") or "db" in p.get("name", "")]
        
        return {
            "nodes": {
                "count": len(nodes),
                "list": nodes
            },
            "pods": {
                "count": len(microservice_pods),
                "list": microservice_pods
            },
            "deployments": {
                "count": len(deployments),
                "list": [d for d in deployments if "microservice" in d.get("name", "") or "db" in d.get("name", "")]
            }
        }
    
    def is_healthy(self) -> bool:
        """Check if we can connect to Kubernetes."""
        try:
            nodes = self.get_nodes()
            return len(nodes) > 0
        except Exception:
            return False