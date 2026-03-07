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
                # Get resource limits from the container spec
                containers = item.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
                cpu_limit = None
                memory_limit = None
                if containers:
                    resources = containers[0].get("resources", {})
                    limits = resources.get("limits", {})
                    cpu_limit = limits.get("cpu")
                    memory_limit = limits.get("memory")
                
                deployment = {
                    "name": item.get("metadata", {}).get("name"),
                    "replicas_desired": item.get("spec", {}).get("replicas", 0),
                    "replicas_ready": item.get("status", {}).get("readyReplicas", 0),
                    "replicas_available": item.get("status", {}).get("availableReplicas", 0),
                    "cpu_limit": cpu_limit,
                    "memory_limit": memory_limit
                }
                deployments.append(deployment)
            return deployments
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse deployments JSON: {e}")
            return []
    
    def scale_deployment(self, deployment_name: str, replicas: int, namespace: str = "default") -> dict:
        """
        Scale a deployment to a specific number of replicas.
        
        Args:
            deployment_name: Name of the deployment to scale
            replicas: Desired number of replicas
            namespace: Kubernetes namespace
            
        Returns:
            Dictionary with success status and message
        """
        logger.info(f"Scaling {deployment_name} to {replicas} replicas")
        output = self._run_kubectl(f"scale deployment {deployment_name} --replicas={replicas} -n {namespace}")
        
        # kubectl scale returns something like "deployment.apps/xxx scaled"
        if "scaled" in output.lower():
            logger.info(f"Successfully scaled {deployment_name} to {replicas} replicas")
            return {"success": True, "message": f"Scaled {deployment_name} to {replicas} replicas"}
        else:
            logger.error(f"Failed to scale {deployment_name}: {output}")
            return {"success": False, "error": output}
    
    def set_resources(self, deployment_name: str, cpu_limit: str, memory_limit: str, namespace: str = "default") -> dict:
        """
        Set resource limits for a deployment using 'kubectl set resources'.
        
        This is more reliable than patching for resource updates.
        
        Args:
            deployment_name: Name of the deployment
            cpu_limit: CPU limit (e.g., "500m", "1")
            memory_limit: Memory limit (e.g., "512Mi", "1Gi")
            namespace: Kubernetes namespace
            
        Returns:
            Dictionary with success status and message
        """
        logger.info(f"Setting resources for {deployment_name}: CPU={cpu_limit}, Memory={memory_limit}")
        
        command = f"set resources deployment {deployment_name} --limits=cpu={cpu_limit},memory={memory_limit} --requests=cpu={cpu_limit},memory={memory_limit} -n {namespace}"
        output = self._run_kubectl(command)
        
        # kubectl set resources returns something like "deployment.apps/xxx resource requirements updated"
        if "updated" in output.lower() or "configured" in output.lower():
            logger.info(f"Successfully set resources for {deployment_name}")
            return {"success": True, "message": f"Set resources for {deployment_name}: CPU={cpu_limit}, Memory={memory_limit}"}
        else:
            logger.error(f"Failed to set resources for {deployment_name}: {output}")
            return {"success": False, "error": output or "Unknown error"}
        
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
        # Escape the JSON for shell - replace single quotes with escaped version
        # Use double quotes in the outer command instead
        escaped_json = patch_json.replace('"', '\\"')
        command = f'patch deployment {deployment_name} -n {namespace} --type=strategic -p "{escaped_json}"'
        
        logger.info(f"Patching deployment {deployment_name}")
        output = self._run_kubectl(command)
        
        # kubectl patch returns something like "deployment.apps/xxx patched"
        if "patched" in output.lower():
            logger.info(f"Successfully patched {deployment_name}")
            return {"success": True, "message": f"Patched {deployment_name}"}
        else:
            logger.error(f"Failed to patch {deployment_name}: {output}")
            return {"success": False, "error": output or "Unknown error"}
    
    def get_cluster_summary(self) -> dict:
        """
        Get a summary of the cluster state for LLM analysis.
        
        Returns:
            Dictionary with nodes, pods, and deployments information.
            Deployments include both resource limits and current usage.
        """
        nodes = self.get_nodes()
        pods = self.get_pods()
        # Use get_deployment_metrics to include current CPU/memory usage
        deployments = self.get_deployment_metrics()
        
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
    
    def get_pod_metrics(self, namespace: str = "default") -> dict:
        """
        Get current CPU and memory usage for pods using kubectl top.
        
        Args:
            namespace: Kubernetes namespace
            
        Returns:
            Dictionary mapping pod names to their current resource usage.
        """
        output = self._run_kubectl(f"top pods -n {namespace} --no-headers")
        if not output:
            return {}
        
        metrics = {}
        try:
            for line in output.strip().split('\n'):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    pod_name = parts[0]
                    cpu_usage = parts[1]  # e.g., "25m" or "100m"
                    memory_usage = parts[2]  # e.g., "128Mi"
                    metrics[pod_name] = {
                        "cpu_usage": cpu_usage,
                        "memory_usage": memory_usage
                    }
            return metrics
        except Exception as e:
            logger.error(f"Failed to parse pod metrics: {e}")
            return {}
    
    def get_deployment_metrics(self, namespace: str = "default") -> list:
        """
        Get deployments with their resource limits AND current usage.
        
        This combines deployment spec (limits) with kubectl top (current usage).
        
        Args:
            namespace: Kubernetes namespace
            
        Returns:
            List of deployment dictionaries with limits and current usage.
        """
        deployments = self.get_deployments(namespace)
        pod_metrics = self.get_pod_metrics(namespace)
        pods = self.get_pods(namespace)
        
        # Build pod -> node mapping
        pod_nodes = {}
        for pod in pods:
            pod_nodes[pod.get("name", "")] = pod.get("node", "unknown")
        
        for dep in deployments:
            dep_name = dep.get("name", "")
            # Find matching pod metrics
            # Pod names are like "microservice1-deployment-76d476998c-4w662"
            # so we match on the deployment name prefix
            matching_pods = []
            for pod_name, metrics in pod_metrics.items():
                if pod_name.startswith(dep_name):
                    matching_pods.append(metrics)
            
            # Aggregate CPU/memory usage across replicas
            if matching_pods:
                # Sum CPU usage across all replicas for total
                total_cpu = 0
                total_mem = 0
                for pm in matching_pods:
                    try:
                        cpu_str = str(pm.get("cpu_usage", "0m")).replace("m", "").strip()
                        total_cpu += int(cpu_str)
                    except (ValueError, TypeError):
                        pass
                    try:
                        mem_str = str(pm.get("memory_usage", "0Mi")).replace("Mi", "").strip()
                        total_mem += int(mem_str)
                    except (ValueError, TypeError):
                        pass
                
                # For single replica, show as-is. For multiple, show per-replica average
                num_pods = len(matching_pods)
                if num_pods > 1:
                    dep["cpu_usage"] = f"{total_cpu // num_pods}m"
                    dep["memory_usage"] = f"{total_mem // num_pods}Mi"
                else:
                    dep["cpu_usage"] = matching_pods[0].get("cpu_usage", "0m")
                    dep["memory_usage"] = matching_pods[0].get("memory_usage", "0Mi")
            else:
                dep["cpu_usage"] = "0m"
                dep["memory_usage"] = "0Mi"
        
        return deployments