"""
Data Collector Module

This module gathers system state from all infrastructure components
and formats it for LLM analysis. It collects:
- Kubernetes cluster info (nodes, pods, deployments)
- Network topology info (ONOS switches, links, hosts)
- Monitoring metrics (sFlow-RT CPU, memory, traffic)
- Application response time history

The collected data is structured as JSON for the Decision Maker (LLM).
"""

import logging
from typing import Optional
from datetime import datetime

from utils.kubernetes_client import KubernetesClient
from utils.onos_client import ONOSClient
from utils.sflow_client import SFlowRTClient

logger = logging.getLogger(__name__)


class DataCollector:
    """
    Collects and aggregates system data from all infrastructure components.
    
    This class is responsible for gathering the current state of:
    - Kubernetes cluster (nodes, pods, deployments, resources)
    - SDN network (topology, flows, hosts)
    - System metrics (CPU, memory, network utilization)
    """
    
    def __init__(self, config: dict):
        """
        Initialize Data Collector with configuration.
        
        Args:
            config: Configuration dictionary containing endpoints and credentials
        """
        self.config = config
        
        # Build deployment defaults map from config
        deployment_defaults = {}
        for dep in config.get("kubernetes", {}).get("deployments", []):
            dep_name = dep.get("name", "")
            if dep_name:
                deployment_defaults[dep_name] = {
                    "cpu": dep.get("default_cpu", "300m"),
                    "memory": dep.get("default_memory", "312Mi")
                }
        
        # Initialize utility clients
        self.k8s_client = KubernetesClient(
            master_ip=config["endpoints"]["kubernetes_master"],
            deployment_defaults=deployment_defaults
        )
        
        self.onos_client = ONOSClient(
            base_url=config["endpoints"]["onos"],
            username=config["endpoints"]["onos_user"],
            password=config["endpoints"]["onos_password"]
        )
        
        self.sflow_client = SFlowRTClient(
            base_url=config["endpoints"]["sflow_rt"],
            sflow_config=config.get("sflow", {})
        )
        
        # Response time history storage
        self.response_times: list[dict] = []
        self.max_history_size = 100  # Keep last 100 response times
    
    def add_response_time(self, response_time: float, timestamp: Optional[datetime] = None):
        """
        Add a response time measurement to history.
        
        Args:
            response_time: Response time in seconds
            timestamp: Optional timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.response_times.append({
            "timestamp": timestamp.isoformat(),
            "response_time": response_time
        })
        
        # Keep only the last N measurements
        if len(self.response_times) > self.max_history_size:
            self.response_times = self.response_times[-self.max_history_size:]
    
    def get_response_time_summary(self, window_size: int = 30) -> dict:
        """
        Get summary statistics of recent response times.
        
        Args:
            window_size: Number of recent measurements to analyze
            
        Returns:
            Dictionary with min, max, avg, and recent response times
        """
        if not self.response_times:
            return {
                "count": 0,
                "min": None,
                "max": None,
                "avg": None,
                "recent": []
            }
        
        recent = self.response_times[-window_size:]
        times = [r["response_time"] for r in recent]
        
        return {
            "count": len(times),
            "min": round(min(times), 3),
            "max": round(max(times), 3),
            "avg": round(sum(times) / len(times), 3),
            "recent": recent[-10:]  # Last 10 for LLM context
        }
    
    def collect_cluster_info(self) -> dict:
        """
        Collect Kubernetes cluster information.
        
        Returns:
            Dictionary with nodes, pods, and deployments data
        """
        logger.info("Collecting Kubernetes cluster info...")
        
        try:
            summary = self.k8s_client.get_cluster_summary()
            return {
                "status": "ok",
                "data": summary
            }
        except Exception as e:
            logger.error(f"Failed to collect cluster info: {e}")
            return {
                "status": "error",
                "error": str(e),
                "data": {}
            }
    
    def collect_network_info(self) -> dict:
        """
        Collect ONOS network topology information.
        
        Returns:
            Dictionary with devices, links, and hosts data
        """
        logger.info("Collecting ONOS network info...")
        
        try:
            summary = self.onos_client.get_topology_summary()
            return {
                "status": "ok",
                "data": summary
            }
        except Exception as e:
            logger.error(f"Failed to collect network info: {e}")
            return {
                "status": "error",
                "error": str(e),
                "data": {}
            }
    
    def collect_monitoring_data(self) -> dict:
        """
        Collect sFlow-RT monitoring metrics.
        
        Returns:
            Dictionary with CPU, memory, and network traffic data
        """
        logger.info("Collecting sFlow-RT monitoring data...")
        
        try:
            summary = self.sflow_client.get_monitoring_summary()
            return {
                "status": "ok",
                "data": summary
            }
        except Exception as e:
            logger.error(f"Failed to collect monitoring data: {e}")
            return {
                "status": "error",
                "error": str(e),
                "data": {}
            }
    
    def collect_all(self) -> dict:
        """
        Collect all system data for LLM analysis.
        
        This is the main method called when a violation is detected.
        It gathers data from all sources and formats it for the LLM.
        
        Returns:
            Complete system state dictionary ready for LLM prompt
        """
        logger.info("Collecting all system data...")
        
        # Collect from all sources
        cluster_info = self.collect_cluster_info()
        network_info = self.collect_network_info()
        monitoring_data = self.collect_monitoring_data()
        response_time_summary = self.get_response_time_summary()
        
        # Build the complete data structure
        # This format matches what the paper describes in Figure 2(a)
        system_state = {
            "timestamp": datetime.now().isoformat(),
            "cluster_info": cluster_info,
            "network_info": network_info,
            "monitoring_data": monitoring_data,
            "response_times": response_time_summary
        }
        
        logger.info("Data collection complete")
        return system_state
    
    def get_health_status(self) -> dict:
        """
        Check health of all data sources.
        
        Returns:
            Dictionary with health status of each component
        """
        return {
            "kubernetes": self.k8s_client.is_healthy(),
            "onos": self.onos_client.is_healthy(),
            "sflow_rt": self.sflow_client.is_healthy()
        }
    
    def format_for_llm(self, system_state: dict, violation_type: str) -> str:
        """
        Format system state as a human-readable string for LLM context.
        
        Args:
            system_state: Complete system state from collect_all()
            violation_type: Type of violation ("UPPER" or "LOWER")
            
        Returns:
            Formatted string describing the system state
        """
        cluster = system_state.get("cluster_info", {}).get("data", {})
        network = system_state.get("network_info", {}).get("data", {})
        monitoring = system_state.get("monitoring_data", {}).get("data", {})
        rt_summary = system_state.get("response_times", {})
        
        # Build readable summary
        lines = []
        lines.append("=== CURRENT SYSTEM STATE ===")
        lines.append("")
        
        # Violation info
        lines.append(f"VIOLATION TYPE: {violation_type}")
        if rt_summary.get("avg"):
            lines.append(f"Average Response Time: {rt_summary['avg']}s")
            lines.append(f"Min: {rt_summary['min']}s, Max: {rt_summary['max']}s")
        lines.append("")
        
        # Cluster info
        lines.append("--- KUBERNETES CLUSTER ---")
        nodes = cluster.get("nodes", {}).get("list", [])
        for node in nodes:
            lines.append(f"Node: {node.get('name')} - Status: {node.get('status')}")
        
        lines.append("")
        pods = cluster.get("pods", {}).get("list", [])
        lines.append(f"Microservice Pods ({len(pods)}):")
        for pod in pods:
            lines.append(f"  - {pod.get('name')} on {pod.get('node')} [{pod.get('status')}]")
            lines.append(f"    CPU limit: {pod.get('cpu_limit')}, Memory limit: {pod.get('memory_limit')}")
        
        lines.append("")
        deployments = cluster.get("deployments", {}).get("list", [])
        lines.append(f"Deployments ({len(deployments)}):")
        for dep in deployments:
            lines.append(f"  - {dep.get('name')}: {dep.get('replicas_ready')}/{dep.get('replicas_desired')} ready")
        
        lines.append("")
        
        # Network info
        lines.append("--- SDN NETWORK ---")
        devices = network.get("devices", {})
        lines.append(f"Switches: {devices.get('count', 0)}")
        links = network.get("links", {})
        lines.append(f"Links: {links.get('count', 0)}")
        hosts = network.get("hosts", {})
        lines.append(f"Hosts: {hosts.get('count', 0)}")
        
        lines.append("")
        
        # Monitoring data
        lines.append("--- RESOURCE UTILIZATION ---")
        cpu_data = monitoring.get("cpu_utilization", [])
        if cpu_data:
            lines.append("CPU Utilization:")
            for item in cpu_data[:5]:
                lines.append(f"  - {item.get('agent')}: {item.get('cpu_percent', 0):.1f}%")
        
        memory_data = monitoring.get("memory_utilization", [])
        if memory_data:
            lines.append("Memory Utilization:")
            for item in memory_data[:5]:
                lines.append(f"  - {item.get('agent')}: {item.get('memory_percent', 0):.1f}%")
        
        lines.append("")
        lines.append("=== END SYSTEM STATE ===")
        
        return "\n".join(lines)
    
    # ===== NEW COMPACT FORMATTING METHODS FOR CLEANER LLM PROMPT =====
    
    def format_monitoring_compact(self, system_state: dict) -> str:
        """
        Format monitoring data in a compact per-node format.
        
        Args:
            system_state: Complete system state from collect_all()
            
        Returns:
            Compact string like "worker1: CPU 80%, Mem 65% | worker2: CPU 70%, Mem 55%"
        """
        monitoring = system_state.get("monitoring_data", {}).get("data", {})
        cpu_data = monitoring.get("cpu_utilization", [])
        memory_data = monitoring.get("memory_utilization", [])
        
        # Build a dict of node -> {cpu, memory}
        node_metrics = {}
        
        for item in cpu_data:
            agent = item.get("agent") or "unknown"  # Handle None values
            # Extract node name from agent (could be IP or hostname)
            node_name = self._extract_node_name(agent)
            if node_name not in node_metrics:
                node_metrics[node_name] = {"cpu": 0, "memory": 0}
            node_metrics[node_name]["cpu"] = item.get("cpu_percent", 0) or 0
        
        for item in memory_data:
            agent = item.get("agent") or "unknown"  # Handle None values
            node_name = self._extract_node_name(agent)
            if node_name not in node_metrics:
                node_metrics[node_name] = {"cpu": 0, "memory": 0}
            node_metrics[node_name]["memory"] = item.get("memory_percent", 0) or 0
        
        if not node_metrics:
            return "No monitoring data available"
        
        # Format as compact string
        parts = []
        for node, metrics in sorted(node_metrics.items()):
            parts.append(f"{node}: CPU {metrics['cpu']:.0f}%, Mem {metrics['memory']:.0f}%")
        
        return " | ".join(parts)
    
    def _extract_node_name(self, agent: str) -> str:
        """
        Extract a readable node name from agent identifier.
        
        Args:
            agent: Agent identifier (IP or hostname)
            
        Returns:
            Readable node name
        """
        # Handle None or empty agent
        if agent is None or agent == "":
            return "unknown"
        
        # Map known IPs to node names (from your setup)
        ip_to_name = {
            "10.0.0.100": "master",
            "10.0.0.101": "worker1",
            "10.0.0.102": "worker2",
            "10.132.0.7": "master",
            "10.132.0.8": "worker1", 
            "10.132.0.9": "worker2"
        }
        
        # Check if agent is a known IP
        if agent in ip_to_name:
            return ip_to_name[agent]
        
        # If it looks like an IP, return last octet
        if agent.count(".") == 3:
            return f"node-{agent.split('.')[-1]}"
        
        # Otherwise return as-is
        return agent
    
    def format_deployments_compact(self, system_state: dict) -> str:
        """
        Format deployments in a compact format showing replicas and node.
        
        Args:
            system_state: Complete system state from collect_all()
            
        Returns:
            Compact multi-line string with deployment info
        """
        cluster = system_state.get("cluster_info", {}).get("data", {})
        deployments = cluster.get("deployments", {}).get("list", [])
        pods = cluster.get("pods", {}).get("list", [])
        
        if not deployments:
            return "No deployments found"
        
        # Map deployment to nodes where its pods run
        deployment_nodes = {}
        for pod in pods:
            pod_name = pod.get("name", "")
            node = pod.get("node", "unknown")
            
            # Find which deployment this pod belongs to
            for dep in deployments:
                dep_name = dep.get("name", "")
                # Pod name typically starts with deployment name
                if pod_name.startswith(dep_name.replace("-deployment", "")):
                    if dep_name not in deployment_nodes:
                        deployment_nodes[dep_name] = []
                    # Only add non-None nodes
                    if node and node not in deployment_nodes[dep_name]:
                        deployment_nodes[dep_name].append(node)
        
        # Format each deployment
        lines = []
        for dep in deployments:
            name = dep.get("name", "unknown")
            replicas = dep.get("replicas_ready", 0)
            desired = dep.get("replicas_desired", 0)
            nodes = deployment_nodes.get(name, [])
            # Handle empty nodes list
            nodes_str = ", ".join(nodes) if nodes else "pending"
            lines.append(f"- {name}: {replicas}/{desired} replicas ({nodes_str})")
        
        return "\n".join(lines)
    
    def format_nodes_compact(self, system_state: dict) -> str:
        """
        Format available nodes in a compact format.
        
        Args:
            system_state: Complete system state from collect_all()
            
        Returns:
            Compact string listing available nodes
        """
        cluster = system_state.get("cluster_info", {}).get("data", {})
        nodes = cluster.get("nodes", {}).get("list", [])
        
        if not nodes:
            return "No nodes found"
        
        # Filter to ready worker nodes only
        worker_nodes = []
        for node in nodes:
            name = node.get("name", "unknown")
            status = node.get("status", "unknown")
            role = node.get("role", "")
            
            # Skip master/control-plane nodes for placement
            if "master" in name.lower() or "control-plane" in role.lower():
                continue
            
            if status.lower() == "ready":
                worker_nodes.append(name)
        
        if not worker_nodes:
            return "No worker nodes available"
        
        return ", ".join(worker_nodes)
    
    def format_network_compact(self, system_state: dict) -> str:
        """
        Format network info in a compact format.
        
        Args:
            system_state: Complete system state from collect_all()
            
        Returns:
            Compact string with switch and link info
        """
        network = system_state.get("network_info", {}).get("data", {})
        devices = network.get("devices", {})
        links = network.get("links", {})
        
        switch_count = devices.get("count", 0)
        link_count = links.get("count", 0)
        
        # Get switch names if available
        switch_list = devices.get("list", [])
        switch_names = [s.get("id", "unknown") for s in switch_list[:6]]  # Max 6 switches
        
        if switch_names:
            return f"Switches: {', '.join(switch_names)} | Links: {link_count}"
        else:
            return f"Switches: {switch_count} | Links: {link_count}"

    # ===== STRUCTURED STATE FOR 8-MESSAGE LLM FORMAT =====

    def compute_critical_path(self, system_state: dict) -> list:
        """
        Derive the application critical path by ordering deployments
        by sFlow traffic volume (bytes_in), highest first.
        
        Args:
            system_state: Complete system state from collect_all()
            
        Returns:
            Ordered list of deployment names, e.g.
            ["microservice3-deployment", "microservice1-deployment", ...]
        """
        monitoring = system_state.get("monitoring_data", {}).get("data", {})
        pod_metrics = monitoring.get("pod_metrics", [])
        
        # Aggregate bytes_in per deployment (sum across replicas)
        dep_traffic = {}
        for pm in pod_metrics:
            dep = pm.get("deployment", "")
            if dep:
                dep_traffic[dep] = dep_traffic.get(dep, 0) + pm.get("bytes_in", 0)
        
        if not dep_traffic:
            # Fallback: return deployments from cluster info in config order
            cluster = system_state.get("cluster_info", {}).get("data", {})
            deployments = cluster.get("deployments", {}).get("list", [])
            return [d.get("name", "") for d in deployments if d.get("name")]
        
        # Sort by traffic volume descending
        sorted_deps = sorted(dep_traffic.items(), key=lambda x: x[1], reverse=True)
        return [name for name, _ in sorted_deps]

    def build_structured_state(
        self,
        system_state: dict,
        violation_type: str,
        ema_rt: float,
        threshold: float
    ) -> dict:
        """
        Build the structured JSON state matching the 8-message LLM input format.
        
        This produces the exact structure used in the test suite:
        {intent, application, services, nodes, network}
        
        Args:
            system_state: Complete system state from collect_all()
            violation_type: "UPPER_THRESHOLD_EXCEEDED" or "LOWER_THRESHOLD_EXCEEDED"
            ema_rt: Current EMA response time
            threshold: The threshold that was exceeded
            
        Returns:
            Structured dict ready for the 8-message LLM conversation
        """
        cluster = system_state.get("cluster_info", {}).get("data", {})
        monitoring = system_state.get("monitoring_data", {}).get("data", {})
        
        # --- Intent ---
        if violation_type == "UPPER_THRESHOLD_EXCEEDED":
            status = "upper_violation"
        else:
            status = "lower_violation"
        
        intent = {
            "status": status,
            "metric": "response_time_seconds",
            "threshold": threshold,
            "observed": round(ema_rt, 2)
        }
        
        # --- Application (critical path) ---
        critical_path = self.compute_critical_path(system_state)
        application = {"critical_path": critical_path}
        
        # --- Services ---
        services = self._build_services_list(cluster, monitoring)
        
        # --- Nodes ---
        nodes = self._build_nodes_list(monitoring)
        
        # --- Network ---
        network = self._build_network_data()
        
        return {
            "intent": intent,
            "application": application,
            "services": services,
            "nodes": nodes,
            "network": network
        }

    def _build_services_list(self, cluster: dict, monitoring: dict) -> list:
        """
        Build the services list with per-deployment metrics.
        
        Computes cpu_util_pct from sFlow pod metrics (averaged across replicas),
        cpu_max_reached and replica_max_reached from config limits.
        """
        deployments = cluster.get("deployments", {}).get("list", [])
        pods = cluster.get("pods", {}).get("list", [])
        pod_metrics = monitoring.get("pod_metrics", [])
        
        # Map deployment -> pod sFlow metrics
        dep_sflow = {}
        for pm in pod_metrics:
            dep = pm.get("deployment", "")
            if dep:
                dep_sflow.setdefault(dep, []).append(pm)
        
        # Map deployment -> nodes from pod placement
        dep_nodes = {}
        for pod in pods:
            pod_name = pod.get("name", "")
            node = pod.get("node", "")
            if not node or node == "None":
                continue
            for d in deployments:
                dep_name = d.get("name", "")
                if dep_name and pod_name.startswith(dep_name):
                    dep_nodes.setdefault(dep_name, [])
                    if node not in dep_nodes[dep_name]:
                        dep_nodes[dep_name].append(node)
        
        services = []
        for d in deployments:
            name = d.get("name", "")
            if not name:
                continue
            
            replicas = d.get("replicas_ready", d.get("replicas_desired", 1))
            cpu_limit_str = str(d.get("cpu_limit", "300m")).replace("m", "").strip()
            try:
                cpu_limit_m = int(cpu_limit_str)
            except (ValueError, TypeError):
                cpu_limit_m = 300
            
            # Average CPU utilization across replicas from sFlow
            sflow_data = dep_sflow.get(name, [])
            if sflow_data:
                cpu_util_pct = sum(p.get("cpu_percent", 0) for p in sflow_data) / len(sflow_data)
            else:
                cpu_util_pct = 0.0
            
            # Determine limits from config
            dep_config = self._get_deployment_config(name)
            max_replicas = dep_config.get("max_replicas", 5)
            cpu_max = cpu_limit_m >= 1000
            replica_max = replicas >= max_replicas
            
            # Primary node (first node where pods are running)
            nodes_list = dep_nodes.get(name, ["unknown"])
            node = nodes_list[0] if nodes_list else "unknown"
            
            services.append({
                "name": name,
                "cpu_util_pct": round(cpu_util_pct, 1),
                "cpu_limit_m": cpu_limit_m,
                "replicas": replicas,
                "cpu_max_reached": cpu_max,
                "replica_max_reached": replica_max,
                "node": node
            })
        
        return services

    def _build_nodes_list(self, monitoring: dict) -> list:
        """Build the nodes list with per-node metrics from sFlow."""
        node_metrics = monitoring.get("node_metrics", {})
        
        nodes = []
        for name in sorted(node_metrics.keys()):
            m = node_metrics[name]
            nodes.append({
                "name": name,
                "cpu_util_pct": round(m.get("cpu", 0), 1),
                "mem_util_pct": round(m.get("memory", 0), 1),
                "load": round(m.get("load", 0), 2)
            })
        
        return nodes

    def _build_network_data(self) -> dict:
        """
        Build network data by combining sFlow SDN bandwidth and ONOS link stats
        into a composite congestion score and hot_links list.
        """
        # Get sFlow-based congestion score
        sflow_congestion = self.sflow_client.compute_congestion_score()
        
        # Get ONOS link utilization for additional hot_links
        onos_links = self.onos_client.get_link_utilization()
        
        # Merge hot_links from both sources
        hot_links = list(sflow_congestion.get("hot_links", []))
        for link in onos_links:
            # Only add ONOS links that aren't already represented
            link_name = link.get("name", "")
            if link_name and not any(h["name"] == link_name for h in hot_links):
                # Estimate utilization from bytes (rough heuristic)
                max_bw = self.sflow_client.sflow_config.get("max_bandwidth_bps", 1_000_000_000)
                bytes_total = link.get("bytes_in", 0) + link.get("bytes_out", 0)
                util_pct = (bytes_total * 8) / max(max_bw, 1) * 100
                if util_pct > 50:
                    hot_links.append({"name": link_name, "utilization_pct": round(util_pct, 1)})
        
        hot_links.sort(key=lambda x: x.get("utilization_pct", 0), reverse=True)
        
        return {
            "congestion_score": sflow_congestion.get("congestion_score", 0.0),
            "hot_links": hot_links
        }

    def _get_deployment_config(self, dep_name: str) -> dict:
        """Get deployment-specific config (limits etc) from config.yaml."""
        for dep in self.config.get("kubernetes", {}).get("deployments", []):
            if dep.get("name", "").lower() == dep_name.lower():
                return dep
        return {"min_replicas": 1, "max_replicas": 5}