"""
sFlow-RT Real-Time Analytics Client

This module provides functions to interact with the sFlow-RT REST API.
sFlow-RT collects real-time telemetry from network devices and hosts.

Supports three types of agents:
- Node agents (host sFlow): CPU, memory, load per cluster node
- Pod agents (sidecar sFlow): CPU, memory, traffic per pod/container
- SDN overlay agents: Network bandwidth per node's SDN interface

API Documentation: https://sflow-rt.com/reference.php
"""

import requests
import logging
import re

logger = logging.getLogger(__name__)


class SFlowRTClient:
    """Client for interacting with sFlow-RT analytics engine."""
    
    def __init__(self, base_url: str, sflow_config: dict = None):
        """
        Initialize sFlow-RT client.
        
        Args:
            base_url: sFlow-RT API base URL (e.g., http://localhost:8008)
            sflow_config: sFlow configuration from config.yaml containing
                          node IPs, SDN overlay IPs, and pod CIDR prefix
        """
        self.base_url = base_url.rstrip('/')
        self.sflow_config = sflow_config or {}
        
        # Build IP-to-name mappings from config
        self.node_ip_to_name = {}
        for name, ip in self.sflow_config.get("nodes", {}).items():
            self.node_ip_to_name[ip] = name
        
        self.sdn_ip_to_name = {}
        for name, ip in self.sflow_config.get("sdn_overlay", {}).items():
            self.sdn_ip_to_name[ip] = name
        
        self.pod_cidr = self.sflow_config.get("pod_cidr", "10.42.")
    
    def _get(self, endpoint: str):
        """Make GET request to sFlow-RT API."""
        url = f"{self.base_url}/{endpoint}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"sFlow-RT API error for {endpoint}: {e}")
            return {}
    
    # ===== AGENT DISCOVERY =====
    
    def get_agents(self) -> dict:
        """Get all sFlow agents reporting to sFlow-RT."""
        return self._get("agents/json")
    
    def discover_pod_agents(self) -> list:
        """
        Discover pod sFlow agents by filtering agents with pod CIDR prefix.
        
        Returns:
            List of pod agent IPs (e.g., ["10.42.1.12", "10.42.3.17"])
        """
        agents = self.get_agents()
        if not isinstance(agents, dict):
            return []
        
        pod_agents = [ip for ip in agents.keys() if ip.startswith(self.pod_cidr)]
        logger.debug(f"Discovered {len(pod_agents)} pod agents: {pod_agents}")
        return pod_agents
    
    # ===== NODE METRICS =====
    
    def get_node_metrics(self) -> dict:
        """
        Get CPU, memory, and load metrics for all configured cluster nodes.
        
        Returns:
            Dict like:
            {
                "master": {"cpu": 12.8, "memory": 7.9, "load": 0.30},
                "worker1": {"cpu": 7.1, "memory": 8.1, "load": 0.44},
                ...
            }
        """
        node_metrics = {}
        
        for name, ip in self.sflow_config.get("nodes", {}).items():
            metrics = {"cpu": 0, "memory": 0, "load": 0}
            
            # CPU utilization
            data = self._get(f"metric/{ip}/cpu_utilization/json")
            if isinstance(data, list) and data:
                metrics["cpu"] = data[0].get("metricValue", 0) or 0
            
            # Memory utilization
            data = self._get(f"metric/{ip}/mem_utilization/json")
            if isinstance(data, list) and data:
                metrics["memory"] = data[0].get("metricValue", 0) or 0
            
            # Load average (1 min)
            data = self._get(f"metric/{ip}/load_one/json")
            if isinstance(data, list) and data:
                metrics["load"] = data[0].get("metricValue", 0) or 0
            
            node_metrics[name] = metrics
        
        return node_metrics
    
    # ===== POD METRICS =====
    
    def get_pod_metrics(self) -> list:
        """
        Get metrics for all discovered pod agents via sFlow sidecars.
        
        Auto-discovers pod IPs from /agents/json, then queries each one
        for CPU, memory, bytes in/out, and hostname (deployment name).
        
        Returns:
            List of dicts like:
            [
                {
                    "pod_ip": "10.42.3.17",
                    "hostname": "microservice3-deployment-cc65cdd94-bxzgc",
                    "deployment": "microservice3-deployment",
                    "cpu_percent": 33.6,
                    "mem_used_bytes": 683696128,
                    "mem_total_bytes": 8324816896,
                    "bytes_in": 57994.3,
                    "bytes_out": 11497.1
                },
                ...
            ]
        """
        pod_agents = self.discover_pod_agents()
        pod_metrics = []
        
        for pod_ip in pod_agents:
            # Get all metrics for this pod agent in one call
            all_metrics = self._get(f"metric/{pod_ip}/json")
            if not isinstance(all_metrics, dict):
                continue
            
            # Extract hostname (pod name)
            hostname = all_metrics.get("2.1.host_name", "")
            if not hostname:
                continue
            
            # Derive deployment name from pod hostname
            # Pod name format: "microservice3-deployment-cc65cdd94-bxzgc"
            # We need to strip the last two hash segments
            deployment = self._pod_name_to_deployment(hostname)
            if not deployment:
                continue
            
            # Get CPU from dedicated endpoint (more reliable)
            cpu_percent = 0
            cpu_data = self._get(f"metric/{pod_ip}/cpu_utilization/json")
            if isinstance(cpu_data, list) and cpu_data:
                cpu_percent = cpu_data[0].get("metricValue", 0) or 0
            
            pod_metrics.append({
                "pod_ip": pod_ip,
                "hostname": hostname,
                "deployment": deployment,
                "cpu_percent": round(cpu_percent, 1),
                "mem_used_bytes": all_metrics.get("2.1.mem_used", 0) or 0,
                "mem_total_bytes": all_metrics.get("2.1.mem_total", 0) or 0,
                "bytes_in": all_metrics.get("2.1.bytes_in", 0) or 0,
                "bytes_out": all_metrics.get("2.1.bytes_out", 0) or 0,
            })
        
        return pod_metrics
    
    @staticmethod
    def _pod_name_to_deployment(pod_name: str) -> str:
        """
        Extract deployment name from a pod hostname.
        
        Handles formats like:
        - "microservice3-deployment-cc65cdd94-bxzgc" -> "microservice3-deployment"
        - "db-deployment-9c7d67545-l67ld" -> "db-deployment"
        
        Returns:
            Deployment name or empty string if can't parse
        """
        # Match pattern: <name>-<hash>-<hash>
        # Deployment name is everything before the last two dash-separated segments
        match = re.match(r'^(.+-deployment)-[a-z0-9]+-[a-z0-9]+$', pod_name)
        if match:
            return match.group(1)
        
        # Fallback: try splitting on last two segments
        parts = pod_name.rsplit('-', 2)
        if len(parts) >= 3:
            return parts[0]
        
        return ""
    
    # ===== SDN BANDWIDTH =====
    
    def get_sdn_bandwidth(self) -> dict:
        """
        Get SDN overlay network bandwidth for each node.
        
        Returns:
            Dict like:
            {
                "master": {"bytes_in": 34175.3, "bytes_out": 28000.5},
                "worker1": {"bytes_in": 173006.5, "bytes_out": 95000.2},
                ...
            }
        """
        bandwidth = {}
        
        for name, ip in self.sflow_config.get("sdn_overlay", {}).items():
            bw = {"bytes_in": 0, "bytes_out": 0}
            
            data_in = self._get(f"metric/{ip}/ifinoctets/json")
            if isinstance(data_in, list) and data_in:
                bw["bytes_in"] = data_in[0].get("metricValue", 0) or 0
            
            data_out = self._get(f"metric/{ip}/ifoutoctets/json")
            if isinstance(data_out, list) and data_out:
                bw["bytes_out"] = data_out[0].get("metricValue", 0) or 0
            
            bandwidth[name] = bw
        
        return bandwidth
    
    # ===== COMBINED SUMMARY =====
    
    def get_monitoring_summary(self) -> dict:
        """
        Get complete monitoring summary from all sFlow sources.
        
        Returns:
            Dictionary with node metrics, pod metrics, and SDN bandwidth.
        """
        node_metrics = self.get_node_metrics()
        pod_metrics = self.get_pod_metrics()
        sdn_bandwidth = self.get_sdn_bandwidth()
        
        return {
            "node_metrics": node_metrics,
            "pod_metrics": pod_metrics,
            "sdn_bandwidth": sdn_bandwidth,
            # Keep legacy fields for backward compatibility
            "cpu_utilization": [
                {"agent": ip, "cpu_percent": m["cpu"]}
                for ip, m in [(self.sflow_config.get("nodes", {}).get(n, ""), v) 
                              for n, v in node_metrics.items()]
                if ip
            ],
            "memory_utilization": [
                {"agent": ip, "memory_percent": m["memory"]}
                for ip, m in [(self.sflow_config.get("nodes", {}).get(n, ""), v) 
                              for n, v in node_metrics.items()]
                if ip
            ],
        }
    
    def compute_congestion_score(self) -> dict:
        """
        Compute a normalized network congestion score (0-1) from SDN bandwidth data.
        
        Combines sFlow SDN overlay bandwidth across all nodes and normalizes
        against a configured max_bandwidth_bps value.
        
        Returns:
            Dict like:
            {
                "congestion_score": 0.72,
                "hot_links": [{"name": "worker1-in", "utilization_pct": 85.0}, ...]
            }
        """
        sdn_bandwidth = self.get_sdn_bandwidth()
        max_bw = self.sflow_config.get("max_bandwidth_bps", 1_000_000_000)  # 1 Gbps default
        
        if not sdn_bandwidth or max_bw <= 0:
            return {"congestion_score": 0.0, "hot_links": []}
        
        hot_links = []
        max_utilization = 0.0
        
        for node_name, bw in sdn_bandwidth.items():
            bytes_in = bw.get("bytes_in", 0)
            bytes_out = bw.get("bytes_out", 0)
            
            # Convert bytes/s to bits/s for comparison with max_bandwidth_bps
            in_utilization = (bytes_in * 8) / max_bw * 100
            out_utilization = (bytes_out * 8) / max_bw * 100
            peak = max(in_utilization, out_utilization)
            
            if peak > max_utilization:
                max_utilization = peak
            
            # Report links with utilization above 50%
            if in_utilization > 50:
                hot_links.append({"name": f"{node_name}-in", "utilization_pct": round(in_utilization, 1)})
            if out_utilization > 50:
                hot_links.append({"name": f"{node_name}-out", "utilization_pct": round(out_utilization, 1)})
        
        # Normalize to 0-1 (cap at 1.0)
        congestion_score = min(max_utilization / 100.0, 1.0)
        
        # Sort hot_links by utilization descending
        hot_links.sort(key=lambda x: x["utilization_pct"], reverse=True)
        
        return {
            "congestion_score": round(congestion_score, 2),
            "hot_links": hot_links
        }

    def is_healthy(self) -> bool:
        """Check if sFlow-RT is responding."""
        try:
            url = f"{self.base_url}/version"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False