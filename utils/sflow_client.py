"""
sFlow-RT Real-Time Analytics Client

This module provides functions to interact with the sFlow-RT REST API.
sFlow-RT collects real-time telemetry from network devices and hosts.

API Documentation: https://sflow-rt.com/reference.php
"""

import requests
import logging

logger = logging.getLogger(__name__)


class SFlowRTClient:
    """Client for interacting with sFlow-RT analytics engine."""
    
    def __init__(self, base_url: str):
        """
        Initialize sFlow-RT client.
        
        Args:
            base_url: sFlow-RT API base URL (e.g., http://localhost:8008)
        """
        self.base_url = base_url.rstrip('/')
    
    def _get(self, endpoint: str) -> dict | list:
        """Make GET request to sFlow-RT API."""
        url = f"{self.base_url}/{endpoint}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"sFlow-RT API error for {endpoint}: {e}")
            return {}
    
    def get_agents(self) -> dict:
        """
        Get all sFlow agents (hosts sending metrics).
        
        Returns:
            Dictionary of agent IPs and their metadata.
        """
        return self._get("agents/json")
    
    def get_metric(self, agent: str, metric_name: str) -> list:
        """
        Get a specific metric for an agent.
        
        Args:
            agent: Agent IP or "ALL" for all agents
            metric_name: Metric name (e.g., ifinoctets, cpu_utilization)
            
        Returns:
            List of metric values.
        """
        return self._get(f"metric/{agent}/{metric_name}/json")
    
    def get_interface_traffic(self) -> list:
        """
        Get network interface input traffic (bytes/sec) for all agents.
        
        Returns:
            List of dictionaries with agent, interface, and traffic rate.
        """
        data = self._get("metric/ALL/ifinoctets/json")
        if isinstance(data, list):
            return [{"agent": item.get("agent"),
                    "interface": item.get("dataSource"),
                    "bytes_per_sec": item.get("metricValue", 0)} for item in data]
        return []
    
    def get_cpu_utilization(self) -> list:
        """
        Get CPU utilization for all agents.
        
        Returns:
            List of dictionaries with agent and CPU percentage.
        """
        data = self._get("metric/ALL/cpu_utilization/json")
        if isinstance(data, list):
            return [{"agent": item.get("agent"),
                    "cpu_percent": item.get("metricValue", 0)} for item in data]
        return []
    
    def get_memory_utilization(self) -> list:
        """
        Get memory utilization for all agents.
        
        Returns:
            List of dictionaries with agent and memory percentage.
        """
        data = self._get("metric/ALL/mem_utilization/json")
        if isinstance(data, list):
            return [{"agent": item.get("agent"),
                    "memory_percent": item.get("metricValue", 0)} for item in data]
        return []
    
    def get_monitoring_summary(self) -> dict:
        """
        Get a summary of all monitoring metrics for LLM analysis.
        
        Returns:
            Dictionary with CPU, memory, and network metrics.
        """
        agents = self.get_agents()
        cpu = self.get_cpu_utilization()
        memory = self.get_memory_utilization()
        traffic = self.get_interface_traffic()
        
        return {
            "agents_count": len(agents) if isinstance(agents, dict) else 0,
            "cpu_utilization": cpu[:10] if cpu else [],  # Limit to 10 for LLM context
            "memory_utilization": memory[:10] if memory else [],
            "interface_traffic": traffic[:10] if traffic else []
        }
    
    def is_healthy(self) -> bool:
        """Check if sFlow-RT is responding."""
        try:
            url = f"{self.base_url}/version"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False