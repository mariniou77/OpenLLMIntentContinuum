"""
ONOS SDN Controller Client

This module provides functions to interact with the ONOS REST API.
ONOS manages the SDN network topology and flow rules.

API Documentation: https://wiki.onosproject.org/display/ONOS/REST+API
"""

import requests
from requests.auth import HTTPBasicAuth
import logging

logger = logging.getLogger(__name__)


class ONOSClient:
    """Client for interacting with ONOS SDN Controller."""
    
    def __init__(self, base_url: str, username: str, password: str):
        """
        Initialize ONOS client.
        
        Args:
            base_url: ONOS API base URL (e.g., http://localhost:8181)
            username: ONOS username (default: onos)
            password: ONOS password (default: rocks)
        """
        self.base_url = base_url.rstrip('/')
        self.auth = HTTPBasicAuth(username, password)
        self.headers = {"Accept": "application/json"}
    
    def _get(self, endpoint: str) -> dict:
        """Make GET request to ONOS API."""
        url = f"{self.base_url}/onos/v1/{endpoint}"
        try:
            response = requests.get(url, auth=self.auth, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"ONOS API error for {endpoint}: {e}")
            return {}
    
    def get_devices(self) -> list:
        """
        Get all network devices (switches).
        
        Returns:
            List of switch dictionaries with id, type, available status, etc.
        """
        data = self._get("devices")
        return data.get("devices", [])
    
    def get_links(self) -> list:
        """
        Get all network links between switches.
        
        Returns:
            List of link dictionaries with src/dst switch and port info.
        """
        data = self._get("links")
        return data.get("links", [])
    
    def get_hosts(self) -> list:
        """
        Get all hosts connected to the network.
        
        Returns:
            List of host dictionaries with MAC, IP, location (switch/port).
        """
        data = self._get("hosts")
        return data.get("hosts", [])
    
    def get_flows(self, device_id: str = None) -> list:
        """
        Get flow rules from switches.
        
        Args:
            device_id: Optional specific device ID. If None, gets all flows.
            
        Returns:
            List of flow rule dictionaries.
        """
        endpoint = f"flows/{device_id}" if device_id else "flows"
        data = self._get(endpoint)
        return data.get("flows", [])
    
    def get_topology_summary(self) -> dict:
        """
        Get a summary of the network topology for LLM analysis.
        
        Returns:
            Dictionary with devices, links, and hosts counts and details.
        """
        devices = self.get_devices()
        links = self.get_links()
        hosts = self.get_hosts()
        
        return {
            "devices": {
                "count": len(devices),
                "list": [{"id": d.get("id"), "available": d.get("available")} for d in devices]
            },
            "links": {
                "count": len(links),
                "list": [{"src": l.get("src", {}).get("device"), 
                         "dst": l.get("dst", {}).get("device")} for l in links]
            },
            "hosts": {
                "count": len(hosts),
                "list": [{"mac": h.get("mac"), 
                         "ips": h.get("ipAddresses", []),
                         "location": h.get("locations", [{}])[0].get("elementId")} for h in hosts]
            }
        }
    
    def is_healthy(self) -> bool:
        """Check if ONOS is responding."""
        try:
            devices = self.get_devices()
            return True
        except Exception:
            return False