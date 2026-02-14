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
        
    def add_point_to_point_intent(self, ingress_device: str, egress_device: str, path: list = None) -> dict:
        """
        Add a point-to-point intent in ONOS.
        
        This creates a connectivity intent between two switches.
        ONOS will compute the path automatically if not specified.
        
        Args:
            ingress_device: Source switch ID (e.g., "of:0000000000000001")
            egress_device: Destination switch ID
            path: Optional list of switches for explicit path
            
        Returns:
            Dictionary with success status
        """
        # ONOS uses intents for high-level connectivity
        # The PointToPointIntent connects two specific ports
        
        intent_data = {
            "type": "PointToPointIntent",
            "appId": "org.onosproject.cli",
            "ingressPoint": {
                "device": ingress_device,
                "port": "1"
            },
            "egressPoint": {
                "device": egress_device,
                "port": "1"
            }
        }
        
        # If path is specified, we use it as waypoints
        if path and len(path) > 2:
            # ONOS doesn't directly support waypoints in basic intents
            # We would need to use more advanced intent types or flow rules
            # For now, we let ONOS compute the path
            logger.info(f"Path hint provided: {path} (ONOS will compute actual path)")
        
        try:
            response = requests.post(
                f"{self.base_url}/onos/v1/intents",
                auth=self.auth,
                json=intent_data,
                timeout=10
            )
            
            if response.status_code in [200, 201]:
                logger.info(f"Intent created: {ingress_device} -> {egress_device}")
                return {"success": True, "message": "Intent created"}
            else:
                logger.error(f"Failed to create intent: {response.status_code} - {response.text}")
                return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"ONOS request failed: {e}")
            return {"success": False, "error": str(e)}
    
    def delete_intent(self, app_id: str, intent_key: str) -> dict:
        """
        Delete an intent from ONOS.
        
        Args:
            app_id: Application ID that created the intent
            intent_key: Unique key of the intent
            
        Returns:
            Dictionary with success status
        """
        try:
            response = requests.delete(
                f"{self.base_url}/onos/v1/intents/{app_id}/{intent_key}",
                auth=self.auth,
                timeout=10
            )
            
            if response.status_code in [200, 204]:
                return {"success": True}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
                
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}
    
    def get_intents(self) -> list:
        """
        Get all intents from ONOS.
        
        Returns:
            List of intent dictionaries
        """
        try:
            response = requests.get(
                f"{self.base_url}/onos/v1/intents",
                auth=self.auth,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json().get("intents", [])
            else:
                logger.error(f"Failed to get intents: {response.status_code}")
                return []
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get intents: {e}")
            return []