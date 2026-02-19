"""
Decision Maker Module

This module integrates with the LLM (TinyLlama via Ollama) to analyze
system state and recommend actions when SLO violations occur.

Supports 4 action types:
1. horizontal_scaling - Change replica count
2. vertical_scaling - Change CPU/memory limits
3. service_placement - Move pod to different node
4. flow_scheduling - Change network path via ONOS

The LLM analyzes the system state and decides which action to take.
"""

import json
import logging
import re
import requests
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


class DecisionMaker:
    """
    LLM-powered decision maker for intent-based resource management.
    
    This class handles communication with the Ollama API to get
    intelligent recommendations for handling SLO violations.
    """
    
    def __init__(self, config: dict):
        """
        Initialize Decision Maker with configuration.
        
        Args:
            config: Configuration dictionary containing LLM settings
        """
        self.config = config
        self.ollama_url = config["endpoints"]["ollama"]
        self.model = config["llm"]["model"]
        self.temperature = config["llm"]["temperature"]
        self.debug_llm = config.get("debug_llm", False)
        
        # Load prompt template
        self.prompt_template = self._load_prompt_template()
        
        # Intent thresholds for context
        self.upper_threshold = config["intent"]["upper_threshold"]
        self.lower_threshold = config["intent"]["lower_threshold"]
        
        # Enabled actions
        self.actions_enabled = config.get("actions", {})
    
    def _load_prompt_template(self) -> str:
        """Load the prompt template from file."""
        template_path = Path(__file__).parent / "prompts" / "analysis_prompt.txt"
        try:
            with open(template_path, "r") as f:
                return f.read()
        except FileNotFoundError:
            logger.warning(f"Prompt template not found at {template_path}, using default")
            return self._get_default_prompt_template()
    
    def _get_default_prompt_template(self) -> str:
        """Return a default prompt template if file not found."""
        return """You must respond with ONLY a JSON object. No other text.

Problem: {violation_type}
Current response time: {current_rt}s (EMA: {ema_rt}s)
Target range: {lower_threshold}s - {upper_threshold}s

Current System State:
{system_state}

Available Actions:
{available_actions}

Previous decisions:
{history}

CRITICAL INSTRUCTIONS:

1. UNDERSTAND THE PROBLEM:
   - LOWER_THRESHOLD_EXCEEDED means response time is TOO LOW (too fast) - system has TOO MANY resources
   - UPPER_THRESHOLD_EXCEEDED means response time is TOO HIGH (too slow) - system needs MORE resources

2. FOR LOWER_THRESHOLD_EXCEEDED (current problem if applicable):
   - Goal: INCREASE response time to get it back above {lower_threshold}s
   - REDUCE replicas (e.g., from 3 to 2, or 2 to 1)
   - REDUCE CPU/memory limits (e.g., cpu_limit: "200m" instead of "500m")
   - Do NOT increase replicas - that makes response time FASTER (wrong direction!)

3. FOR UPPER_THRESHOLD_EXCEEDED:
   - Goal: DECREASE response time to get it below {upper_threshold}s
   - INCREASE replicas (e.g., from 1 to 2, or 2 to 3)
   - INCREASE CPU/memory limits

4. LEARN FROM PREVIOUS OUTCOMES:
   - If previous action showed "WORSENED", try a DIFFERENT action or opposite approach
   - If previous action showed "NO_CHANGE", try a different deployment or action type
   - If previous action showed "IMPROVED", continue in same direction if still violating
   - Do NOT repeat the exact same action that failed

5. TRY DIFFERENT APPROACHES:
   - If horizontal_scaling didn't help, try vertical_scaling
   - If one deployment didn't help, try a different deployment (microservice1, microservice3, microservice4)
   - Consider which microservice is the bottleneck based on monitoring data

Choose ONE action and respond with ONLY a JSON object matching the format shown above.
"""

    def _get_enabled_actions_description(self) -> str:
        """Get description of enabled actions for the prompt."""
        descriptions = []
        
        if self.actions_enabled.get("horizontal_scaling"):
            descriptions.append(
                '1. horizontal_scaling: Change replica count\n'
                '   JSON format: {"action": "horizontal_scaling", "deployment_name": "name", "replicas": N}'
            )
        
        if self.actions_enabled.get("vertical_scaling"):
            descriptions.append(
                '2. vertical_scaling: Change CPU/memory limits\n'
                '   JSON format: {"action": "vertical_scaling", "deployment_name": "name", "cpu_limit": "500m", "memory_limit": "512Mi"}'
            )
        
        if self.actions_enabled.get("service_placement"):
            descriptions.append(
                '3. service_placement: Move pod to different node\n'
                '   JSON format: {"action": "service_placement", "deployment_name": "name", "target_node": "worker1"}'
            )
        
        if self.actions_enabled.get("flow_scheduling"):
            descriptions.append(
                '4. flow_scheduling: Change network path\n'
                '   JSON format: {"action": "flow_scheduling", "source_switch": "of:...", "destination_switch": "of:..."}'
            )
        
        if not descriptions:
            descriptions.append('No actions enabled')
        
        return '\n'.join(descriptions)

    def _format_system_state(self, cluster_data: dict, network_data: dict, monitoring_data: dict) -> str:
        """Format system state for the prompt."""
        lines = []
        
        # Debug: log what we received
        logger.debug(f"cluster_data keys: {cluster_data.keys() if cluster_data else 'None'}")
        logger.debug(f"network_data keys: {network_data.keys() if network_data else 'None'}")
        
        # Deployments
        deployments = cluster_data.get("deployments", {}).get("list", [])
        if deployments:
            lines.append("Deployments:")
            for d in deployments:
                name = d.get("name", "unknown")
                replicas = d.get("replicas_desired", 0)
                ready = d.get("replicas_ready", 0)
                lines.append(f"  - {name}: {ready}/{replicas} replicas")
        else:
            logger.debug(f"No deployments found. cluster_data: {cluster_data}")
        
        # Nodes
        nodes = cluster_data.get("nodes", {}).get("list", [])
        if nodes:
            lines.append("Nodes:")
            for n in nodes:
                name = n.get("name", "unknown")
                status = n.get("status", "unknown")
                lines.append(f"  - {name}: {status}")
        
        # Network devices
        devices = network_data.get("devices", {}).get("list", [])
        if devices:
            lines.append("Network switches:")
            for d in devices:
                dev_id = d.get("id", "unknown")
                available = d.get("available", False)
                lines.append(f"  - {dev_id}: {'available' if available else 'unavailable'}")
        
        return '\n'.join(lines) if lines else "No system data available"

    def build_prompt(
        self,
        violation_type: str,
        current_rt: float,
        ema_rt: float,
        cluster_data: dict,
        network_data: dict,
        monitoring_data: dict,
        history: str
    ) -> str:
        """
        Build the complete prompt for the LLM.
        
        Args:
            violation_type: Type of violation
            current_rt: Current response time in seconds
            ema_rt: EMA response time in seconds
            cluster_data: Kubernetes cluster state
            network_data: ONOS network state
            monitoring_data: sFlow monitoring metrics
            history: Formatted history string
            
        Returns:
            Complete prompt string
        """
        system_state = self._format_system_state(cluster_data, network_data, monitoring_data)
        available_actions = self._get_enabled_actions_description()
        
        prompt = self.prompt_template.format(
            violation_type=violation_type,
            current_rt=round(current_rt, 2),
            ema_rt=round(ema_rt, 2),
            lower_threshold=self.lower_threshold,
            upper_threshold=self.upper_threshold,
            system_state=system_state,
            available_actions=available_actions,
            history=history if history else "No previous decisions"
        )
        return prompt
    
    def _query_ollama(self, prompt: str) -> Optional[str]:
        """
        Send prompt to Ollama API and get response.
        
        Args:
            prompt: The complete prompt to send
            
        Returns:
            LLM response text or None if failed
        """
        url = f"{self.ollama_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self.temperature,
                "num_predict": 256
            }
        }
        
        # Debug logging
        if self.debug_llm:
            logger.info("=" * 60)
            logger.info("LLM DEBUG - PROMPT:")
            logger.info("=" * 60)
            logger.info(f"\n{prompt}")
            logger.info("=" * 60)
        
        try:
            logger.info(f"Querying Ollama ({self.model})...")
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            
            result = response.json()
            llm_response = result.get("response", "")
            
            if self.debug_llm:
                logger.info("LLM DEBUG - RESPONSE:")
                logger.info("=" * 60)
                logger.info(f"\n{llm_response}")
                logger.info("=" * 60)
            
            return llm_response
            
        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            return None
    
    def _parse_response(self, response_text: str) -> dict:
        """
        Parse LLM response to extract JSON action.
        
        Handles various formats TinyLlama might produce.
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            Parsed action dictionary with 'action' and 'parameters' keys
        """
        if not response_text:
            return self._get_fallback_response("No response from LLM")
        
        # Clean up the response
        cleaned = response_text.strip()
        
        # Try 1: Direct JSON parse
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return self._normalize_response(parsed)
        except json.JSONDecodeError:
            pass
        
        # Try 2: Find JSON object in the response (between { and })
        try:
            start = cleaned.find('{')
            end = cleaned.rfind('}') + 1
            
            if start != -1 and end > start:
                json_str = cleaned[start:end]
                parsed = json.loads(json_str)
                if isinstance(parsed, dict):
                    return self._normalize_response(parsed)
        except json.JSONDecodeError:
            pass
        
        # Try 3: Extract using regex patterns
        try:
            return self._extract_with_regex(response_text)
        except Exception as e:
            logger.warning(f"Regex extraction failed: {e}")
        
        logger.warning(f"Could not parse LLM response: {response_text[:200]}")
        return self._get_fallback_response("Could not parse LLM response")
    
    def _extract_with_regex(self, response_text: str) -> dict:
        """
        Extract action information using regex patterns.
        
        This is a fallback for when JSON parsing fails.
        """
        result = {"action": "none", "parameters": {}}
        
        # Try to find action type
        action_match = re.search(r'"action"\s*:\s*"([^"]+)"', response_text)
        if action_match:
            action = action_match.group(1).lower().strip()
            result["action"] = action
        
        # Extract parameters based on action type
        if "horizontal_scaling" in response_text.lower() or "scale" in response_text.lower():
            dep_match = re.search(r'"deployment_name"\s*:\s*"([^"]+)"', response_text)
            rep_match = re.search(r'"replicas"\s*:\s*(\d+)', response_text)
            
            if dep_match:
                result["action"] = "horizontal_scaling"
                result["parameters"] = {
                    "deployment_name": dep_match.group(1),
                    "replicas": int(rep_match.group(1)) if rep_match else 2
                }
        
        elif "vertical_scaling" in response_text.lower():
            dep_match = re.search(r'"deployment_name"\s*:\s*"([^"]+)"', response_text)
            cpu_match = re.search(r'"cpu_limit"\s*:\s*"([^"]+)"', response_text)
            mem_match = re.search(r'"memory_limit"\s*:\s*"([^"]+)"', response_text)
            
            if dep_match:
                result["action"] = "vertical_scaling"
                result["parameters"] = {
                    "deployment_name": dep_match.group(1),
                    "cpu_limit": cpu_match.group(1) if cpu_match else "500m",
                    "memory_limit": mem_match.group(1) if mem_match else "512Mi"
                }
        
        elif "service_placement" in response_text.lower() or "placement" in response_text.lower():
            dep_match = re.search(r'"deployment_name"\s*:\s*"([^"]+)"', response_text)
            node_match = re.search(r'"target_node"\s*:\s*"([^"]+)"', response_text)
            
            if dep_match and node_match:
                result["action"] = "service_placement"
                result["parameters"] = {
                    "deployment_name": dep_match.group(1),
                    "target_node": node_match.group(1)
                }
        
        elif "flow_scheduling" in response_text.lower() or "reroute" in response_text.lower():
            src_match = re.search(r'"source_switch"\s*:\s*"([^"]+)"', response_text)
            dst_match = re.search(r'"destination_switch"\s*:\s*"([^"]+)"', response_text)
            
            if src_match and dst_match:
                result["action"] = "flow_scheduling"
                result["parameters"] = {
                    "source_switch": src_match.group(1),
                    "destination_switch": dst_match.group(1),
                    "new_path": []
                }
        
        return result
    
    def _normalize_response(self, parsed: dict) -> dict:
        """
        Normalize parsed JSON into standard format.
        
        Handles various field names the LLM might use.
        
        Args:
            parsed: Parsed JSON dictionary
            
        Returns:
            Normalized dictionary with 'action' and 'parameters'
        """
        # Normalize action name
        action = str(parsed.get("action", "none")).lower().strip()
        
        action_mapping = {
            "horizontal_scaling": "horizontal_scaling",
            "horizontalscaling": "horizontal_scaling",
            "scale": "horizontal_scaling",
            "scale_up": "horizontal_scaling",
            "scale_down": "horizontal_scaling",
            "vertical_scaling": "vertical_scaling",
            "verticalscaling": "vertical_scaling",
            "resources": "vertical_scaling",
            "service_placement": "service_placement",
            "serviceplacement": "service_placement",
            "placement": "service_placement",
            "migrate": "service_placement",
            "move": "service_placement",
            "flow_scheduling": "flow_scheduling",
            "flowscheduling": "flow_scheduling",
            "reroute": "flow_scheduling",
            "network": "flow_scheduling",
            "none": "none",
            "no_action": "none"
        }
        
        normalized_action = action_mapping.get(action, "none")
        
        # Check if action is enabled
        if normalized_action != "none" and not self.actions_enabled.get(normalized_action, False):
            logger.warning(f"Action '{normalized_action}' is not enabled, falling back to none")
            return {"action": "none", "parameters": {}}
        
        # Build parameters based on action type
        parameters = {}
        
        if normalized_action == "horizontal_scaling":
            dep_name = parsed.get("deployment_name") or parsed.get("deployment") or parsed.get("name")
            replicas = parsed.get("replicas") or parsed.get("replica_count") or parsed.get("replica") or 2
            
            # Handle non-numeric replicas
            if isinstance(replicas, (list, dict)):
                replicas = 2
            try:
                replicas = int(replicas)
            except (ValueError, TypeError):
                replicas = 2
            
            if dep_name:
                parameters = {
                    "deployment_name": str(dep_name),
                    "replicas": max(1, min(5, replicas))
                }
            else:
                normalized_action = "none"
                
        elif normalized_action == "vertical_scaling":
            dep_name = parsed.get("deployment_name") or parsed.get("deployment") or parsed.get("name")
            cpu = parsed.get("cpu_limit") or parsed.get("cpu") or "500m"
            mem = parsed.get("memory_limit") or parsed.get("memory") or "512Mi"
            
            if dep_name:
                parameters = {
                    "deployment_name": str(dep_name),
                    "cpu_limit": str(cpu),
                    "memory_limit": str(mem)
                }
            else:
                normalized_action = "none"
                
        elif normalized_action == "service_placement":
            dep_name = parsed.get("deployment_name") or parsed.get("deployment") or parsed.get("name")
            target = parsed.get("target_node") or parsed.get("node") or parsed.get("target")
            
            if dep_name and target:
                parameters = {
                    "deployment_name": str(dep_name),
                    "target_node": str(target)
                }
            else:
                normalized_action = "none"
                
        elif normalized_action == "flow_scheduling":
            src = parsed.get("source_switch") or parsed.get("source") or parsed.get("ingress")
            dst = parsed.get("destination_switch") or parsed.get("destination") or parsed.get("egress")
            path = parsed.get("new_path") or parsed.get("path") or []
            
            if src and dst:
                parameters = {
                    "source_switch": str(src),
                    "destination_switch": str(dst),
                    "new_path": path if isinstance(path, list) else []
                }
            else:
                normalized_action = "none"
        
        return {
            "action": normalized_action,
            "parameters": parameters
        }
    
    def _get_fallback_response(self, reason: str) -> dict:
        """Return a safe fallback response when LLM fails."""
        logger.warning(f"Using fallback response: {reason}")
        return {
            "action": "none",
            "parameters": {},
            "fallback_reason": reason
        }
    
    def analyze_and_recommend(
        self,
        violation_type: str,
        current_rt: float,
        ema_rt: float,
        cluster_data: dict = None,
        network_data: dict = None,
        monitoring_data: dict = None,
        history: str = "",
        # Legacy parameters for backward compatibility
        monitoring_data_str: str = "",
        deployments_data: str = "",
        available_nodes: str = ""
    ) -> dict:
        """
        Main method: Analyze system state and recommend an action.
        
        This is called by the Intent Watch Loop when a violation is detected.
        
        Args:
            violation_type: "UPPER_THRESHOLD_EXCEEDED" or "LOWER_THRESHOLD_EXCEEDED"
            current_rt: Current response time in seconds
            ema_rt: EMA response time in seconds
            cluster_data: Kubernetes cluster state dict
            network_data: ONOS network state dict
            monitoring_data: sFlow monitoring data dict
            history: Formatted decision history string
            
        Returns:
            Dictionary with 'action' and 'parameters' keys
        """
        logger.info(f"Analyzing {violation_type} violation...")
        logger.info(f"Current RT: {current_rt:.2f}s, EMA: {ema_rt:.2f}s")
        logger.info(f"Thresholds: [{self.lower_threshold}, {self.upper_threshold}]")
        
        # Use empty dicts if not provided
        cluster_data = cluster_data or {}
        network_data = network_data or {}
        monitoring_data = monitoring_data or {}
        
        # Build the prompt
        prompt = self.build_prompt(
            violation_type=violation_type,
            current_rt=current_rt,
            ema_rt=ema_rt,
            cluster_data=cluster_data,
            network_data=network_data,
            monitoring_data=monitoring_data,
            history=history
        )
        
        logger.debug(f"Prompt length: {len(prompt)} characters")
        
        # Query the LLM
        response_text = self._query_ollama(prompt)
        
        # Parse the response
        action = self._parse_response(response_text)
        
        # Check if this action has repeatedly failed
        action = self._check_and_adjust_repeated_action(action, history, violation_type, cluster_data)
        
        logger.info(f"LLM recommended action: {action['action']}")
        if action['action'] != 'none':
            logger.info(f"Parameters: {action['parameters']}")
        
        return action
    
    def _check_and_adjust_repeated_action(self, action: dict, history: str, violation_type: str, cluster_data: dict) -> dict:
        """
        Check if the LLM is repeating a failed action and suggest an alternative.
        
        This helps overcome limitations of smaller LLMs that may not learn from feedback.
        """
        if action['action'] == 'none':
            return action
        
        # Count how many times the same action appears in history with WORSENED or NO_CHANGE
        action_str = f"action={action['action']}, params={action['parameters']}"
        
        # Look for patterns in history
        failed_count = history.count("WORSENED") + history.count("NO_CHANGE")
        same_deployment_failures = 0
        
        if action['action'] == 'horizontal_scaling':
            dep_name = action['parameters'].get('deployment_name', '')
            if dep_name and dep_name in history:
                # Count failures involving this deployment
                same_deployment_failures = history.count(f"'{dep_name}'") 
        
        # If we see repeated failures (3+) with horizontal_scaling, try vertical_scaling
        if failed_count >= 3 and action['action'] == 'horizontal_scaling':
            logger.warning(f"Detected {failed_count} failed attempts. Switching to vertical_scaling.")
            
            # Get the deployment name from the original action
            dep_name = action['parameters'].get('deployment_name', 'microservice1-deployment')
            
            # For LOWER_THRESHOLD: reduce resources to slow down
            if violation_type == "LOWER_THRESHOLD_EXCEEDED":
                return {
                    "action": "vertical_scaling",
                    "parameters": {
                        "deployment_name": dep_name,
                        "cpu_limit": "100m",  # Reduce CPU to slow down
                        "memory_limit": "128Mi"
                    },
                    "adjusted_reason": "Switched from repeated failed horizontal_scaling"
                }
            else:
                # For UPPER_THRESHOLD: increase resources
                return {
                    "action": "vertical_scaling",
                    "parameters": {
                        "deployment_name": dep_name,
                        "cpu_limit": "1000m",  # Increase CPU
                        "memory_limit": "1024Mi"
                    },
                    "adjusted_reason": "Switched from repeated failed horizontal_scaling"
                }
        
        # If same deployment failed multiple times, try a different deployment
        if same_deployment_failures >= 2 and action['action'] == 'horizontal_scaling':
            # Get list of deployments from cluster data
            # Structure: cluster_data["data"]["deployments"]["list"]
            try:
                deployments = cluster_data.get('data', {}).get('deployments', {}).get('list', [])
            except (AttributeError, TypeError):
                deployments = []
            
            current_dep = action['parameters'].get('deployment_name', '')
            
            # Find a different deployment to try
            for dep in deployments:
                # Handle both dict and string formats
                if isinstance(dep, dict):
                    dep_name = dep.get('name', '')
                    replicas = dep.get('replicas_desired', 1)
                elif isinstance(dep, str):
                    dep_name = dep
                    replicas = 1
                else:
                    continue
                
                if dep_name and dep_name != current_dep and 'microservice' in dep_name:
                    if violation_type == "LOWER_THRESHOLD_EXCEEDED":
                        new_replicas = max(1, replicas - 1)  # Reduce
                    else:
                        new_replicas = replicas + 1  # Increase
                    
                    logger.warning(f"Same deployment failed {same_deployment_failures} times. Trying {dep_name} instead.")
                    return {
                        "action": "horizontal_scaling",
                        "parameters": {
                            "deployment_name": dep_name,
                            "replicas": new_replicas
                        },
                        "adjusted_reason": f"Switched from {current_dep} due to repeated failures"
                    }
        
        return action
    
    def is_healthy(self) -> bool:
        """Check if Ollama is responding."""
        try:
            url = f"{self.ollama_url}/api/tags"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False