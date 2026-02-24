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
            logger.warning(f"Prompt template not found at {template_path}, using embedded default")
            return self._get_default_prompt_template()
    
    def _get_default_prompt_template(self) -> str:
        """Return embedded default template if file not found."""
        return """You are a Kubernetes resource manager. Recommend ONE action.

## PROBLEM
EMA Response Time: {ema_rt}s
Target Range: {lower_threshold}s - {upper_threshold}s
Status: {status}

## WHAT TO DO
{what_to_do}

## DEPLOYMENTS
{deployments_table}

## CONSTRAINTS
{constraints}
{history_section}
Respond ONLY with JSON:
{{"action": "horizontal_scaling", "parameters": {{"deployment_name": "X-deployment", "replicas": N}}}}
or
{{"action": "vertical_scaling", "parameters": {{"deployment_name": "X-deployment", "cpu_limit": "Xm", "memory_limit": "XMi"}}}}

JSON:
"""
    
    def _get_enabled_actions_description(self) -> str:
        """Get description of enabled actions for the prompt."""
        # Only return horizontal and vertical scaling descriptions
        # Other actions are kept in code but not offered to the LLM
        return """1. horizontal_scaling: Change replicas.
   {"action": "horizontal_scaling", "parameters": {"deployment_name": "X-deployment", "replicas": N}}

2. vertical_scaling: Change CPU/memory limits.
   {"action": "vertical_scaling", "parameters": {"deployment_name": "X-deployment", "cpu_limit": "Xm", "memory_limit": "XMi"}}"""

    def _format_system_state(self, cluster_data: dict, network_data: dict, monitoring_data: dict) -> str:
        """
        Format system state as a deployments table for the prompt.
        
        Returns a table format like:
        | Name                       | Replicas | CPU Lim | CPU Used | Mem Lim | Mem Used |
        |----------------------------|----------|---------|----------|---------|----------|
        | microservice1-deployment   | 3        | 300m    | 25m      | 312Mi   | 128Mi    |
        """
        lines = []
        
        # Get valid deployment names from config
        valid_deployment_names = set()
        k8s_config = self.config.get("kubernetes", {})
        for dep in k8s_config.get("deployments", []):
            dep_name = dep.get("name", "")
            if dep_name:
                valid_deployment_names.add(dep_name)
        
        # Get deployments from cluster data
        deployments = cluster_data.get("deployments", {}).get("list", [])
        
        if not deployments:
            # Try alternative structure
            deployments = cluster_data.get("data", {}).get("deployments", {}).get("list", [])
        
        if deployments:
            lines.append("| Name                       | Replicas | CPU Lim | CPU Used | Mem Lim | Mem Used |")
            lines.append("|----------------------------|----------|---------|----------|---------|----------|")
            
            for d in deployments:
                name = d.get("name", "unknown")
                # Only include deployments that are in config
                if valid_deployment_names and name not in valid_deployment_names:
                    continue
                    
                replicas = d.get("replicas_ready", d.get("replicas_desired", 0))
                
                # Get resource limits (from deployment spec)
                cpu_limit = d.get("cpu_limit") or "N/A"
                memory_limit = d.get("memory_limit") or "N/A"
                
                # Get current usage (from kubectl top)
                cpu_usage = d.get("cpu_usage") or "N/A"
                memory_usage = d.get("memory_usage") or "N/A"
                
                # Pad name for alignment
                padded_name = name.ljust(26)
                lines.append(f"| {padded_name} | {str(replicas).ljust(8)} | {str(cpu_limit).ljust(7)} | {str(cpu_usage).ljust(8)} | {str(memory_limit).ljust(7)} | {str(memory_usage).ljust(8)} |")
        
        return '\n'.join(lines) if lines else "No deployment data available"
    
    def _format_history_for_prompt(self, history: str, violation_type: str) -> tuple:
        """
        Parse history and return (failed_deployments, successful_pattern).
        
        For UPPER threshold: list deployments that WORSENED
        For LOWER threshold: list deployments that IMPROVED (as a pattern to follow)
        """
        failed_deployments = []
        successful_deployments = []
        
        if not history or history == "No previous decisions":
            return [], []
        
        # Get deployment names from config (dynamic, not hardcoded)
        deployment_names = []
        k8s_config = self.config.get("kubernetes", {})
        for dep in k8s_config.get("deployments", []):
            dep_name = dep.get("name", "")
            if dep_name:
                deployment_names.append(dep_name)
        
        # Fallback: if no deployments in config, try to extract from history using regex
        if not deployment_names:
            import re
            # Match patterns like "microservice1-deployment" or "my-app-deployment"
            found = re.findall(r'[\w-]+-deployment', history)
            deployment_names = list(set(found))
        
        # Parse history lines looking for outcomes
        for line in history.split('\n'):
            if 'WORSENED' in line:
                # Extract deployment name
                for dep_name in deployment_names:
                    if dep_name in line:
                        if dep_name not in failed_deployments:
                            failed_deployments.append(dep_name)
                        break
            elif 'IMPROVED' in line:
                for dep_name in deployment_names:
                    if dep_name in line:
                        if dep_name not in successful_deployments:
                            successful_deployments.append(dep_name)
                        break
        
        return failed_deployments, successful_deployments
    
    def _get_available_deployments(self, cluster_data: dict, failed_deployments: list) -> list:
        """Get list of deployments that haven't failed, with a hint about good candidates."""
        deployments = cluster_data.get("deployments", {}).get("list", [])
        if not deployments:
            deployments = cluster_data.get("data", {}).get("deployments", {}).get("list", [])
        
        # Get valid deployment names from config
        valid_deployment_names = set()
        k8s_config = self.config.get("kubernetes", {})
        for dep in k8s_config.get("deployments", []):
            dep_name = dep.get("name", "")
            if dep_name:
                valid_deployment_names.add(dep_name)
        
        available = []
        for d in deployments:
            name = d.get("name", "")
            # Only include deployments that are in config and haven't failed
            if name in valid_deployment_names and name not in failed_deployments:
                replicas = d.get("replicas_ready", d.get("replicas_desired", 0))
                hint = ""
                if replicas == 1:
                    hint = " (only 1 replica - good candidate)"
                elif replicas >= 4:
                    hint = f" ({replicas} replicas - can reduce)"
                available.append(f"- {name}{hint}")
        
        return available

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
        Build the v3 prompt using the template file.
        
        This prompt structure has been tested and validated to produce
        correct horizontal and vertical scaling decisions with qwen2.5:3b.
        
        Args:
            violation_type: Type of violation
            current_rt: Current response time in seconds
            ema_rt: EMA response time in seconds
            cluster_data: Kubernetes cluster state
            network_data: ONOS network state (not used in v3)
            monitoring_data: sFlow monitoring metrics
            history: Formatted decision history string
            
        Returns:
            Complete prompt string
        """
        # Determine the problem status and what to do based on violation type
        if violation_type == "UPPER_THRESHOLD_EXCEEDED":
            status = "TOO SLOW - must speed up"
            what_to_do = """- INCREASE replicas (e.g., 1→2 or 2→3) to add capacity
- OR INCREASE cpu_limit (e.g., 300m→400m) to add power"""
        else:  # LOWER_THRESHOLD_EXCEEDED
            status = "TOO FAST - must slow down to save resources"
            what_to_do = """- DECREASE replicas (e.g., 3→2 or 2→1) to reduce capacity
- OR DECREASE cpu_limit (e.g., 400m→300m) to reduce power
- IMPORTANT: Only target deployments with replicas >= 2 (minimum is 1)
- IMPORTANT: Only target deployments with cpu_limit >= 200m (minimum is 100m)"""
        
        # Format deployments table
        deployments_table = self._format_system_state(cluster_data, network_data, monitoring_data)
        
        # Parse history to find failed and successful deployments
        failed_deployments, successful_deployments = self._format_history_for_prompt(history, violation_type)
        
        # Build history section based on violation type
        history_section = ""
        if violation_type == "UPPER_THRESHOLD_EXCEEDED" and failed_deployments:
            history_section = "\n## HISTORY (do not repeat failed actions)\n"
            for dep in failed_deployments:
                history_section += f"- {dep}: WORSENED (do not use)\n"
            
            # Add available deployments hint
            available = self._get_available_deployments(cluster_data, failed_deployments)
            if available:
                history_section += "\n## AVAILABLE DEPLOYMENTS\n"
                history_section += '\n'.join(available) + "\n"
        elif violation_type == "LOWER_THRESHOLD_EXCEEDED" and successful_deployments:
            history_section = "\n## HISTORY (follow successful pattern)\n"
            for dep in successful_deployments:
                history_section += f"- {dep}: IMPROVED (good choice)\n"
        elif failed_deployments:
            history_section = "\n## HISTORY (do not repeat failed actions)\n"
            for dep in failed_deployments:
                history_section += f"- {dep}: WORSENED (do not use)\n"
        
        # Get constraints from config or use defaults
        constraints = "Replicas: 1-5 | CPU: 100m-500m | Memory: 128Mi-512Mi"
        
        # Fill in the template
        prompt = self.prompt_template.format(
            ema_rt=f"{ema_rt:.2f}",
            lower_threshold=self.lower_threshold,
            upper_threshold=self.upper_threshold,
            status=status,
            what_to_do=what_to_do,
            deployments_table=deployments_table,
            constraints=constraints,
            history_section=history_section
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
        
        # Extract parameters - they might be nested in "parameters" key or at top level
        params = parsed.get("parameters", {})
        if not params:
            params = parsed  # Fall back to top-level if no nested parameters
        
        # Build parameters based on action type
        parameters = {}
        
        if normalized_action == "horizontal_scaling":
            dep_name = params.get("deployment_name") or params.get("deployment") or params.get("name")
            
            # Get replicas - handle 0 explicitly since it's falsy
            replicas = params.get("replicas")
            if replicas is None:
                replicas = params.get("replica_count")
            if replicas is None:
                replicas = params.get("replica")
            if replicas is None:
                replicas = 2  # Default
            
            # Handle non-numeric replicas
            if isinstance(replicas, (list, dict)):
                replicas = 2
            try:
                replicas = int(replicas)
            except (ValueError, TypeError):
                replicas = 2
            
            # Clamp replicas to valid range (1-5)
            replicas = max(1, min(5, replicas))
            
            if dep_name:
                parameters = {
                    "deployment_name": str(dep_name),
                    "replicas": replicas
                }
            else:
                normalized_action = "none"
                
        elif normalized_action == "vertical_scaling":
            dep_name = params.get("deployment_name") or params.get("deployment") or params.get("name")
            cpu = params.get("cpu_limit") or params.get("cpu") or "500m"
            mem = params.get("memory_limit") or params.get("memory") or "512Mi"
            
            if dep_name:
                parameters = {
                    "deployment_name": str(dep_name),
                    "cpu_limit": str(cpu),
                    "memory_limit": str(mem)
                }
            else:
                normalized_action = "none"
                
        elif normalized_action == "service_placement":
            dep_name = params.get("deployment_name") or params.get("deployment") or params.get("name")
            target = params.get("target_node") or params.get("node") or params.get("target")
            
            if dep_name and target:
                parameters = {
                    "deployment_name": str(dep_name),
                    "target_node": str(target)
                }
            else:
                normalized_action = "none"
                
        elif normalized_action == "flow_scheduling":
            src = params.get("source_switch") or params.get("source") or params.get("ingress")
            dst = params.get("destination_switch") or params.get("destination") or params.get("egress")
            path = params.get("new_path") or params.get("path") or []
            
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
        
        # 100% LLM decision-making - no Python override
        # The LLM is fully responsible for learning from history and choosing appropriate actions
        
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