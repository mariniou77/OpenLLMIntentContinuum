"""
Decision Maker Module

This module integrates with the LLM (TinyLlama via Ollama) to analyze
system state and recommend actions when SLO violations occur.

Updated to use:
- Cleaner, more structured prompts
- Decision history for context
- Simpler expected output format
"""

import json
import logging
import re
import requests
from typing import Optional, Dict, Any
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
        return """Respond with ONLY a JSON object. No other text.

Problem: {violation_type}
Response time: {current_rt}s, Target: {lower_threshold}s-{upper_threshold}s

Current deployments:
{deployments_data}

Rule: If LOWER_THRESHOLD_EXCEEDED, set replicas to 1. If UPPER_THRESHOLD_EXCEEDED, set replicas to 3.

Pick one deployment and respond with exactly this format:
{{"action": "horizontal_scaling", "deployment_name": "microservice3-deployment", "replicas": 1}}
"""

    def build_prompt(
        self,
        violation_type: str,
        current_rt: float,
        ema_rt: float,
        monitoring_data: str,
        deployments_data: str,
        available_nodes: str,
        history: str
    ) -> str:
        """
        Build the complete prompt for the LLM using the new clean format.
        
        Args:
            violation_type: Type of violation (UPPER_THRESHOLD_EXCEEDED or LOWER_THRESHOLD_EXCEEDED)
            current_rt: Current response time in seconds
            ema_rt: EMA response time in seconds
            monitoring_data: Compact monitoring string from DataCollector
            deployments_data: Compact deployments string from DataCollector
            available_nodes: Comma-separated list of available worker nodes
            history: Formatted history string from DecisionHistory
            
        Returns:
            Complete prompt string
        """
        prompt = self.prompt_template.format(
            violation_type=violation_type,
            current_rt=round(current_rt, 2),
            ema_rt=round(ema_rt, 2),
            lower_threshold=self.lower_threshold,
            upper_threshold=self.upper_threshold,
            monitoring_data=monitoring_data,
            deployments_data=deployments_data,
            available_nodes=available_nodes,
            history=history
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
        
        # Debug logging - log full prompt
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
            
            # Debug logging - log full response
            if self.debug_llm:
                logger.info("=" * 60)
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
        
        Fallback method when JSON parsing fails.
        
        Args:
            response_text: Raw response text
            
        Returns:
            Extracted action dictionary
        """
        response_lower = response_text.lower()
        
        # Detect action type
        action = "none"
        if "horizontal_scaling" in response_lower or "replica" in response_lower:
            action = "horizontal_scaling"
        elif "vertical_scaling" in response_lower or "cpu_limit" in response_lower:
            action = "vertical_scaling"
        elif "service_placement" in response_lower or "target_node" in response_lower:
            action = "service_placement"
        elif "flow_scheduling" in response_lower or "new_path" in response_lower:
            action = "flow_scheduling"
        
        # Extract deployment name
        deployment = None
        dep_match = re.search(r'"deployment_name"\s*:\s*"([^"]+)"', response_text)
        if dep_match:
            deployment = dep_match.group(1)
        else:
            # Try to find microservice mentions
            for ms in ["microservice1", "microservice2", "microservice3", "microservice4"]:
                if ms in response_lower:
                    deployment = f"{ms}-deployment"
                    break
        
        # Build result based on action type
        result = {
            "action": action,
            "parameters": {}
        }
        
        if action == "horizontal_scaling" and deployment:
            replicas = 2  # default
            # Try both "replicas" and "replica_count"
            rep_match = re.search(r'"replicas"\s*:\s*(\d+)', response_text)
            if not rep_match:
                rep_match = re.search(r'"replica_count"\s*:\s*(\d+)', response_text)
            if rep_match:
                replicas = int(rep_match.group(1))
            result["parameters"] = {
                "deployment_name": deployment,
                "replicas": min(max(replicas, 1), 5)
            }
            
        elif action == "vertical_scaling" and deployment:
            cpu = "500m"
            mem = "512Mi"
            cpu_match = re.search(r'"cpu_limit"\s*:\s*"([^"]+)"', response_text)
            mem_match = re.search(r'"memory_limit"\s*:\s*"([^"]+)"', response_text)
            if cpu_match:
                cpu = cpu_match.group(1)
            if mem_match:
                mem = mem_match.group(1)
            result["parameters"] = {
                "deployment_name": deployment,
                "cpu_limit": cpu,
                "memory_limit": mem
            }
            
        elif action == "service_placement" and deployment:
            target = "worker1"
            node_match = re.search(r'"target_node"\s*:\s*"([^"]+)"', response_text)
            if node_match:
                target = node_match.group(1)
            result["parameters"] = {
                "deployment_name": deployment,
                "target_node": target
            }
            
        elif action == "flow_scheduling":
            src = dst = None
            src_match = re.search(r'"source_switch"\s*:\s*"([^"]+)"', response_text)
            dst_match = re.search(r'"destination_switch"\s*:\s*"([^"]+)"', response_text)
            if src_match:
                src = src_match.group(1)
            if dst_match:
                dst = dst_match.group(1)
            if src and dst:
                result["parameters"] = {
                    "source_switch": src,
                    "destination_switch": dst,
                    "new_path": []
                }
            else:
                result["action"] = "none"
        else:
            result["action"] = "none"
        
        return result
    
    def _normalize_response(self, parsed: dict) -> dict:
        """
        Normalize parsed JSON into standard format.
        
        Handles various field names TinyLlama might use.
        
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
            "service_placement": "service_placement",
            "serviceplacement": "service_placement",
            "placement": "service_placement",
            "migrate": "service_placement",
            "flow_scheduling": "flow_scheduling",
            "flowscheduling": "flow_scheduling",
            "reroute": "flow_scheduling"
        }
        
        normalized_action = action_mapping.get(action, "none")
        
        # Build parameters based on action type
        parameters = {}
        
        if normalized_action == "horizontal_scaling":
            dep_name = parsed.get("deployment_name") or parsed.get("deployment")
            # Accept both "replicas" and "replica_count" (TinyLlama sometimes uses replica_count)
            replicas = parsed.get("replicas") or parsed.get("replica_count", 2)
            if dep_name:
                parameters = {
                    "deployment_name": dep_name,
                    "replicas": min(max(int(replicas), 1), 5)
                }
            else:
                normalized_action = "none"
                
        elif normalized_action == "vertical_scaling":
            dep_name = parsed.get("deployment_name") or parsed.get("deployment")
            cpu = parsed.get("cpu_limit", "500m")
            mem = parsed.get("memory_limit", "512Mi")
            if dep_name:
                parameters = {
                    "deployment_name": dep_name,
                    "cpu_limit": cpu,
                    "memory_limit": mem
                }
            else:
                normalized_action = "none"
                
        elif normalized_action == "service_placement":
            dep_name = parsed.get("deployment_name") or parsed.get("deployment")
            target = parsed.get("target_node") or parsed.get("node")
            if dep_name and target:
                parameters = {
                    "deployment_name": dep_name,
                    "target_node": target
                }
            else:
                normalized_action = "none"
                
        elif normalized_action == "flow_scheduling":
            src = parsed.get("source_switch") or parsed.get("source")
            dst = parsed.get("destination_switch") or parsed.get("destination")
            path = parsed.get("new_path", [])
            if src and dst:
                parameters = {
                    "source_switch": src,
                    "destination_switch": dst,
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
    
    def _validate_action(self, action: dict) -> dict:
        """
        Validate and sanitize the action from LLM.
        
        Args:
            action: Parsed action dictionary
            
        Returns:
            Validated action dictionary
        """
        if not isinstance(action, dict):
            return self._get_fallback_response("Invalid action format")
        
        if "action" not in action:
            action["action"] = "none"
        
        if "parameters" not in action:
            action["parameters"] = {}
        
        # Validate specific action parameters
        if action["action"] == "horizontal_scaling":
            params = action.get("parameters", {})
            if not params.get("deployment_name"):
                action["action"] = "none"
                action["parameters"] = {}
            elif not isinstance(params.get("replicas"), int):
                try:
                    params["replicas"] = int(params.get("replicas", 2))
                except (ValueError, TypeError):
                    params["replicas"] = 2
        
        return action
    
    def analyze_and_recommend(
        self,
        violation_type: str,
        current_rt: float,
        ema_rt: float,
        monitoring_data: str,
        deployments_data: str,
        available_nodes: str,
        history: str
    ) -> dict:
        """
        Main method: Analyze system state and recommend an action.
        
        This is called by the Intent Watch Loop when a violation is detected.
        
        Args:
            violation_type: "UPPER_THRESHOLD_EXCEEDED" or "LOWER_THRESHOLD_EXCEEDED"
            current_rt: Current response time in seconds
            ema_rt: EMA response time in seconds
            monitoring_data: Compact monitoring string
            deployments_data: Compact deployments string
            available_nodes: Available worker nodes string
            history: Formatted decision history string
            
        Returns:
            Dictionary with 'action' and 'parameters' keys
        """
        logger.info(f"Analyzing {violation_type} violation...")
        logger.info(f"Current RT: {current_rt:.2f}s, EMA: {ema_rt:.2f}s")
        
        # Build the prompt
        prompt = self.build_prompt(
            violation_type=violation_type,
            current_rt=current_rt,
            ema_rt=ema_rt,
            monitoring_data=monitoring_data,
            deployments_data=deployments_data,
            available_nodes=available_nodes,
            history=history
        )
        
        logger.debug(f"Prompt length: {len(prompt)} characters")
        
        # Query the LLM
        response_text = self._query_ollama(prompt)
        
        # Parse the response
        action = self._parse_response(response_text)
        
        # Validate the action
        action = self._validate_action(action)
        
        logger.info(f"Recommended action: {action['action']}")
        if action['action'] != 'none':
            logger.info(f"Parameters: {action['parameters']}")
        
        return action
    
    def is_healthy(self) -> bool:
        """Check if Ollama is responding."""
        try:
            url = f"{self.ollama_url}/api/tags"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False