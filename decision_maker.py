"""
Decision Maker Module

This module integrates with the LLM (TinyLlama via Ollama) to analyze
system state and recommend actions when SLO violations occur.

It follows the IntentContinuum paper's approach:
1. Receive system state from Data Collector
2. Build a structured prompt for the LLM
3. Query the LLM for root cause analysis and recommended action
4. Parse and validate the LLM response
"""

import json
import logging
import requests
from typing import Optional
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
        return """You are a Kubernetes administrator. Analyze this system state and recommend an action.

Violation: {violation_type}
Thresholds: Upper={upper_threshold}s, Lower={lower_threshold}s

{system_state}

Respond with JSON: {{"analysis": "...", "source": "compute|network", "action": "horizontal_scaling|none", "parameters": {{"deployment_name": "...", "replicas": N}}}}

JSON Response:"""
    
    def _build_prompt(self, system_state_formatted: str, violation_type: str) -> str:
        """
        Build the complete prompt for the LLM.
        
        Args:
            system_state_formatted: Formatted system state string
            violation_type: Type of violation (UPPER or LOWER)
            
        Returns:
            Complete prompt string
        """
        prompt = self.prompt_template.format(
            violation_type=violation_type,
            upper_threshold=self.upper_threshold,
            lower_threshold=self.lower_threshold,
            system_state=system_state_formatted
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
            "options": {
                "temperature": self.temperature
            }
        }
        
        try:
            logger.info(f"Querying Ollama ({self.model})...")
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
            
        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            return None
    
    def _parse_response(self, response_text: str) -> dict:
        """
        Parse LLM response to extract JSON action.
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            Parsed action dictionary
        """
        if not response_text:
            return self._get_fallback_response("No response from LLM")
        
        # Try to extract JSON from the response
        try:
            # First, try to parse the entire response as JSON
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON object in the response
        try:
            # Look for JSON between curly braces
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            
            if start != -1 and end > start:
                json_str = response_text[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        logger.warning(f"Could not parse LLM response as JSON: {response_text[:200]}")
        return self._get_fallback_response("Could not parse LLM response")
    
    def _get_fallback_response(self, reason: str) -> dict:
        """Return a safe fallback response when LLM fails."""
        return {
            "analysis": f"Fallback response: {reason}",
            "source": "unknown",
            "action": "none",
            "parameters": {}
        }
    
    def _validate_action(self, action: dict) -> dict:
        """
        Validate and sanitize the action from LLM.
        
        Args:
            action: Parsed action dictionary
            
        Returns:
            Validated action dictionary
        """
        # Ensure required fields exist
        if "action" not in action:
            action["action"] = "none"
        
        if "parameters" not in action:
            action["parameters"] = {}
        
        if "analysis" not in action:
            action["analysis"] = "No analysis provided"
        
        if "source" not in action:
            action["source"] = "unknown"
        
        # Validate horizontal_scaling parameters
        if action["action"] == "horizontal_scaling":
            params = action["parameters"]
            
            # Check deployment name exists
            if "deployment_name" not in params:
                logger.warning("horizontal_scaling missing deployment_name")
                action["action"] = "none"
                return action
            
            # Check replicas is a valid number
            if "replicas" not in params:
                logger.warning("horizontal_scaling missing replicas")
                action["action"] = "none"
                return action
            
            try:
                replicas = int(params["replicas"])
                # Enforce reasonable limits
                if replicas < 1:
                    replicas = 1
                if replicas > 10:
                    replicas = 10
                params["replicas"] = replicas
            except (ValueError, TypeError):
                logger.warning(f"Invalid replicas value: {params.get('replicas')}")
                action["action"] = "none"
        
        return action
    
    def analyze_and_recommend(self, system_state_formatted: str, violation_type: str) -> dict:
        """
        Main method: Analyze system state and recommend an action.
        
        This is called by the Intent Watch Loop when a violation is detected.
        
        Args:
            system_state_formatted: Formatted system state from DataCollector
            violation_type: "UPPER" (response time too high) or "LOWER" (too low)
            
        Returns:
            Dictionary with analysis and recommended action
        """
        logger.info(f"Analyzing {violation_type} violation...")
        
        # Build the prompt
        prompt = self._build_prompt(system_state_formatted, violation_type)
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