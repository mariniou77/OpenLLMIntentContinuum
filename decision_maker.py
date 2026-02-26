"""
Decision Maker Module (Improved for Small LLMs)

This module integrates with a local LLM (e.g., Qwen2.5:3b via Ollama) to analyze
system state and recommend actions when SLO violations occur.

Key improvements over the original:
- Few-shot examples in prompt for better structured output
- Validation + retry loop (up to 3 attempts with corrective micro-prompts)
- Deployment name validation against actual config
- Re-enabled safety net for repeated failed actions
- Smarter default values and clamping

Supports 4 action types:
1. horizontal_scaling - Change replica count
2. vertical_scaling - Change CPU/memory limits
3. service_placement - Move pod to different node
4. flow_scheduling - Change network path via ONOS
"""

import json
import logging
import re
import requests
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)

# Maximum number of LLM query attempts before falling back
MAX_LLM_RETRIES = 3


class DecisionMaker:
    """
    LLM-powered decision maker for intent-based resource management.
    
    Optimized for small open-source LLMs (1-7B parameters) with:
    - Constrained prompts with few-shot examples
    - Validation and retry logic
    - Fallback safety nets for repeated failures
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
        
        # Build valid deployment names set from config (used for validation)
        self.valid_deployment_names = set()
        k8s_config = config.get("kubernetes", {})
        for dep in k8s_config.get("deployments", []):
            dep_name = dep.get("name", "")
            if dep_name:
                self.valid_deployment_names.add(dep_name.lower())
        
        # Track consecutive parse failures for diagnostics
        self._consecutive_failures = 0
    
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
        return """You are a Kubernetes resource manager. Pick ONE action to fix the problem.

PROBLEM: EMA Response Time is {ema_rt}s (target: {lower_threshold}s-{upper_threshold}s)
STATUS: {status}

RULE: {what_to_do}

DEPLOYMENTS:
{deployments_table}

LIMITS: {constraints}
{history_section}
EXAMPLES:
- Too slow with 1 replica: {{"action":"horizontal_scaling","parameters":{{"deployment_name":"microservice1-deployment","replicas":2}}}}
- Too slow, need more CPU: {{"action":"vertical_scaling","parameters":{{"deployment_name":"microservice3-deployment","cpu_limit":"600m","memory_limit":"612Mi"}}}}
- Too fast with 3 replicas: {{"action":"horizontal_scaling","parameters":{{"deployment_name":"microservice1-deployment","replicas":2}}}}
- Too fast, reduce CPU: {{"action":"vertical_scaling","parameters":{{"deployment_name":"microservice3-deployment","cpu_limit":"300m","memory_limit":"312Mi"}}}}

Pick ONE deployment from the table above. Respond with ONLY a JSON object, nothing else.
JSON:
"""

    def _get_enabled_actions_description(self) -> str:
        """Get description of enabled actions for the prompt."""
        return """1. horizontal_scaling: Change replicas.
   {"action": "horizontal_scaling", "parameters": {"deployment_name": "X-deployment", "replicas": N}}

2. vertical_scaling: Change CPU/memory limits.
   {"action": "vertical_scaling", "parameters": {"deployment_name": "X-deployment", "cpu_limit": "Xm", "memory_limit": "XMi"}}"""

    def _format_system_state(self, cluster_data: dict, network_data: dict, monitoring_data: dict) -> str:
        """
        Format system state as a deployments table for the prompt.
        
        Returns a table format like:
        | Name                       | Replicas | CPU Lim | CPU Used | Mem Lim | Mem Used |
        """
        lines = []
        
        # Get deployments from cluster data
        deployments = cluster_data.get("deployments", {}).get("list", [])
        
        if not deployments:
            deployments = cluster_data.get("data", {}).get("deployments", {}).get("list", [])
        
        if deployments:
            lines.append("| Name                       | Replicas | CPU Lim | CPU Used | Mem Lim | Mem Used |")
            lines.append("|----------------------------|----------|---------|----------|---------|----------|")
            
            for d in deployments:
                name = d.get("name", "unknown")
                # Only include deployments that are in config
                if self.valid_deployment_names and name.lower() not in self.valid_deployment_names:
                    continue
                    
                replicas = d.get("replicas_ready", d.get("replicas_desired", 0))
                cpu_limit = d.get("cpu_limit") or "N/A"
                memory_limit = d.get("memory_limit") or "N/A"
                cpu_usage = d.get("cpu_usage") or "N/A"
                memory_usage = d.get("memory_usage") or "N/A"
                
                padded_name = name.ljust(26)
                lines.append(f"| {padded_name} | {str(replicas).ljust(8)} | {str(cpu_limit).ljust(7)} | {str(cpu_usage).ljust(8)} | {str(memory_limit).ljust(7)} | {str(memory_usage).ljust(8)} |")
        
        return '\n'.join(lines) if lines else "No deployment data available"
    
    def _format_history_for_prompt(self, history: str, violation_type: str) -> tuple:
        """
        Parse history and return (failed_deployments, successful_deployments).
        """
        failed_deployments = []
        successful_deployments = []
        
        if not history or history == "(none)":
            return [], []
        
        # Get deployment names from config
        deployment_names = [dep.get("name", "") for dep in 
                          self.config.get("kubernetes", {}).get("deployments", [])
                          if dep.get("name")]
        
        # Fallback: extract from history using regex
        if not deployment_names:
            found = re.findall(r'[\w-]+-deployment', history)
            deployment_names = list(set(found))
        
        for line in history.split('\n'):
            if 'WORSENED' in line:
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
        """Get list of deployments that haven't failed."""
        deployments = cluster_data.get("deployments", {}).get("list", [])
        if not deployments:
            deployments = cluster_data.get("data", {}).get("deployments", {}).get("list", [])
        
        available = []
        for d in deployments:
            name = d.get("name", "")
            if self.valid_deployment_names and name.lower() not in self.valid_deployment_names:
                continue
            if name not in failed_deployments:
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
        Build the prompt using the template file with few-shot examples.
        """
        # Determine the problem status and what to do
        if violation_type == "UPPER_THRESHOLD_EXCEEDED":
            status = "TOO SLOW - must speed up"
            what_to_do = "INCREASE replicas (e.g., 1->2) OR INCREASE cpu_limit (e.g., 300m->500m)"
        else:
            status = "TOO FAST - must slow down to save resources"
            what_to_do = "DECREASE replicas (e.g., 2->1, min=1) OR DECREASE cpu_limit (e.g., 500m->300m, min=100m)"
        
        # Format deployments table
        deployments_table = self._format_system_state(cluster_data, network_data, monitoring_data)
        
        # Parse history for failed/successful deployments
        failed_deployments, successful_deployments = self._format_history_for_prompt(history, violation_type)
        
        # Build history section
        history_section = ""
        if violation_type == "UPPER_THRESHOLD_EXCEEDED" and failed_deployments:
            history_section = "\nHISTORY (avoid these - they WORSENED):\n"
            for dep in failed_deployments:
                history_section += f"- {dep}: FAILED\n"
            available = self._get_available_deployments(cluster_data, failed_deployments)
            if available:
                history_section += "TRY INSTEAD:\n" + '\n'.join(available) + "\n"
        elif violation_type == "LOWER_THRESHOLD_EXCEEDED" and successful_deployments:
            history_section = "\nHISTORY (these worked well):\n"
            for dep in successful_deployments:
                history_section += f"- {dep}: IMPROVED\n"
        elif failed_deployments:
            history_section = "\nHISTORY (avoid these):\n"
            for dep in failed_deployments:
                history_section += f"- {dep}: FAILED\n"
        
        # Constraints
        constraints = "Replicas: 1-5 | CPU: 100m-1000m | Memory: 128Mi-1024Mi"
        
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
    
    def _build_retry_prompt(self, previous_response: str, error_reason: str) -> str:
        """
        Build a short corrective prompt for retry attempts.
        
        Small LLMs often fix their output when given specific feedback
        about what was wrong.
        
        Args:
            previous_response: The LLM's previous (failed) response
            error_reason: What was wrong with it
            
        Returns:
            Short corrective prompt
        """
        valid_names = sorted(self.valid_deployment_names)
        names_str = ", ".join(valid_names) if valid_names else "microservice1-deployment, microservice3-deployment"
        
        return f"""Your previous response was invalid: {error_reason}

Valid deployment names: {names_str}

Respond with ONLY a JSON object like one of these:
{{"action":"horizontal_scaling","parameters":{{"deployment_name":"{valid_names[0] if valid_names else 'microservice1-deployment'}","replicas":2}}}}
{{"action":"vertical_scaling","parameters":{{"deployment_name":"{valid_names[0] if valid_names else 'microservice1-deployment'}","cpu_limit":"500m","memory_limit":"512Mi"}}}}

JSON:"""

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
    
    def _validate_action(self, action: dict) -> tuple:
        """
        Validate a parsed action against known constraints.
        
        Returns:
            (is_valid: bool, error_reason: str)
        """
        action_type = action.get("action", "none")
        params = action.get("parameters", {})
        
        if action_type == "none":
            return False, "No action recommended"
        
        # Check action type is known
        valid_actions = {"horizontal_scaling", "vertical_scaling", "service_placement", "flow_scheduling"}
        if action_type not in valid_actions:
            return False, f"Unknown action type: {action_type}"
        
        # Check action is enabled
        if not self.actions_enabled.get(action_type, False):
            return False, f"Action '{action_type}' is not enabled"
        
        # Validate deployment name exists
        dep_name = params.get("deployment_name", "")
        if action_type in ("horizontal_scaling", "vertical_scaling", "service_placement"):
            if not dep_name:
                return False, "Missing deployment_name"
            if self.valid_deployment_names and dep_name.lower() not in self.valid_deployment_names:
                return False, f"Invalid deployment_name '{dep_name}'. Must be one of: {sorted(self.valid_deployment_names)}"
        
        # Validate horizontal_scaling parameters
        if action_type == "horizontal_scaling":
            replicas = params.get("replicas")
            if replicas is None:
                return False, "Missing replicas count"
            try:
                replicas = int(replicas)
            except (ValueError, TypeError):
                return False, f"Invalid replicas value: {replicas}"
            if replicas < 1 or replicas > 10:
                return False, f"Replicas {replicas} out of range (1-10)"
        
        # Validate vertical_scaling parameters
        if action_type == "vertical_scaling":
            cpu = params.get("cpu_limit", "")
            mem = params.get("memory_limit", "")
            if not cpu or not mem:
                return False, "Missing cpu_limit or memory_limit"
        
        return True, ""
    
    def _parse_response(self, response_text: str) -> dict:
        """
        Parse LLM response to extract JSON action.
        
        Handles various formats small LLMs might produce.
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            Parsed action dictionary with 'action' and 'parameters' keys
        """
        if not response_text:
            return self._get_fallback_response("No response from LLM")
        
        # Clean up the response
        cleaned = response_text.strip()
        
        # Remove markdown code fences if present
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)
        cleaned = cleaned.strip()
        
        # Try 1: Direct JSON parse
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return self._normalize_response(parsed)
        except json.JSONDecodeError:
            pass
        
        # Try 2: Find JSON object in the response (between first { and last })
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
        
        # Try 3: Find nested JSON (some models wrap in extra braces or arrays)
        try:
            # Find all JSON-like objects
            json_objects = re.findall(r'\{[^{}]*\}', cleaned)
            for obj_str in json_objects:
                try:
                    parsed = json.loads(obj_str)
                    if isinstance(parsed, dict) and ("action" in parsed or "deployment_name" in parsed):
                        return self._normalize_response(parsed)
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass
        
        # Try 4: Extract using regex patterns
        try:
            result = self._extract_with_regex(response_text)
            if result.get("action") != "none":
                return result
        except Exception as e:
            logger.warning(f"Regex extraction failed: {e}")
        
        logger.warning(f"Could not parse LLM response: {response_text[:300]}")
        return self._get_fallback_response("Could not parse LLM response")
    
    def _extract_with_regex(self, response_text: str) -> dict:
        """
        Extract action information using regex patterns.
        Fallback for when JSON parsing fails.
        """
        result = {"action": "none", "parameters": {}}
        
        # Try to find action type
        action_match = re.search(r'"action"\s*:\s*"([^"]+)"', response_text)
        if action_match:
            action = action_match.group(1).lower().strip()
            result["action"] = action
        
        if "horizontal_scaling" in response_text.lower() or (result["action"] == "none" and "scale" in response_text.lower() and "replicas" in response_text.lower()):
            dep_match = re.search(r'"deployment_name"\s*:\s*"([^"]+)"', response_text)
            rep_match = re.search(r'"replicas"\s*:\s*(\d+)', response_text)
            
            if dep_match:
                result["action"] = "horizontal_scaling"
                result["parameters"] = {
                    "deployment_name": dep_match.group(1),
                    "replicas": int(rep_match.group(1)) if rep_match else 2
                }
        
        elif "vertical_scaling" in response_text.lower() or (result["action"] == "none" and "cpu_limit" in response_text.lower()):
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
            "resize": "vertical_scaling",
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
        
        # Extract parameters - might be nested in "parameters" key or at top level
        params = parsed.get("parameters", {})
        if not isinstance(params, dict):
            params = {}
        if not params:
            # Fall back to top-level keys (some models flatten the structure)
            params = {k: v for k, v in parsed.items() if k != "action"}
        
        # Build parameters based on action type
        parameters = {}
        
        if normalized_action == "horizontal_scaling":
            dep_name = (params.get("deployment_name") or params.get("deployment") 
                       or params.get("name") or params.get("pod") or "")
            
            replicas = params.get("replicas")
            if replicas is None:
                replicas = params.get("replica_count") or params.get("replica") or params.get("count")
            if replicas is None:
                replicas = 2
            
            # Handle non-numeric replicas
            if isinstance(replicas, (list, dict)):
                replicas = 2
            try:
                replicas = int(replicas)
            except (ValueError, TypeError):
                replicas = 2
            
            # Clamp replicas to valid range
            replicas = max(1, min(5, replicas))
            
            if dep_name:
                parameters = {
                    "deployment_name": str(dep_name).lower(),
                    "replicas": replicas
                }
            else:
                normalized_action = "none"
                
        elif normalized_action == "vertical_scaling":
            dep_name = (params.get("deployment_name") or params.get("deployment") 
                       or params.get("name") or "")
            cpu = params.get("cpu_limit") or params.get("cpu") or params.get("cpu_limits") or "500m"
            mem = (params.get("memory_limit") or params.get("memory") 
                  or params.get("mem_limit") or params.get("memory_limits") or "512Mi")
            
            # Normalize CPU format (ensure it ends with 'm')
            cpu = str(cpu)
            if cpu.isdigit():
                cpu = f"{cpu}m"
            
            # Normalize memory format (ensure it ends with 'Mi')
            mem = str(mem)
            if mem.isdigit():
                mem = f"{mem}Mi"
            
            if dep_name:
                parameters = {
                    "deployment_name": str(dep_name).lower(),
                    "cpu_limit": cpu,
                    "memory_limit": mem
                }
            else:
                normalized_action = "none"
                
        elif normalized_action == "service_placement":
            dep_name = (params.get("deployment_name") or params.get("deployment") 
                       or params.get("name") or "")
            target = (params.get("target_node") or params.get("node") 
                     or params.get("target") or "")
            
            if dep_name and target:
                parameters = {
                    "deployment_name": str(dep_name).lower(),
                    "target_node": str(target)
                }
            else:
                normalized_action = "none"
                
        elif normalized_action == "flow_scheduling":
            src = (params.get("source_switch") or params.get("source") 
                  or params.get("ingress") or "")
            dst = (params.get("destination_switch") or params.get("destination") 
                  or params.get("egress") or "")
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
    
    def _get_smart_fallback(self, violation_type: str, cluster_data: dict) -> dict:
        """
        Generate an intelligent fallback when the LLM fails after all retries.
        
        Instead of returning 'none', pick a reasonable default action based on
        the violation type and current cluster state.
        
        Args:
            violation_type: Type of violation
            cluster_data: Current cluster state
            
        Returns:
            A reasonable default action
        """
        # Get deployments from cluster data
        deployments = cluster_data.get("deployments", {}).get("list", [])
        if not deployments:
            deployments = cluster_data.get("data", {}).get("deployments", {}).get("list", [])
        
        # Filter to valid deployments
        valid_deps = []
        for d in deployments:
            name = d.get("name", "")
            if self.valid_deployment_names and name.lower() not in self.valid_deployment_names:
                continue
            valid_deps.append(d)
        
        if not valid_deps:
            return self._get_fallback_response("No valid deployments found for smart fallback")
        
        if violation_type == "UPPER_THRESHOLD_EXCEEDED":
            # Find deployment with lowest replica count (bottleneck candidate)
            best = min(valid_deps, key=lambda d: d.get("replicas_ready", d.get("replicas_desired", 1)))
            current_replicas = best.get("replicas_ready", best.get("replicas_desired", 1))
            new_replicas = min(5, current_replicas + 1)
            
            logger.warning(f"Smart fallback: scaling up {best['name']} from {current_replicas} to {new_replicas}")
            return {
                "action": "horizontal_scaling",
                "parameters": {
                    "deployment_name": best["name"],
                    "replicas": new_replicas
                },
                "fallback_reason": "LLM failed after retries, using smart fallback"
            }
        else:  # LOWER_THRESHOLD_EXCEEDED
            # Find deployment with highest replica count (can be scaled down)
            candidates = [d for d in valid_deps 
                         if d.get("replicas_ready", d.get("replicas_desired", 1)) > 1]
            
            if candidates:
                best = max(candidates, key=lambda d: d.get("replicas_ready", d.get("replicas_desired", 1)))
                current_replicas = best.get("replicas_ready", best.get("replicas_desired", 1))
                new_replicas = max(1, current_replicas - 1)
                
                logger.warning(f"Smart fallback: scaling down {best['name']} from {current_replicas} to {new_replicas}")
                return {
                    "action": "horizontal_scaling",
                    "parameters": {
                        "deployment_name": best["name"],
                        "replicas": new_replicas
                    },
                    "fallback_reason": "LLM failed after retries, using smart fallback"
                }
            else:
                return self._get_fallback_response("All deployments at minimum replicas, no scale-down possible")

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
        
        Uses a retry loop: if the first LLM response is invalid, retries
        with a corrective micro-prompt up to MAX_LLM_RETRIES times.
        
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
        
        # Build the initial prompt
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
        
        # === RETRY LOOP ===
        last_response_text = None
        last_error = ""
        
        for attempt in range(1, MAX_LLM_RETRIES + 1):
            if attempt == 1:
                current_prompt = prompt
            else:
                # Use corrective micro-prompt on retry
                logger.warning(f"Retry {attempt}/{MAX_LLM_RETRIES}: {last_error}")
                current_prompt = self._build_retry_prompt(last_response_text or "", last_error)
            
            # Query the LLM
            response_text = self._query_ollama(current_prompt)
            last_response_text = response_text
            
            if not response_text:
                last_error = "No response from LLM"
                continue
            
            # Parse the response
            action = self._parse_response(response_text)
            
            # Validate the parsed action
            is_valid, error_reason = self._validate_action(action)
            
            if is_valid:
                self._consecutive_failures = 0
                logger.info(f"Valid action on attempt {attempt}: {action['action']}")
                
                # Apply safety net for repeated failures
                action = self._check_and_adjust_repeated_action(
                    action, history, violation_type, cluster_data
                )
                
                logger.info(f"Final recommended action: {action['action']}")
                if action['action'] != 'none':
                    logger.info(f"Parameters: {action['parameters']}")
                
                return action
            else:
                last_error = error_reason
                logger.warning(f"Attempt {attempt}: invalid action - {error_reason}")
        
        # All retries exhausted
        self._consecutive_failures += 1
        logger.error(f"All {MAX_LLM_RETRIES} LLM attempts failed. Consecutive failures: {self._consecutive_failures}")
        
        # Use smart fallback instead of returning 'none'
        if self._consecutive_failures <= 3:
            return self._get_smart_fallback(violation_type, cluster_data)
        else:
            logger.error("Too many consecutive failures, returning no-action to avoid instability")
            return self._get_fallback_response(f"All {MAX_LLM_RETRIES} attempts failed: {last_error}")
    
    def _check_and_adjust_repeated_action(self, action: dict, history: str, violation_type: str, cluster_data: dict) -> dict:
        """
        Check if the LLM is repeating a failed action and suggest an alternative.
        
        This safety net helps overcome limitations of smaller LLMs that may not 
        learn from feedback in the history.
        """
        if action['action'] == 'none':
            return action
        
        # Count failures in history
        failed_count = history.count("WORSENED") + history.count("NO_CHANGE")
        
        # Track failures for the specific deployment
        same_deployment_failures = 0
        if action['action'] in ('horizontal_scaling', 'vertical_scaling'):
            dep_name = action['parameters'].get('deployment_name', '')
            if dep_name and dep_name in history:
                same_deployment_failures = history.count(dep_name)
        
        # If we see 3+ failures with horizontal_scaling, try vertical_scaling
        if failed_count >= 3 and action['action'] == 'horizontal_scaling' and self.actions_enabled.get('vertical_scaling', False):
            dep_name = action['parameters'].get('deployment_name', '')
            if not dep_name:
                return action
                
            logger.warning(f"Detected {failed_count} failed attempts. Switching to vertical_scaling.")
            
            if violation_type == "LOWER_THRESHOLD_EXCEEDED":
                return {
                    "action": "vertical_scaling",
                    "parameters": {
                        "deployment_name": dep_name,
                        "cpu_limit": "100m",
                        "memory_limit": "128Mi"
                    },
                    "adjusted_reason": "Switched from repeated failed horizontal_scaling"
                }
            else:
                return {
                    "action": "vertical_scaling",
                    "parameters": {
                        "deployment_name": dep_name,
                        "cpu_limit": "1000m",
                        "memory_limit": "1024Mi"
                    },
                    "adjusted_reason": "Switched from repeated failed horizontal_scaling"
                }
        
        # If same deployment failed 2+ times, try a different deployment
        if same_deployment_failures >= 2 and action['action'] == 'horizontal_scaling':
            try:
                deployments = cluster_data.get('deployments', {}).get('list', [])
                if not deployments:
                    deployments = cluster_data.get('data', {}).get('deployments', {}).get('list', [])
            except (AttributeError, TypeError):
                deployments = []
            
            current_dep = action['parameters'].get('deployment_name', '')
            
            for dep in deployments:
                if isinstance(dep, dict):
                    dep_name = dep.get('name', '')
                    replicas = dep.get('replicas_desired', 1)
                elif isinstance(dep, str):
                    dep_name = dep
                    replicas = 1
                else:
                    continue
                
                # Only consider valid deployments that aren't the current one
                if (dep_name and dep_name.lower() != current_dep.lower() 
                    and dep_name.lower() in self.valid_deployment_names):
                    if violation_type == "LOWER_THRESHOLD_EXCEEDED":
                        new_replicas = max(1, replicas - 1)
                    else:
                        new_replicas = replicas + 1
                    
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