"""
Decision Maker Module (Optimized for Small MoE LLMs)

This module integrates with a local LLM via Ollama to analyze
system state and recommend actions when SLO violations occur.

Key optimizations for Qwen3.5:2b:
- Uses a Multiple-Choice Question (MCQ) format to bypass generation loops.
- `think: False` and `temperature: 0.0` for deterministic, lightning-fast inference (< 12s).
- Pre-calculates exact scaling bounds to prevent LLM hallucinations.
"""

import logging
import re
import requests
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)

# Maximum number of LLM query attempts before falling back
MAX_LLM_RETRIES = 3

class DecisionMaker:
    def __init__(self, config: dict):
        self.config = config
        self.ollama_url = config["endpoints"]["ollama"]
        self.model = config["llm"]["model"]
        
        # Core sampling parameters for precise MCQ
        self.temperature = 0.0
        self.top_p = config["llm"].get("top_p", 0.95)
        self.presence_penalty = config["llm"].get("presence_penalty", 0.0)
        self.repeat_penalty = config["llm"].get("repeat_penalty", 1.05)
        
        self.debug_llm = config.get("debug_llm", False)
        
        # Intent thresholds for context
        self.upper_threshold = config["intent"]["upper_threshold"]
        self.lower_threshold = config["intent"]["lower_threshold"]
        
        # Enabled actions
        self.actions_enabled = config.get("actions", {})
        
        # Build valid deployment names set from config
        self.valid_deployment_names = set()
        k8s_config = config.get("kubernetes", {})
        for dep in k8s_config.get("deployments", []):
            dep_name = dep.get("name", "")
            if dep_name:
                self.valid_deployment_names.add(dep_name.lower())
        
        self._consecutive_failures = 0
        self._last_cluster_data = {}
        self._current_action_mapping = {}
        
        # Load prompt template (must happen after thresholds are initialized)
        self.prompt_template = self._load_prompt_template()

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
        return """### ROLE
You are a Kubernetes Resource Orchestrator. Your sole objective is to maintain the Service Level Objective (SLO) for an image processing pipeline.

### TASK
Analyze the current performance violation and select exactly ONE action to return the system to a stable state.

### SYSTEM METRICS
- EMA Response Time: {ema_rt}s
- Target Range: {lower_threshold}s to {upper_threshold}s
- Status: {status}

### OPERATIONAL RULE
{direction}

### CLUSTER STATE
{deployments_table}

### AVAILABLE ACTIONS
Which action is best to fix the issue?
{available_targets}

Please show your choice by outputting ONLY the single choice letter (e.g., A, B, C).
### FINAL DECISION:
"""

    def _build_system_prompt(self) -> str:
        return """You are an intelligent Kubernetes resource manager. Analyze the cluster state and choose the best action to fix the performance violation. You must output ONLY the single letter corresponding to your choice (e.g., A, B, C). Do not output any internal thoughts, reasoning, or XML."""

    def _build_retry_prompt(self, previous_response: str, error_reason: str) -> str:
        return f"""Your previous response was invalid: {error_reason}

You MUST output ONLY a single valid letter from the options provided (e.g., A, B, or C). Do not write any other words."""

    def _format_system_state(self, cluster_data: dict, network_data: dict, monitoring_data: dict) -> str:
        lines = []
        deployments = cluster_data.get("deployments", {}).get("list", [])
        if not deployments:
            deployments = cluster_data.get("data", {}).get("deployments", {}).get("list", [])
        
        pods = cluster_data.get("pods", {}).get("list", [])
        if not pods:
            pods = cluster_data.get("data", {}).get("pods", {}).get("list", [])
        
        dep_to_nodes = {}
        for pod in pods:
            pod_name = pod.get("name", "")
            node = pod.get("node", "unknown")
            if not node or node == "None":
                continue
            for dep in deployments:
                dep_name = dep.get("name", "")
                if dep_name and pod_name.startswith(dep_name):
                    if dep_name not in dep_to_nodes:
                        dep_to_nodes[dep_name] = []
                    if node not in dep_to_nodes[dep_name]:
                        dep_to_nodes[dep_name].append(node)
        
        sflow_pod_metrics = monitoring_data.get("pod_metrics", [])
        dep_to_sflow = {}
        for pm in sflow_pod_metrics:
            dep_name = pm.get("deployment", "")
            if dep_name:
                if dep_name not in dep_to_sflow:
                    dep_to_sflow[dep_name] = []
                dep_to_sflow[dep_name].append(pm)
        
        for d in deployments:
            name = d.get("name", "unknown")
            if self.valid_deployment_names and name.lower() not in self.valid_deployment_names:
                continue
            
            replicas = d.get("replicas_ready", d.get("replicas_desired", 0))
            cpu_limit = d.get("cpu_limit") or "unknown"
            memory_limit = d.get("memory_limit") or "unknown"
            
            nodes = dep_to_nodes.get(name, ["unknown"])
            node_str = ", ".join(nodes)
            
            warnings = []
            try:
                cpu_limit_val = int(str(cpu_limit).replace("m", "").strip())
                if cpu_limit_val >= 1000:
                    warnings.append("CPU AT MAX")
            except (ValueError, TypeError):
                pass
            
            dep_config = self._get_deployment_config(name)
            max_reps = dep_config.get("max_replicas", 5)
            if replicas >= max_reps:
                warnings.append("REPLICAS AT MAX")
            
            warning_str = f" ⚠ {', '.join(warnings)}" if warnings else ""
            
            sflow_data = dep_to_sflow.get(name, [])
            if sflow_data:
                avg_cpu = sum(p["cpu_percent"] for p in sflow_data) / len(sflow_data)
                avg_mem_used = sum(p["mem_used_bytes"] for p in sflow_data) / len(sflow_data)
                avg_bytes_in = sum(p["bytes_in"] for p in sflow_data) / len(sflow_data)
                
                mem_used_mi = int(avg_mem_used / (1024 * 1024))
                traffic_in_kb = int(avg_bytes_in / 1024)
                
                lines.append(
                    f"- {name} ({node_str}): replicas={replicas}, "
                    f"cpu={avg_cpu:.1f}%, cpu_limit={cpu_limit}, "
                    f"mem={mem_used_mi}Mi/{memory_limit}, "
                    f"traffic_in={traffic_in_kb}KB/s{warning_str}"
                )
            else:
                cpu_usage = d.get("cpu_usage") or "0m"
                memory_usage = d.get("memory_usage") or "0Mi"
                lines.append(
                    f"- {name} ({node_str}): replicas={replicas}, "
                    f"cpu_usage={cpu_usage}, cpu_limit={cpu_limit}, "
                    f"mem={memory_usage}/{memory_limit}{warning_str}"
                )
        
        return '\n'.join(lines) if lines else "No deployment data available"

    def _get_deployment_config(self, dep_name: str) -> dict:
        for dep in self.config.get("kubernetes", {}).get("deployments", []):
            if dep.get("name", "").lower() == dep_name.lower():
                return dep
        return {"min_replicas": 1, "max_replicas": 5}

    def _compute_available_actions(self, violation_type: str, cluster_data: dict) -> tuple:
        """Generates dynamic Multiple-Choice options based on valid actions."""
        deployments = cluster_data.get("deployments", {}).get("list", [])
        if not deployments:
            deployments = cluster_data.get("data", {}).get("deployments", {}).get("list", [])
        
        valid_deps = [d for d in deployments if not self.valid_deployment_names or d.get("name", "").lower() in self.valid_deployment_names]
        
        mcq_options = []
        action_mapping = {}  
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        opt_idx = 0
        
        if violation_type == "UPPER_THRESHOLD_EXCEEDED":
            direction = "INCREASE resources to reduce response time."
            for d in valid_deps:
                name = d.get("name", "")
                cpu_val = int(str(d.get("cpu_limit", "300m")).replace("m", "").strip()) if str(d.get("cpu_limit", "300m")).replace("m", "").strip().isdigit() else 300
                replicas = d.get("replicas_ready", d.get("replicas_desired", 1))
                max_replicas = self._get_deployment_config(name).get("max_replicas", 5)
                
                # Option: Vertical Scale Up
                if cpu_val < 1000 and self.actions_enabled.get("vertical_scaling", False):
                    new_cpu = min(cpu_val + 200, 1000)
                    mem_limit = d.get("memory_limit", "512Mi")
                    letter = alphabet[opt_idx]
                    mcq_options.append(f"{letter}) Increase CPU limit of {name} to {new_cpu}m")
                    action_mapping[letter] = {
                        "action": "vertical_scaling", 
                        "parameters": {"deployment_name": name, "cpu_limit": f"{new_cpu}m", "memory_limit": mem_limit}
                    }
                    opt_idx += 1
                
                # Option: Horizontal Scale Up
                if replicas < max_replicas and self.actions_enabled.get("horizontal_scaling", False):
                    letter = alphabet[opt_idx]
                    mcq_options.append(f"{letter}) Scale up {name} to {replicas + 1} replicas")
                    action_mapping[letter] = {
                        "action": "horizontal_scaling", 
                        "parameters": {"deployment_name": name, "replicas": replicas + 1}
                    }
                    opt_idx += 1
        else:
            direction = "DECREASE resources to save costs."
            for d in valid_deps:
                name = d.get("name", "")
                cpu_val = int(str(d.get("cpu_limit", "300m")).replace("m", "").strip()) if str(d.get("cpu_limit", "300m")).replace("m", "").strip().isdigit() else 300
                replicas = d.get("replicas_ready", d.get("replicas_desired", 1))
                
                # Option: Horizontal Scale Down
                if replicas > 1 and self.actions_enabled.get("horizontal_scaling", False):
                    letter = alphabet[opt_idx]
                    mcq_options.append(f"{letter}) Scale down {name} to {replicas - 1} replicas")
                    action_mapping[letter] = {
                        "action": "horizontal_scaling", 
                        "parameters": {"deployment_name": name, "replicas": replicas - 1}
                    }
                    opt_idx += 1
                
                # Option: Vertical Scale Down
                if cpu_val > 100 and self.actions_enabled.get("vertical_scaling", False):
                    new_cpu = max(cpu_val - 100, 100)
                    mem_limit = d.get("memory_limit", "312Mi")
                    letter = alphabet[opt_idx]
                    mcq_options.append(f"{letter}) Reduce CPU limit of {name} to {new_cpu}m")
                    action_mapping[letter] = {
                        "action": "vertical_scaling", 
                        "parameters": {"deployment_name": name, "cpu_limit": f"{new_cpu}m", "memory_limit": mem_limit}
                    }
                    opt_idx += 1
        
        if not mcq_options:
            mcq_options.append("A) No action available")
            action_mapping["A"] = {"action": "none", "parameters": {}}
        
        self._current_action_mapping = action_mapping # Store mapping for parsing phase
        return direction, "\n".join(mcq_options)

    def build_prompt(
        self,
        violation_type: str,
        current_rt: float,
        ema_rt: float,
        cluster_data: dict,
        network_data: dict,
        monitoring_data: dict,
    ) -> str:
        if violation_type == "UPPER_THRESHOLD_EXCEEDED":
            status = f"TOO SLOW (above {self.upper_threshold}s). Response time needs to decrease."
        else:
            status = f"TOO FAST (below {self.lower_threshold}s). Resources are over-provisioned and being wasted."
        
        deployments_table = self._format_system_state(cluster_data, network_data, monitoring_data)
        direction, available_targets = self._compute_available_actions(violation_type, cluster_data)
        
        prompt = self.prompt_template.format(
            ema_rt=f"{ema_rt:.2f}",
            lower_threshold=self.lower_threshold,
            upper_threshold=self.upper_threshold,
            status=status,
            deployments_table=deployments_table,
            direction=direction,
            available_targets=available_targets,
        )
        return prompt

    def _query_llm(self, user_prompt: str) -> Optional[str]:
        """Query LLM directly for the single MCQ letter."""
        url = f"{self.ollama_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "think": False, # CRITICAL: Disables MoE endless thinking loops
            "options": {
                "temperature": 0.0, # CRITICAL: Forces deterministic, immediate letter output
                "top_p": self.top_p,
                "presence_penalty": self.presence_penalty,
                "repeat_penalty": self.repeat_penalty,
                "num_predict": 8  # Only needs enough room for a single letter
            }
        }
        
        if self.debug_llm:
            logger.info("=" * 60)
            logger.info(f"LLM PROMPT:\n{user_prompt}")
            logger.info("=" * 60)
            
        try:
            logger.info(f"Querying Ollama ({self.model})...")
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return None

    def _parse_response(self, response_text: str) -> dict:
        """Extracts the single letter choice from the LLM and maps it to the real action."""
        if not response_text:
            return self._get_fallback_response("No response from LLM")
            
        cleaned = response_text.strip().upper()
        
        # Regex to find a single standalone letter A-Z
        match = re.search(r'\b([A-Z])\b', cleaned)
        if match:
            letter = match.group(1)
            # Look up the real action dict in our stored mapping
            if hasattr(self, '_current_action_mapping') and letter in self._current_action_mapping:
                return self._current_action_mapping[letter]
                
        return self._get_fallback_response(f"Could not extract a valid choice letter from LLM output: {cleaned}")

    def _get_fallback_response(self, reason: str) -> dict:
        logger.warning(f"Using fallback response: {reason}")
        return {"action": "none", "parameters": {}, "fallback_reason": reason}

    def _get_smart_fallback(self, violation_type: str, cluster_data: dict) -> dict:
        deployments = cluster_data.get("deployments", {}).get("list", [])
        if not deployments:
            deployments = cluster_data.get("data", {}).get("deployments", {}).get("list", [])
            
        valid_deps = [d for d in deployments if not self.valid_deployment_names or d.get("name", "").lower() in self.valid_deployment_names]
        if not valid_deps:
            return self._get_fallback_response("No valid deployments found for smart fallback")
            
        if violation_type == "UPPER_THRESHOLD_EXCEEDED":
            best = min(valid_deps, key=lambda d: d.get("replicas_ready", d.get("replicas_desired", 1)))
            current_replicas = best.get("replicas_ready", best.get("replicas_desired", 1))
            new_replicas = min(5, current_replicas + 1)
            logger.warning(f"Smart fallback: scaling up {best['name']} from {current_replicas} to {new_replicas}")
            return {"action": "horizontal_scaling", "parameters": {"deployment_name": best["name"], "replicas": new_replicas}}
        else:
            candidates = [d for d in valid_deps if d.get("replicas_ready", d.get("replicas_desired", 1)) > 1]
            if candidates:
                best = max(candidates, key=lambda d: d.get("replicas_ready", d.get("replicas_desired", 1)))
                current_replicas = best.get("replicas_ready", best.get("replicas_desired", 1))
                new_replicas = max(1, current_replicas - 1)
                logger.warning(f"Smart fallback: scaling down {best['name']} from {current_replicas} to {new_replicas}")
                return {"action": "horizontal_scaling", "parameters": {"deployment_name": best["name"], "replicas": new_replicas}}
            return self._get_fallback_response("All deployments at minimum replicas")

    def analyze_and_recommend(
        self,
        violation_type: str,
        current_rt: float,
        ema_rt: float,
        cluster_data: dict = None,
        network_data: dict = None,
        monitoring_data: dict = None,
        history: str = "",
        **kwargs
    ) -> dict:
        logger.info(f"Analyzing {violation_type} violation... RT: {current_rt:.2f}s")
        cluster_data = cluster_data or {}
        self._last_cluster_data = cluster_data
        
        prompt = self.build_prompt(violation_type, current_rt, ema_rt, cluster_data, network_data or {}, monitoring_data or {})
        
        last_error = ""
        for attempt in range(1, MAX_LLM_RETRIES + 1):
            if attempt == 1:
                current_prompt = prompt
            else:
                current_prompt = prompt + f"\n\nIMPORTANT: Previous attempt failed ({last_error}).\n" + self._build_retry_prompt("", last_error)
            
            response_text = self._query_llm(current_prompt)
            if not response_text:
                last_error = "No response from LLM"
                continue
                
            if self.debug_llm:
                logger.info(f"LLM Raw Output:\n{response_text}")
            
            action = self._parse_response(response_text)
            
            # Since the options were pre-computed and valid, if we got a real action, it's valid.
            if action.get("action") != "none":
                self._consecutive_failures = 0
                logger.info(f"Valid MCQ decision on attempt {attempt}: {action['action']} - {action['parameters']}")
                return action
            else:
                last_error = action.get("fallback_reason", "Invalid letter chosen")
                logger.warning(f"Attempt {attempt} invalid: {last_error}")
                
        self._consecutive_failures += 1
        if self._consecutive_failures <= 3:
            return self._get_smart_fallback(violation_type, cluster_data)
        else:
            return self._get_fallback_response("All attempts failed")

    def _check_and_adjust_repeated_action(self, action: dict, history: str, violation_type: str, cluster_data: dict) -> dict:
        return action

    def is_healthy(self) -> bool:
        try:
            url = f"{self.ollama_url}/api/tags"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False