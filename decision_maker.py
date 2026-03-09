"""
Decision Maker Module (Improved for Small LLMs like Qwen3.5:2b)

This module integrates with a local LLM via Ollama to analyze
system state and recommend actions when SLO violations occur.

Key optimizations for MoE / Small LLMs:
- Uses native XML tool-calling syntax instead of generic JSON.
- Removes history and bottleneck hints to test true reasoning and reduce context dilution.
- Bypasses Ollama's automatic tool wrappers to avoid schema confusion.
- Validation + retry loop (up to 3 attempts with corrective micro-prompts).
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
    def __init__(self, config: dict):
        self.config = config
        self.ollama_url = config["endpoints"]["ollama"]
        self.model = config["llm"]["model"]
        
        # Qwen-specific inference parameters for precise coding/tool tasks
        self.temperature = config["llm"].get("temperature", 0.6)
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

### REQUIRED FORMAT
You must respond ONLY with one of the following XML tags. Do not provide reasoning. Do not add text before or after the tag.

{available_targets}

### FINAL DECISION:
"""

    def _build_system_prompt(self) -> str:
        return """You are an intelligent Kubernetes resource manager responsible for maintaining application performance.
Your goal is to keep the application's average response time within a defined range.

When you decide on an action, you must output your decision strictly in this exact XML format and nothing else. Do not use JSON.
Example for scaling up:
<function=horizontal_scaling><parameter=deployment_name>microservice1-deployment</parameter><parameter=replicas>2</parameter></function>
Example for vertical scaling:
<function=vertical_scaling><parameter=deployment_name>microservice3-deployment</parameter><parameter=cpu_limit>600m</parameter><parameter=memory_limit>612Mi</parameter></function>

Do not add any reasoning or extra text before or after the XML tags. Just output the XML."""

    def _build_retry_prompt(self, previous_response: str, error_reason: str) -> str:
        valid_names = sorted(self.valid_deployment_names)
        names_str = ", ".join(valid_names) if valid_names else "microservice1-deployment"
        return f"""Your previous response was invalid: {error_reason}

Valid deployment names: {names_str}

Respond with ONLY the XML tags like one of these examples:
<function=horizontal_scaling><parameter=deployment_name>{valid_names[0] if valid_names else 'microservice1-deployment'}</parameter><parameter=replicas>2</parameter></function>
<function=vertical_scaling><parameter=deployment_name>{valid_names[0] if valid_names else 'microservice1-deployment'}</parameter><parameter=cpu_limit>500m</parameter><parameter=memory_limit>512Mi</parameter></function>"""

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

    @staticmethod
    def _expand_name(name: str) -> str:
        name = str(name).strip().lower()
        if name.endswith("-deployment") and "microservice" in name:
            return name
        match = re.match(r'^ms(\d+)$', name)
        if match: return f"microservice{match.group(1)}-deployment"
        match = re.match(r'^microservice(\d+)$', name)
        if match: return f"microservice{match.group(1)}-deployment"
        match = re.match(r'^microservice-(\d+)$', name)
        if match: return f"microservice{match.group(1)}-deployment"
        return name

    def _compute_available_actions(self, violation_type: str, cluster_data: dict) -> tuple:
        deployments = cluster_data.get("deployments", {}).get("list", [])
        if not deployments:
            deployments = cluster_data.get("data", {}).get("deployments", {}).get("list", [])
        
        valid_deps = [d for d in deployments if not self.valid_deployment_names or d.get("name", "").lower() in self.valid_deployment_names]
        targets = []
        
        if violation_type == "UPPER_THRESHOLD_EXCEEDED":
            direction = "INCREASE resources to reduce response time. Pick one action from the list below:"
            for d in valid_deps:
                name = d.get("name", "")
                cpu_val = int(str(d.get("cpu_limit", "300m")).replace("m", "").strip()) if str(d.get("cpu_limit", "300m")).replace("m", "").strip().isdigit() else 300
                replicas = d.get("replicas_ready", d.get("replicas_desired", 1))
                max_replicas = self._get_deployment_config(name).get("max_replicas", 5)
                
                if cpu_val < 1000:
                    new_cpu = min(cpu_val + 200, 1000)
                    mem_limit = d.get("memory_limit", "512Mi")
                    targets.append(f'- <function=vertical_scaling><parameter=deployment_name>{name}</parameter><parameter=cpu_limit>{new_cpu}m</parameter><parameter=memory_limit>{mem_limit}</parameter></function>')
                
                if replicas < max_replicas:
                    targets.append(f'- <function=horizontal_scaling><parameter=deployment_name>{name}</parameter><parameter=replicas>{replicas + 1}</parameter></function>')
        
        else:
            direction = "DECREASE resources to save costs. Pick one action from the list below:"
            for d in valid_deps:
                name = d.get("name", "")
                cpu_val = int(str(d.get("cpu_limit", "300m")).replace("m", "").strip()) if str(d.get("cpu_limit", "300m")).replace("m", "").strip().isdigit() else 300
                replicas = d.get("replicas_ready", d.get("replicas_desired", 1))
                
                if replicas > 1:
                    targets.append(f'- <function=horizontal_scaling><parameter=deployment_name>{name}</parameter><parameter=replicas>{replicas - 1}</parameter></function>')
                
                if cpu_val > 100:
                    new_cpu = max(cpu_val - 100, 100)
                    mem_limit = d.get("memory_limit", "312Mi")
                    targets.append(f'- <function=vertical_scaling><parameter=deployment_name>{name}</parameter><parameter=cpu_limit>{new_cpu}m</parameter><parameter=memory_limit>{mem_limit}</parameter></function>')
        
        if not targets:
            targets.append('<function=none></function>')
        
        return direction, "AVAILABLE ACTIONS:\n" + "\n".join(targets)

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
        
        # Note: bottleneck_hint and history have been explicitly removed to test LLM reasoning
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
        """Query LLM directly for XML without Ollama's forced JSON tool wrappers."""
        url = f"{self.ollama_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "presence_penalty": self.presence_penalty,
                "repeat_penalty": self.repeat_penalty,
                "num_predict": 512
            }
        }
        
        if self.debug_llm:
            logger.info("=" * 60)
            logger.info(f"LLM PROMPT:\n{user_prompt}")
            logger.info("=" * 60)
            
        try:
            logger.info(f"Querying Ollama ({self.model})...")
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return None

    def _parse_xml(self, text: str) -> dict:
        """Extracts native Qwen XML tool formats."""
        func_match = re.search(r'<function=([^>]+)>', text)
        if not func_match:
            return {}
            
        action = func_match.group(1).strip()
        params = {}
        
        param_matches = re.finditer(r'<parameter=([^>]+)>([^<]*)</parameter>', text)
        for match in param_matches:
            key = match.group(1).strip()
            val = match.group(2).strip()
            if val.isdigit():
                val = int(val)
            params[key] = val
            
        return {"action": action, "parameters": params}

    def _parse_response(self, response_text: str) -> dict:
        if not response_text:
            return self._get_fallback_response("No response from LLM")
        cleaned = response_text.strip()
        
        # Try 0: Qwen Native XML
        try:
            xml_parsed = self._parse_xml(cleaned)
            if xml_parsed.get("action"):
                return self._normalize_response(xml_parsed)
        except Exception as e:
            logger.debug(f"XML parse failed: {e}")

        # Fallback routines in case LLM outputs JSON despite XML instructions
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)
        cleaned = cleaned.strip()
        
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return self._normalize_response(parsed)
        except json.JSONDecodeError:
            pass
        
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
            
        return self._get_fallback_response("Could not parse LLM response into XML or JSON")

    def _validate_action(self, action: dict, cluster_data: dict = None) -> tuple:
        action_type = action.get("action", "none")
        params = action.get("parameters", {})
        
        if action_type == "none":
            return False, "No action recommended"
            
        valid_actions = {"horizontal_scaling", "vertical_scaling", "service_placement", "flow_scheduling"}
        if action_type not in valid_actions:
            return False, f"Unknown action type: {action_type}"
            
        if not self.actions_enabled.get(action_type, False):
            return False, f"Action '{action_type}' is not enabled"
            
        dep_name = params.get("deployment_name", "")
        if action_type in ("horizontal_scaling", "vertical_scaling", "service_placement"):
            if not dep_name:
                return False, "Missing deployment_name"
            if self.valid_deployment_names and dep_name.lower() not in self.valid_deployment_names:
                return False, f"Invalid deployment_name '{dep_name}'. Must be one of: {sorted(self.valid_deployment_names)}"
                
        dep_config = self._get_deployment_config(dep_name)
        max_replicas = dep_config.get("max_replicas", 5)
        min_replicas = dep_config.get("min_replicas", 1)
        current_state = self._get_current_deployment_state(dep_name, cluster_data)
        
        if action_type == "horizontal_scaling":
            replicas = params.get("replicas")
            if replicas is None:
                return False, "Missing replicas count"
            try:
                replicas = int(replicas)
            except (ValueError, TypeError):
                return False, f"Invalid replicas value: {replicas}"
                
            if replicas < min_replicas: replicas = min_replicas
            if replicas > max_replicas: replicas = max_replicas
            params["replicas"] = replicas
            
            current_replicas = current_state.get("replicas", 0)
            if replicas == current_replicas:
                return False, f"No change: {dep_name} already has {replicas} replicas."
                
        if action_type == "vertical_scaling":
            cpu = params.get("cpu_limit", "")
            mem = params.get("memory_limit", "")
            if not cpu:
                return False, "Missing cpu_limit"
                
            try:
                cpu_val = int(str(cpu).replace("m", "").strip())
                if cpu_val < 100: cpu_val = 100
                if cpu_val > 1000: cpu_val = 1000
                params["cpu_limit"] = f"{cpu_val}m"
            except (ValueError, TypeError):
                return False, f"Invalid cpu_limit value: {cpu}"
                
            if not mem:
                mem_val = max(128, (cpu_val // 100) * 100 + 12)
                params["memory_limit"] = f"{mem_val}Mi"
            else:
                try:
                    mem_val = int(str(mem).replace("Mi", "").replace("Gi", "000").strip())
                    if mem_val < 128: mem_val = 128
                    if mem_val > 1024: mem_val = 1024
                    params["memory_limit"] = f"{mem_val}Mi"
                except (ValueError, TypeError):
                    pass
                    
            current_cpu = current_state.get("cpu_limit_m", 0)
            if cpu_val == current_cpu:
                return False, f"No change: {dep_name} already has cpu_limit={cpu_val}m."
                
        return True, ""

    def _get_current_deployment_state(self, dep_name: str, cluster_data: dict = None) -> dict:
        if not cluster_data:
            cluster_data = self._last_cluster_data or {}
        
        deployments = cluster_data.get("deployments", {}).get("list", [])
        if not deployments:
            deployments = cluster_data.get("data", {}).get("deployments", {}).get("list", [])
        
        for d in deployments:
            if d.get("name", "").lower() == dep_name.lower():
                cpu_str = str(d.get("cpu_limit", "0m")).replace("m", "").strip()
                try:
                    cpu_m = int(cpu_str)
                except (ValueError, TypeError):
                    cpu_m = 0
                return {
                    "replicas": d.get("replicas_ready", d.get("replicas_desired", 0)),
                    "cpu_limit_m": cpu_m,
                    "memory_limit": d.get("memory_limit", "")
                }
        return {"replicas": 0, "cpu_limit_m": 0, "memory_limit": ""}

    def _get_deployment_config(self, dep_name: str) -> dict:
        for dep in self.config.get("kubernetes", {}).get("deployments", []):
            if dep.get("name", "").lower() == dep_name.lower():
                return dep
        return {"min_replicas": 1, "max_replicas": 5}

    def _normalize_response(self, parsed: dict) -> dict:
        action = str(parsed.get("action", "none")).lower().strip()
        
        action_mapping = {
            "horizontal_scaling": "horizontal_scaling",
            "scale": "horizontal_scaling",
            "scale_up": "horizontal_scaling",
            "scale_down": "horizontal_scaling",
            "vertical_scaling": "vertical_scaling",
            "resources": "vertical_scaling",
            "resize": "vertical_scaling",
            "none": "none"
        }
        normalized_action = action_mapping.get(action, "none")
        
        if normalized_action != "none" and not self.actions_enabled.get(normalized_action, False):
            return {"action": "none", "parameters": {}}
            
        params = parsed.get("parameters", {})
        if not isinstance(params, dict):
            params = {}
            
        parameters = {}
        if normalized_action == "horizontal_scaling":
            dep_name = params.get("deployment_name") or ""
            dep_name = self._expand_name(dep_name)
            replicas = params.get("replicas", 2)
            try:
                replicas = int(replicas)
            except (ValueError, TypeError):
                replicas = 2
            replicas = max(1, min(5, replicas))
            
            if dep_name:
                parameters = {"deployment_name": str(dep_name).lower(), "replicas": replicas}
            else:
                normalized_action = "none"
                
        elif normalized_action == "vertical_scaling":
            dep_name = params.get("deployment_name") or ""
            dep_name = self._expand_name(dep_name)
            cpu = str(params.get("cpu_limit", "500m"))
            mem = str(params.get("memory_limit", "512Mi"))
            if cpu.isdigit(): cpu = f"{cpu}m"
            if mem.isdigit(): mem = f"{mem}Mi"
            
            if dep_name:
                parameters = {"deployment_name": str(dep_name).lower(), "cpu_limit": cpu, "memory_limit": mem}
            else:
                normalized_action = "none"
                
        return {"action": normalized_action, "parameters": parameters}

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

    # Kept signature exact for compatibility with main.py & intent_watch_loop.py
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
                logger.info(f"LLM Response:\n{response_text}")
            
            action = self._parse_response(response_text)
            is_valid, error_reason = self._validate_action(action, cluster_data)
            
            if is_valid:
                self._consecutive_failures = 0
                logger.info(f"Valid XML response on attempt {attempt}: {action['action']} - {action['parameters']}")
                return action
            else:
                last_error = error_reason
                logger.warning(f"Attempt {attempt} invalid: {error_reason}")
                
        self._consecutive_failures += 1
        if self._consecutive_failures <= 3:
            return self._get_smart_fallback(violation_type, cluster_data)
        else:
            return self._get_fallback_response("All attempts failed")

    def _check_and_adjust_repeated_action(self, action: dict, history: str, violation_type: str, cluster_data: dict) -> dict:
        # Kept for backward compatibility if called externally, though removed from primary flow
        return action

    def is_healthy(self) -> bool:
        try:
            url = f"{self.ollama_url}/api/tags"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False