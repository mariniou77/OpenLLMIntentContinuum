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
        self.config = config
        self.ollama_url = config["endpoints"]["ollama"]
        self.model = config["llm"]["model"]
        
        # New Qwen-specific inference parameters
        self.temperature = config["llm"].get("temperature", 0.6)
        self.top_p = config["llm"].get("top_p", 0.95)
        self.presence_penalty = config["llm"].get("presence_penalty", 0.0)
        self.repeat_penalty = config["llm"].get("repeat_penalty", 1.05)
        
        self.debug_llm = config.get("debug_llm", False)
        
        # Load prompt template
        self.prompt_template = self._load_prompt_template()
        
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
        return """You are a Kubernetes resource manager. Pick ONE action to fix the problem.

PROBLEM: EMA Response Time is {ema_rt}s (target: {lower_threshold}s-{upper_threshold}s)
STATUS: {status}

RULE: {direction}

CURRENT STATE:
{deployments_table}
{bottleneck_hint}

{available_targets}

{history_section}

Pick the deployment that needs adjustment. You MUST output exactly ONE action using this XML format:
<function=ACTION_NAME><parameter=KEY>VALUE</parameter></function>
"""

    def _get_enabled_actions_description(self) -> str:
        """Get description of enabled actions for the prompt."""
        return """1. horizontal_scaling: Change replicas.
   {"action": "horizontal_scaling", "parameters": {"deployment_name": "X-deployment", "replicas": N}}

2. vertical_scaling: Change CPU/memory limits.
   {"action": "vertical_scaling", "parameters": {"deployment_name": "X-deployment", "cpu_limit": "Xm", "memory_limit": "XMi"}}"""

    def _format_system_state(self, cluster_data: dict, network_data: dict, monitoring_data: dict) -> str:
        """
        Format system state as a list for the prompt.
        
        Combines:
        - Kubernetes: deployment config (replicas, cpu_limit, memory_limit), pod-to-node mapping
        - sFlow-RT: real-time pod CPU %, memory usage, network traffic
        
        Returns format like:
        - microservice1-deployment (worker1): replicas=1, cpu=12.7%, cpu_limit=300m, mem=44Mi/312Mi, traffic_in=57KB/s
        """
        lines = []
        
        # Get deployments from K8s (replicas, limits)
        deployments = cluster_data.get("deployments", {}).get("list", [])
        if not deployments:
            deployments = cluster_data.get("data", {}).get("deployments", {}).get("list", [])
        
        # Get pods for node mapping
        pods = cluster_data.get("pods", {}).get("list", [])
        if not pods:
            pods = cluster_data.get("data", {}).get("pods", {}).get("list", [])
        
        # Build deployment -> node mapping from K8s pods
        dep_to_nodes = {}
        for pod in pods:
            pod_name = pod.get("name", "")
            node = pod.get("node", "unknown")
            # Skip pending pods (no node assigned)
            if not node or node == "None":
                continue
            for dep in deployments:
                dep_name = dep.get("name", "")
                if dep_name and pod_name.startswith(dep_name):
                    if dep_name not in dep_to_nodes:
                        dep_to_nodes[dep_name] = []
                    if node not in dep_to_nodes[dep_name]:
                        dep_to_nodes[dep_name].append(node)
        
        # Get sFlow pod metrics (real-time CPU, memory, traffic)
        sflow_pod_metrics = monitoring_data.get("pod_metrics", [])
        
        # Build deployment -> sFlow metrics mapping
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
            
            # Get node placement
            nodes = dep_to_nodes.get(name, ["unknown"])
            node_str = ", ".join(nodes)
            
            # Check if at limits for warnings
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
            
            # Get sFlow metrics for this deployment
            sflow_data = dep_to_sflow.get(name, [])
            if sflow_data:
                # Average across replicas
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
                # Fallback to kubectl top data if sFlow not available
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
        """
        Expand short deployment names back to full Kubernetes names.
        
        Handles: ms1, ms2, ms3, ms4, microservice1, etc.
        Returns the input unchanged if it's already a full name.
        """
        import re
        name = str(name).strip().lower()
        
        # Already a full name
        if name.endswith("-deployment") and "microservice" in name:
            return name
        
        # ms1 -> microservice1-deployment
        match = re.match(r'^ms(\d+)$', name)
        if match:
            return f"microservice{match.group(1)}-deployment"
        
        # microservice1 -> microservice1-deployment
        match = re.match(r'^microservice(\d+)$', name)
        if match:
            return f"microservice{match.group(1)}-deployment"
        
        # microservice-1 -> microservice1-deployment
        match = re.match(r'^microservice-(\d+)$', name)
        if match:
            return f"microservice{match.group(1)}-deployment"
        
        return name
        
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
        ema_rt: float,
        cluster_data: dict,
        network_data: dict,
        monitoring_data: dict,
    ) -> str:
        """
        Build the prompt with structured direction and pre-computed action targets.
        
        The LLM receives:
        - Rich monitoring data (sFlow + K8s)
        - Clear direction (INCREASE or DECREASE resources)
        - Pre-computed list of valid actions to choose from
        
        The LLM picks which target and action from the list.
        """
        # Status description
        if violation_type == "UPPER_THRESHOLD_EXCEEDED":
            status = f"TOO SLOW (above {self.upper_threshold}s). Response time needs to decrease."
        else:
            status = f"TOO FAST (below {self.lower_threshold}s). Resources are over-provisioned and being wasted."
        
        # Format deployments with node placement and sFlow metrics
        deployments_table = self._format_system_state(cluster_data, network_data, monitoring_data)
        
        # Format node-level metrics
        node_metrics = self._format_node_metrics(monitoring_data)
        
        # Pre-compute direction and available targets
        direction, available_targets = self._compute_available_actions(
            violation_type, cluster_data
        )
        
        # Fill in the template
        prompt = self.prompt_template.format(
            ema_rt=f"{ema_rt:.2f}",
            lower_threshold=self.lower_threshold,
            upper_threshold=self.upper_threshold,
            status=status,
            deployments_table=deployments_table,
            node_metrics=node_metrics,
            direction=direction,
            available_targets=available_targets,
        )
        
        return prompt
    
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
                    targets.append(f'- Increase CPU: <function=vertical_scaling><parameter=deployment_name>{name}</parameter><parameter=cpu_limit>{new_cpu}m</parameter><parameter=memory_limit>{mem_limit}</parameter></function>')
                
                if replicas < max_replicas:
                    targets.append(f'- Add replica: <function=horizontal_scaling><parameter=deployment_name>{name}</parameter><parameter=replicas>{replicas + 1}</parameter></function>')
        
        else:
            direction = "DECREASE resources to save costs. Pick one action from the list below:"
            for d in valid_deps:
                name = d.get("name", "")
                cpu_val = int(str(d.get("cpu_limit", "300m")).replace("m", "").strip()) if str(d.get("cpu_limit", "300m")).replace("m", "").strip().isdigit() else 300
                replicas = d.get("replicas_ready", d.get("replicas_desired", 1))
                
                if replicas > 1:
                    targets.append(f'- Remove replica: <function=horizontal_scaling><parameter=deployment_name>{name}</parameter><parameter=replicas>{replicas - 1}</parameter></function>')
                
                if cpu_val > 100:
                    new_cpu = max(cpu_val - 100, 100)
                    mem_limit = d.get("memory_limit", "312Mi")
                    targets.append(f'- Reduce CPU: <function=vertical_scaling><parameter=deployment_name>{name}</parameter><parameter=cpu_limit>{new_cpu}m</parameter><parameter=memory_limit>{mem_limit}</parameter></function>')
        
        if not targets:
            targets.append('<function=none></function>')
        
        return direction, "AVAILABLE ACTIONS:\n" + "\n".join(targets)
    
    def _format_node_metrics(self, monitoring_data: dict) -> str:
        """
        Format node-level metrics for the prompt using sFlow-RT data.
        
        Args:
            monitoring_data: Monitoring data dict from sFlow-RT
            
        Returns:
            Formatted string like:
            - master: CPU 12.8%, Memory 7.9%, Load 0.30
            - worker1: CPU 7.1%, Memory 8.1%, Load 0.44, SDN bandwidth in: 173KB/s
        """
        node_metrics = monitoring_data.get("node_metrics", {})
        sdn_bandwidth = monitoring_data.get("sdn_bandwidth", {})
        
        if not node_metrics:
            return "- No node metrics available"
        
        lines = []
        for name in sorted(node_metrics.keys()):
            m = node_metrics[name]
            parts = [
                f"CPU {m.get('cpu', 0):.1f}%",
                f"Memory {m.get('memory', 0):.1f}%",
                f"Load {m.get('load', 0):.2f}"
            ]
            
            # Add SDN bandwidth if available
            bw = sdn_bandwidth.get(name, {})
            if bw.get("bytes_in", 0) > 0:
                bw_in_kb = int(bw["bytes_in"] / 1024)
                parts.append(f"SDN in: {bw_in_kb}KB/s")
            
            lines.append(f"- {name}: {', '.join(parts)}")
        
        return '\n'.join(lines)
    
    def _compute_bottleneck_hint(self, cluster_data: dict, violation_type: str) -> str:
        """
        Identify the most likely bottleneck deployment and return a hint string.
        
        For UPPER violations: find the deployment with highest CPU usage ratio.
        For LOWER violations: find the deployment with most replicas or lowest usage.
        
        This guides the small LLM toward the right target instead of picking randomly.
        """
        deployments = cluster_data.get("deployments", {}).get("list", [])
        if not deployments:
            deployments = cluster_data.get("data", {}).get("deployments", {}).get("list", [])
        
        if not deployments:
            return ""
        
        # Filter to valid deployments
        valid_deps = []
        for d in deployments:
            name = d.get("name", "")
            if self.valid_deployment_names and name.lower() not in self.valid_deployment_names:
                continue
            valid_deps.append(d)
        
        if not valid_deps:
            return ""
        
        if violation_type == "UPPER_THRESHOLD_EXCEEDED":
            # Find deployment with highest CPU usage (the bottleneck)
            def cpu_usage_ratio(d):
                usage = d.get("cpu_usage", "0m")
                limit = d.get("cpu_limit", "1000m")
                try:
                    u = int(str(usage).replace("m", "").strip())
                except (ValueError, TypeError):
                    u = 0
                try:
                    l = int(str(limit).replace("m", "").strip())
                except (ValueError, TypeError):
                    l = 1000
                return u / max(l, 1)
            
            busiest = max(valid_deps, key=cpu_usage_ratio)
            ratio = cpu_usage_ratio(busiest)
            if ratio > 0.5:
                return f"BOTTLENECK: {busiest['name']} (cpu_usage={busiest.get('cpu_usage', '?')}, cpu_limit={busiest.get('cpu_limit', '?')}) - target this one"
            else:
                return ""
        else:
            # LOWER: find deployment with most replicas (can be scaled down)
            def replica_count(d):
                return d.get("replicas_ready", d.get("replicas_desired", 1))
            
            biggest = max(valid_deps, key=replica_count)
            reps = replica_count(biggest)
            if reps > 1:
                return f"OVER-PROVISIONED: {biggest['name']} has replicas={reps} - target this one"
            else:
                return ""
    
    def _build_retry_prompt(self, previous_response: str, error_reason: str) -> str:
        valid_names = sorted(self.valid_deployment_names)
        names_str = ", ".join(valid_names) if valid_names else "microservice1-deployment"
        return f"""Your previous response was invalid: {error_reason}

Valid deployment names: {names_str}

Respond with ONLY the XML tags like one of these examples:
<function=horizontal_scaling><parameter=deployment_name>{valid_names[0] if valid_names else 'microservice1-deployment'}</parameter><parameter=replicas>2</parameter></function>
<function=vertical_scaling><parameter=deployment_name>{valid_names[0] if valid_names else 'microservice1-deployment'}</parameter><parameter=cpu_limit>500m</parameter><parameter=memory_limit>512Mi</parameter></function>"""

    def _build_tool_definitions(self, violation_type: str, cluster_data: dict) -> list:
        """
        Build Ollama-compatible tool definitions based on current cluster state.
        
        For UPPER violations: only INCREASE tools (raise CPU, add replicas)
        For LOWER violations: only DECREASE tools (lower CPU, remove replicas)
        
        Returns:
            List of tool definition dicts for the Ollama /api/chat tools parameter
        """
        deployments = cluster_data.get("deployments", {}).get("list", [])
        if not deployments:
            deployments = cluster_data.get("data", {}).get("deployments", {}).get("list", [])
        
        tools = []
        
        for d in deployments:
            name = d.get("name", "")
            if self.valid_deployment_names and name.lower() not in self.valid_deployment_names:
                continue
            
            cpu_str = str(d.get("cpu_limit", "300m")).replace("m", "").strip()
            try:
                cpu_val = int(cpu_str)
            except (ValueError, TypeError):
                cpu_val = 300
            
            replicas = d.get("replicas_ready", d.get("replicas_desired", 1))
            dep_config = self._get_deployment_config(name)
            max_replicas = dep_config.get("max_replicas", 5)
            
            if violation_type == "UPPER_THRESHOLD_EXCEEDED":
                # Vertical scaling up (if not at max)
                if cpu_val < 1000:
                    new_cpu = min(cpu_val + 200, 1000)
                    mem_limit = d.get("memory_limit", "512Mi")
                    tools.append({
                        "type": "function",
                        "function": {
                            "name": "increase_cpu",
                            "description": f"Increase CPU limit for {name} from {cpu_val}m to {new_cpu}m. Use this when {name} has high CPU usage relative to its limit and is slowing down request processing.",
                            "parameters": {
                                "type": "object",
                                "required": ["deployment_name", "cpu_limit", "memory_limit"],
                                "properties": {
                                    "deployment_name": {"type": "string", "enum": [name], "description": f"The deployment to scale. Must be {name}."},
                                    "cpu_limit": {"type": "string", "enum": [f"{new_cpu}m"], "description": f"New CPU limit. Must be {new_cpu}m."},
                                    "memory_limit": {"type": "string", "enum": [mem_limit], "description": f"Memory limit. Must be {mem_limit}."}
                                }
                            }
                        }
                    })
                
                # Horizontal scaling up (if not at max)
                if replicas < max_replicas:
                    tools.append({
                        "type": "function",
                        "function": {
                            "name": "add_replica",
                            "description": f"Add one replica to {name}, going from {replicas} to {replicas + 1} replicas. Use this when the deployment needs to handle more concurrent requests and adding parallelism would reduce queuing delays.",
                            "parameters": {
                                "type": "object",
                                "required": ["deployment_name", "replicas"],
                                "properties": {
                                    "deployment_name": {"type": "string", "enum": [name], "description": f"The deployment to scale. Must be {name}."},
                                    "replicas": {"type": "integer", "enum": [replicas + 1], "description": f"New replica count. Must be {replicas + 1}."}
                                }
                            }
                        }
                    })
            
            else:  # LOWER_THRESHOLD_EXCEEDED
                # Scale down replicas (if more than 1)
                if replicas > 1:
                    tools.append({
                        "type": "function",
                        "function": {
                            "name": "remove_replica",
                            "description": f"Remove one replica from {name}, going from {replicas} to {replicas - 1} replicas. Use this when the deployment has more replicas than needed and resources are being wasted.",
                            "parameters": {
                                "type": "object",
                                "required": ["deployment_name", "replicas"],
                                "properties": {
                                    "deployment_name": {"type": "string", "enum": [name], "description": f"The deployment to scale. Must be {name}."},
                                    "replicas": {"type": "integer", "enum": [replicas - 1], "description": f"New replica count. Must be {replicas - 1}."}
                                }
                            }
                        }
                    })
                
                # Reduce CPU (if above minimum)
                if cpu_val > 100:
                    new_cpu = max(cpu_val - 100, 100)
                    mem_limit = d.get("memory_limit", "312Mi")
                    tools.append({
                        "type": "function",
                        "function": {
                            "name": "reduce_cpu",
                            "description": f"Reduce CPU limit for {name} from {cpu_val}m to {new_cpu}m. Use this when {name} has low CPU usage relative to its limit and is wasting resources.",
                            "parameters": {
                                "type": "object",
                                "required": ["deployment_name", "cpu_limit", "memory_limit"],
                                "properties": {
                                    "deployment_name": {"type": "string", "enum": [name], "description": f"The deployment to scale. Must be {name}."},
                                    "cpu_limit": {"type": "string", "enum": [f"{new_cpu}m"], "description": f"New CPU limit. Must be {new_cpu}m."},
                                    "memory_limit": {"type": "string", "enum": [mem_limit], "description": f"Memory limit. Must be {mem_limit}."}
                                }
                            }
                        }
                    })
        
        return tools
    
    def _build_system_prompt(self) -> str:
        return """You are an intelligent Kubernetes resource manager responsible for maintaining application performance.
Your goal is to keep the application's average response time within a defined range.

When you decide on an action, you must output your decision strictly in this exact XML format and nothing else. Do not use JSON.
Example for scaling up:
<function=horizontal_scaling><parameter=deployment_name>microservice1-deployment</parameter><parameter=replicas>2</parameter></function>
Example for vertical scaling:
<function=vertical_scaling><parameter=deployment_name>microservice3-deployment</parameter><parameter=cpu_limit>600m</parameter><parameter=memory_limit>612Mi</parameter></function>

Do not add any reasoning or extra text before or after the XML tags. Just output the XML."""
    
    def _query_ollama_with_tools(self, user_prompt: str, tools: list) -> Optional[dict]:
        """
        Send prompt to Ollama API using /api/chat with native tool calling.
        
        Instead of asking the model to generate JSON text, we define tools
        and let the model make a structured tool call. This leverages the
        model's trained tool-calling capabilities (BFCL-V4 score: 43.6).
        
        Args:
            user_prompt: The user message describing the current situation
            tools: List of tool definitions for Ollama
            
        Returns:
            Dict with tool call info, or None if failed
        """
        url = f"{self.ollama_url}/api/chat"
        
        system_prompt = self._build_system_prompt()
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "stream": False,
            "think": False,
            "tools": tools,
            "options": {
                "temperature": self.temperature,
                "num_predict": 512
            }
        }
        
        # Debug logging
        if self.debug_llm:
            logger.info("=" * 60)
            logger.info("LLM DEBUG - SYSTEM PROMPT:")
            logger.info("=" * 60)
            logger.info(f"\n{system_prompt[:500]}...")
            logger.info("=" * 60)
            logger.info("LLM DEBUG - USER PROMPT:")
            logger.info("=" * 60)
            logger.info(f"\n{user_prompt}")
            logger.info("=" * 60)
            logger.info(f"LLM DEBUG - TOOLS ({len(tools)} defined):")
            for t in tools:
                fname = t["function"]["name"]
                fdesc = t["function"]["description"][:80]
                logger.info(f"  - {fname}: {fdesc}...")
            logger.info("=" * 60)
        
        try:
            logger.info(f"Querying Ollama ({self.model}) with {len(tools)} tools...")
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            
            result = response.json()
            message = result.get("message", {})
            
            # Log timing info
            total_duration = result.get("total_duration", 0)
            if total_duration:
                logger.info(f"LLM response time: {total_duration / 1e9:.1f}s")
            
            # Check for tool calls (native Ollama tool calling response)
            tool_calls = message.get("tool_calls", [])
            text_content = message.get("content", "")
            
            if self.debug_llm:
                logger.info("LLM DEBUG - RESPONSE:")
                logger.info("=" * 60)
                if tool_calls:
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        logger.info(f"  Tool call: {fn.get('name', '?')}({fn.get('arguments', {})})")
                if text_content:
                    logger.info(f"  Text: {text_content[:200]}")
                logger.info("=" * 60)
            
            if tool_calls:
                return {"tool_calls": tool_calls, "content": text_content}
            elif text_content:
                # Model responded with text instead of tool call - try to parse as JSON
                logger.warning("Model returned text instead of tool call, attempting JSON parse")
                return {"content": text_content, "tool_calls": []}
            else:
                logger.warning("Empty response from model")
                return None
            
        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            return None
    
    def _tool_call_to_action(self, tool_call: dict) -> dict:
        """
        Convert an Ollama tool call response to our internal action format.
        
        Maps tool names:
            increase_cpu / reduce_cpu -> vertical_scaling
            add_replica / remove_replica -> horizontal_scaling
        """
        fn = tool_call.get("function", {})
        fn_name = fn.get("name", "")
        args = fn.get("arguments", {})
        
        if fn_name in ("increase_cpu", "reduce_cpu"):
            return {
                "action": "vertical_scaling",
                "parameters": {
                    "deployment_name": args.get("deployment_name", ""),
                    "cpu_limit": args.get("cpu_limit", ""),
                    "memory_limit": args.get("memory_limit", "")
                }
            }
        elif fn_name in ("add_replica", "remove_replica"):
            return {
                "action": "horizontal_scaling",
                "parameters": {
                    "deployment_name": args.get("deployment_name", ""),
                    "replicas": args.get("replicas", 1)
                }
            }
        else:
            logger.warning(f"Unknown tool call: {fn_name}")
            return {"action": "none", "parameters": {}}

    def _query_ollama(self, prompt: str) -> Optional[str]:
        """
        Legacy text-based query method. Kept for retry fallback.
        """
        url = f"{self.ollama_url}/api/chat"
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": self._build_system_prompt()
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": 256
            }
        }
        
        # Debug logging
        if self.debug_llm:
            logger.info("=" * 60)
            logger.info("LLM DEBUG - PROMPT (text fallback):")
            logger.info("=" * 60)
            logger.info(f"\n{prompt}")
            logger.info("=" * 60)
        
        try:
            logger.info(f"Querying Ollama ({self.model})...")
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            
            result = response.json()
            llm_response = result.get("message", {}).get("content", "")
            
            total_duration = result.get("total_duration", 0)
            if total_duration:
                logger.info(f"LLM response time: {total_duration / 1e9:.1f}s")
            
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
    
    def _validate_action(self, action: dict, cluster_data: dict = None) -> tuple:
        """
        Validate a parsed action against known constraints.
        Rejects actions that would set the same values as currently applied.
        
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
        
        # Get deployment-specific limits from config
        dep_config = self._get_deployment_config(dep_name)
        max_replicas = dep_config.get("max_replicas", 5)
        min_replicas = dep_config.get("min_replicas", 1)
        
        # Get current state for no-change detection
        current_state = self._get_current_deployment_state(dep_name, cluster_data)
        
        # Validate horizontal_scaling parameters with clamping
        if action_type == "horizontal_scaling":
            replicas = params.get("replicas")
            if replicas is None:
                return False, "Missing replicas count"
            try:
                replicas = int(replicas)
            except (ValueError, TypeError):
                return False, f"Invalid replicas value: {replicas}"
            # Clamp to configured limits
            if replicas < min_replicas:
                logger.info(f"Clamping replicas from {replicas} to min={min_replicas}")
                replicas = min_replicas
            if replicas > max_replicas:
                logger.info(f"Clamping replicas from {replicas} to max={max_replicas}")
                replicas = max_replicas
            params["replicas"] = replicas
            
            # No-change detection
            current_replicas = current_state.get("replicas", 0)
            if replicas == current_replicas:
                return False, f"No change: {dep_name} already has {replicas} replicas. Try a different deployment or action type."
        
        # Validate vertical_scaling parameters with clamping
        if action_type == "vertical_scaling":
            cpu = params.get("cpu_limit", "")
            mem = params.get("memory_limit", "")
            if not cpu:
                return False, "Missing cpu_limit"
            
            # Parse and clamp CPU limit (100m - 1000m)
            try:
                cpu_val = int(str(cpu).replace("m", "").strip())
                if cpu_val < 100:
                    logger.info(f"Clamping cpu_limit from {cpu_val}m to min=100m")
                    cpu_val = 100
                if cpu_val > 1000:
                    logger.info(f"Clamping cpu_limit from {cpu_val}m to max=1000m")
                    cpu_val = 1000
                params["cpu_limit"] = f"{cpu_val}m"
            except (ValueError, TypeError):
                return False, f"Invalid cpu_limit value: {cpu}"
            
            # Auto-fill memory_limit if missing
            if not mem:
                mem_val = max(128, (cpu_val // 100) * 100 + 12)
                params["memory_limit"] = f"{mem_val}Mi"
                logger.info(f"Auto-filled missing memory_limit: {params['memory_limit']} (based on cpu_limit={cpu_val}m)")
            else:
                # Clamp memory (128Mi - 1024Mi)
                try:
                    mem_val = int(str(mem).replace("Mi", "").replace("Gi", "000").strip())
                    if mem_val < 128:
                        mem_val = 128
                    if mem_val > 1024:
                        mem_val = 1024
                    params["memory_limit"] = f"{mem_val}Mi"
                except (ValueError, TypeError):
                    pass  # Keep original if can't parse
            
            # No-change detection
            current_cpu = current_state.get("cpu_limit_m", 0)
            if cpu_val == current_cpu:
                return False, f"No change: {dep_name} already has cpu_limit={cpu_val}m. Try horizontal_scaling or a different deployment."
        
        return True, ""
    
    def _get_current_deployment_state(self, dep_name: str, cluster_data: dict = None) -> dict:
        """Get current replicas and cpu_limit for a deployment from cluster data."""
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
        """Get deployment-specific config (limits etc) from config.yaml."""
        for dep in self.config.get("kubernetes", {}).get("deployments", []):
            if dep.get("name", "").lower() == dep_name.lower():
                return dep
        return {"min_replicas": 1, "max_replicas": 5}
    
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
            "increase_replicas": "horizontal_scaling",
            "decrease_replicas": "horizontal_scaling",
            "add_replicas": "horizontal_scaling",
            "remove_replicas": "horizontal_scaling",
            "replicas": "horizontal_scaling",
            "vertical_scaling": "vertical_scaling",
            "verticalscaling": "vertical_scaling",
            "resources": "vertical_scaling",
            "resize": "vertical_scaling",
            "increase_cpu": "vertical_scaling",
            "decrease_cpu": "vertical_scaling",
            "cpu_scaling": "vertical_scaling",
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
            dep_name = self._expand_name(dep_name)
            
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
            dep_name = self._expand_name(dep_name)
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
            dep_name = self._expand_name(dep_name)
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
        
        # Store for no-change detection
        self._last_cluster_data = cluster_data
        
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
        
        # Build tool definitions based on violation type and current state
        tools = self._build_tool_definitions(violation_type, cluster_data)
        logger.info(f"Built {len(tools)} tool definitions for {violation_type}")
        
        # === PRIMARY PATH: Tool calling ===
        last_error = ""
        for attempt in range(1, MAX_LLM_RETRIES + 1):
            if attempt == 1:
                current_prompt = prompt
            else:
                logger.warning(f"Retry {attempt}/{MAX_LLM_RETRIES}: {last_error}")
                current_prompt = prompt + f"\n\nIMPORTANT: Your previous attempt failed. Reason: {last_error}. Please call one of the available tools now."
            
            result = self._query_ollama_with_tools(current_prompt, tools)
            
            if not result:
                last_error = "No response from LLM"
                continue
            
            # Check for tool calls
            tool_calls = result.get("tool_calls", [])
            
            if tool_calls:
                # Convert tool call to our action format
                action = self._tool_call_to_action(tool_calls[0])
                
                # Validate
                is_valid, error_reason = self._validate_action(action, cluster_data)
                
                if is_valid:
                    self._consecutive_failures = 0
                    logger.info(f"Valid tool call on attempt {attempt}: {action['action']}")
                    logger.info(f"Parameters: {action['parameters']}")
                    return action
                else:
                    last_error = error_reason
                    logger.warning(f"Attempt {attempt}: tool call invalid - {error_reason}")
            else:
                # Model returned text instead of tool call - try JSON parse fallback
                text_content = result.get("content", "")
                if text_content:
                    action = self._parse_response(text_content)
                    is_valid, error_reason = self._validate_action(action, cluster_data)
                    if is_valid:
                        self._consecutive_failures = 0
                        logger.info(f"Valid text response on attempt {attempt}: {action['action']}")
                        logger.info(f"Parameters: {action['parameters']}")
                        return action
                    else:
                        last_error = f"Text fallback also invalid: {error_reason}"
                        logger.warning(f"Attempt {attempt}: {last_error}")
                else:
                    last_error = "No tool call and no text in response"
        
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
        
        NOTE: Only WORSENED outcomes count as failures. NO_CHANGE outcomes are
        not counted because with low EMA alpha, even successful actions may
        appear as NO_CHANGE before EMA catches up.
        """
        if action['action'] == 'none':
            return action
        
        # Count only genuine failures (WORSENED), not NO_CHANGE
        # NO_CHANGE with low alpha may just mean EMA hasn't caught up yet
        failed_count = history.count("WORSENED")
        
        # Track failures for the specific deployment
        same_deployment_failures = 0
        if action['action'] in ('horizontal_scaling', 'vertical_scaling'):
            dep_name = action['parameters'].get('deployment_name', '')
            if dep_name and dep_name in history:
                # Count only WORSENED lines that mention this deployment
                for line in history.split('\n'):
                    if dep_name in line and 'WORSENED' in line:
                        same_deployment_failures += 1
        
        # If we see 3+ WORSENED failures with horizontal_scaling, try vertical_scaling
        if failed_count >= 3 and action['action'] == 'horizontal_scaling' and self.actions_enabled.get('vertical_scaling', False):
            dep_name = action['parameters'].get('deployment_name', '')
            if not dep_name:
                return action
                
            logger.warning(f"Detected {failed_count} WORSENED attempts. Switching to vertical_scaling.")
            
            # Get current limits for this deployment to make a proportional change
            current_cpu = self._get_current_cpu_limit(dep_name, cluster_data)
            
            if violation_type == "LOWER_THRESHOLD_EXCEEDED":
                # Reduce by ~30% but not below 100m
                new_cpu = max(100, int(current_cpu * 0.7))
                new_mem = max(128, new_cpu + 12)
                return {
                    "action": "vertical_scaling",
                    "parameters": {
                        "deployment_name": dep_name,
                        "cpu_limit": f"{new_cpu}m",
                        "memory_limit": f"{new_mem}Mi"
                    },
                    "adjusted_reason": "Switched from repeated WORSENED horizontal_scaling"
                }
            else:
                # Increase by ~30% but not above 1000m
                new_cpu = min(1000, int(current_cpu * 1.3))
                new_mem = min(1024, new_cpu + 12)
                return {
                    "action": "vertical_scaling",
                    "parameters": {
                        "deployment_name": dep_name,
                        "cpu_limit": f"{new_cpu}m",
                        "memory_limit": f"{new_mem}Mi"
                    },
                    "adjusted_reason": "Switched from repeated WORSENED horizontal_scaling"
                }
        
        # If same deployment WORSENED 2+ times, try a different deployment
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
                    
                    logger.warning(f"Same deployment WORSENED {same_deployment_failures} times. Trying {dep_name} instead.")
                    return {
                        "action": "horizontal_scaling",
                        "parameters": {
                            "deployment_name": dep_name,
                            "replicas": new_replicas
                        },
                        "adjusted_reason": f"Switched from {current_dep} due to repeated WORSENED outcomes"
                    }
        
        return action
    
    def _get_current_cpu_limit(self, deployment_name: str, cluster_data: dict) -> int:
        """Get current CPU limit in millicores for a deployment."""
        deployments = cluster_data.get("deployments", {}).get("list", [])
        if not deployments:
            deployments = cluster_data.get("data", {}).get("deployments", {}).get("list", [])
        
        for d in deployments:
            if d.get("name", "").lower() == deployment_name.lower():
                cpu_str = str(d.get("cpu_limit", "300m")).replace("m", "").strip()
                try:
                    return int(cpu_str)
                except ValueError:
                    return 300
        return 300
    
    def is_healthy(self) -> bool:
        """Check if Ollama is responding."""
        try:
            url = f"{self.ollama_url}/api/tags"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False