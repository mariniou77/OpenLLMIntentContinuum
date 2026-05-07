"""
Decision Maker Module (8-Message Accumulated History Format)

This module integrates with a local LLM (qwen3.5:4b via Ollama) to analyze
system state and recommend actions when SLO violations occur.

Uses the 8-message conversation structure tested at 90% accuracy:
  Msg 1 (system):    12-rule policy prompt
  Msg 2 (user):      History of recent violations (rolling window)
  Msg 3 (assistant):  "Understood."
  Msg 4 (user):      Intent + application critical path
  Msg 5 (assistant):  "Understood."
  Msg 6 (user):      Services + nodes + network metrics
  Msg 7 (assistant):  "Understood."
  Msg 8 (user):      Candidate actions + "Select the best action."

Supports 6 action types:
1. increase_cpu      → vertical_scaling (executor)
2. reduce_cpu        → vertical_scaling (executor)
3. add_replica       → horizontal_scaling (executor)
4. remove_replica    → horizontal_scaling (executor)
5. service_placement → service_placement (executor)
6. flow_scheduling   → flow_scheduling (executor)
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

# Bridge message used between user turns
ASSISTANT_BRIDGE = "Understood."


class DecisionMaker:
    """
    LLM-powered decision maker using 8-message accumulated history format.
    
    Uses the conversation structure validated at 90% accuracy in the
    6-action test suite with qwen3.5:4b.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.ollama_url = config["endpoints"]["ollama"]
        self.model = config["llm"]["model"]
        self.temperature = config["llm"]["temperature"]
        self.debug_llm = config.get("debug_llm", False)
        
        # Load 8-message system prompt
        self.system_prompt = self._load_system_prompt()
        
        # Legacy prompt template (kept for backward compat)
        self.prompt_template = self._load_prompt_template()
        
        # Intent thresholds
        self.upper_threshold = config["intent"]["upper_threshold"]
        self.lower_threshold = config["intent"]["lower_threshold"]
        
        # Enabled actions
        self.actions_enabled = config.get("actions", {})
        
        # Valid deployment names from config
        self.valid_deployment_names = set()
        k8s_config = config.get("kubernetes", {})
        for dep in k8s_config.get("deployments", []):
            dep_name = dep.get("name", "")
            if dep_name:
                self.valid_deployment_names.add(dep_name.lower())
        
        # Track consecutive parse failures
        self._consecutive_failures = 0
    
    def _load_system_prompt(self) -> str:
        """Load the 6-action system prompt from file."""
        prompt_path = Path(__file__).parent / "prompts" / "system_prompt_6action.txt"
        try:
            with open(prompt_path, "r") as f:
                return f.read()
        except FileNotFoundError:
            logger.warning(f"System prompt not found at {prompt_path}, using embedded default")
            return "You are an orchestration policy selector. Choose the best action from candidate_actions. Return JSON: {\"answer\":\"id\",\"reason\":\"explanation\"}"
    
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

CURRENT STATE:
{deployments_table}
{bottleneck_hint}
LIMITS: {constraints}
{history_section}
Pick the deployment that needs adjustment. Use the exact deployment name in JSON.

EXAMPLES:
{{"action":"vertical_scaling","parameters":{{"deployment_name":"microservice3-deployment","cpu_limit":"600m","memory_limit":"612Mi"}}}}
{{"action":"horizontal_scaling","parameters":{{"deployment_name":"microservice1-deployment","replicas":2}}}}

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
        current_rt: float,
        ema_rt: float,
        cluster_data: dict,
        network_data: dict,
        monitoring_data: dict,
        history: str
    ) -> str:
        """
        Build the prompt for LLM root cause analysis.
        
        Provides the LLM with full system context and lets it reason
        about the root cause and recommend an appropriate action.
        """
        # Status description
        if violation_type == "UPPER_THRESHOLD_EXCEEDED":
            status = f"TOO SLOW (above {self.upper_threshold}s). Response time needs to decrease."
        else:
            status = f"TOO FAST (below {self.lower_threshold}s). Resources are over-provisioned and being wasted."
        
        # Format deployments with node placement
        deployments_table = self._format_system_state(cluster_data, network_data, monitoring_data)
        
        # Format node-level metrics
        node_metrics = self._format_node_metrics(monitoring_data)
        
        # History section
        if history and history != "(none)":
            history_section = f"PREVIOUS ACTIONS:\n{history}"
        else:
            history_section = "PREVIOUS ACTIONS: None yet."
        
        # Fill in the template
        prompt = self.prompt_template.format(
            ema_rt=f"{ema_rt:.2f}",
            lower_threshold=self.lower_threshold,
            upper_threshold=self.upper_threshold,
            status=status,
            deployments_table=deployments_table,
            node_metrics=node_metrics,
            history_section=history_section
        )
        
        return prompt
    
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

    def _query_ollama_8msg(
        self,
        history_entries: list,
        structured_state: dict,
        candidate_actions: list
    ) -> Optional[str]:
        """
        Send 8-message conversation to Ollama matching the tested format.
        
        Messages:
          1. system:    12-rule policy prompt
          2. user:      {"history": [...]}
          3. assistant: "Understood."
          4. user:      {"intent": ..., "application": ...}
          5. assistant: "Understood."
          6. user:      {"services": ..., "nodes": ..., "network": ...}
          7. assistant: "Understood."
          8. user:      {"candidate_actions": [...]} + "Select the best action."
        """
        # Build history JSON
        history_text = json.dumps({"history": history_entries})
        
        # Split state into parts
        part1 = json.dumps({
            "intent": structured_state["intent"],
            "application": structured_state["application"]
        })
        
        part2 = json.dumps({
            "services": structured_state["services"],
            "nodes": structured_state["nodes"],
            "network": structured_state["network"]
        })
        
        part3 = json.dumps({"candidate_actions": candidate_actions}) + "\n\nSelect the best action."
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": history_text},
            {"role": "assistant", "content": ASSISTANT_BRIDGE},
            {"role": "user", "content": part1},
            {"role": "assistant", "content": ASSISTANT_BRIDGE},
            {"role": "user", "content": part2},
            {"role": "assistant", "content": ASSISTANT_BRIDGE},
            {"role": "user", "content": part3}
        ]
        
        return self._send_chat(messages)

    def _send_chat(self, messages: list) -> Optional[str]:
        """Send a chat request to Ollama and return the response text."""
        url = f"{self.ollama_url}/api/chat"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": 200
            }
        }
        
        if self.debug_llm:
            logger.info("=" * 60)
            logger.info("LLM DEBUG - 8-MSG REQUEST:")
            logger.info(f"  Messages: {len(messages)}")
            for i, m in enumerate(messages):
                role = m['role']
                content_len = len(m['content'])
                logger.info(f"  [{i+1}] {role}: [{content_len} chars]")
                if role == "user":
                    logger.info(f"      {m['content'][:200]}...")
            logger.info("=" * 60)
        
        try:
            logger.info(f"Querying Ollama ({self.model}) with {len(messages)} messages...")
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            
            result = response.json()
            llm_response = result.get("message", {}).get("content", "")
            
            total_duration = result.get("total_duration", 0)
            prompt_tokens = result.get("prompt_eval_count", 0)
            output_tokens = result.get("eval_count", 0)
            if total_duration:
                logger.info(f"LLM response time: {total_duration / 1e9:.1f}s "
                           f"(prompt: {prompt_tokens}, output: {output_tokens} tokens)")
            
            if self.debug_llm:
                logger.info(f"LLM DEBUG - RESPONSE: {llm_response}")
            
            return llm_response.strip()
            
        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            return None

    def _parse_candidate_response(
        self, response_text: str, candidate_actions: list
    ) -> Optional[dict]:
        """
        Parse LLM response in {"answer":"A","reason":"..."} format.
        
        Maps the selected candidate action ID back to its full action dict,
        then converts to executor-compatible format.
        
        Returns:
            Executor-ready action dict or None if parsing failed.
        """
        if not response_text:
            return None
        
        # Clean up response
        cleaned = response_text.strip()
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)
        cleaned = cleaned.strip()
        
        # Try to parse JSON
        answer_id = None
        reason = ""
        
        try:
            parsed = json.loads(cleaned)
            answer_id = parsed.get("answer", "").strip().upper()
            reason = parsed.get("reason", "")
        except json.JSONDecodeError:
            # Regex fallback
            m = re.search(r'"answer"\s*:\s*"([A-D])"', cleaned, re.IGNORECASE)
            if m:
                answer_id = m.group(1).upper()
            r = re.search(r'"reason"\s*:\s*"([^"]*)"', cleaned)
            if r:
                reason = r.group(1)
        
        if not answer_id:
            logger.warning(f"Could not extract answer from LLM response: {response_text[:200]}")
            return None
        
        # Find the matching candidate
        selected = None
        for c in candidate_actions:
            if c.get("id", "").upper() == answer_id:
                selected = c
                break
        
        if not selected:
            logger.warning(f"Answer '{answer_id}' not found in candidates {[c['id'] for c in candidate_actions]}")
            return None
        
        logger.info(f"LLM selected: {answer_id} ({selected['type']}) — {reason}")
        
        # Convert candidate to executor-compatible action
        return self._candidate_to_executor_action(selected)

    def _candidate_to_executor_action(self, candidate: dict) -> dict:
        """
        Convert a candidate action dict to an executor-compatible format.
        
        Mapping:
          increase_cpu / reduce_cpu  → vertical_scaling
          add_replica / remove_replica → horizontal_scaling
          service_placement          → service_placement
          flow_scheduling            → flow_scheduling
        """
        action_type = candidate.get("type", "")
        target = candidate.get("target", "")
        
        if action_type == "increase_cpu":
            cpu_m = candidate.get("to_m", 500)
            mem_mi = max(128, (cpu_m // 100) * 100 + 12)
            return {
                "action": "vertical_scaling",
                "parameters": {
                    "deployment_name": target,
                    "cpu_limit": f"{cpu_m}m",
                    "memory_limit": f"{mem_mi}Mi"
                }
            }
        
        elif action_type == "reduce_cpu":
            cpu_m = candidate.get("to_m", 200)
            mem_mi = max(128, (cpu_m // 100) * 100 + 12)
            return {
                "action": "vertical_scaling",
                "parameters": {
                    "deployment_name": target,
                    "cpu_limit": f"{cpu_m}m",
                    "memory_limit": f"{mem_mi}Mi"
                }
            }
        
        elif action_type == "add_replica":
            return {
                "action": "horizontal_scaling",
                "parameters": {
                    "deployment_name": target,
                    "replicas": candidate.get("to_replicas", 2)
                }
            }
        
        elif action_type == "remove_replica":
            return {
                "action": "horizontal_scaling",
                "parameters": {
                    "deployment_name": target,
                    "replicas": candidate.get("to_replicas", 1)
                }
            }
        
        elif action_type == "service_placement":
            return {
                "action": "service_placement",
                "parameters": {
                    "deployment_name": target,
                    "target_node": candidate.get("to_node", "")
                }
            }
        
        elif action_type == "flow_scheduling":
            new_path = candidate.get("new_path", [])
            src = new_path[0] if new_path else ""
            dst = new_path[-1] if new_path else ""
            return {
                "action": "flow_scheduling",
                "parameters": {
                    "source_switch": src,
                    "destination_switch": dst,
                    "new_path": new_path,
                    "description": candidate.get("description", "")
                }
            }
        
        logger.warning(f"Unknown candidate type: {action_type}")
        return {"action": "none", "parameters": {}}

    def _build_retry_messages(
        self, base_messages: list, previous_response: str, error_reason: str
    ) -> list:
        """
        Append a corrective user message to the 8-msg conversation for retry.
        """
        retry_msg = (
            f"Your previous response was invalid: {error_reason}\n"
            f"Previous response: {previous_response[:100]}\n"
            f"Respond with ONLY valid JSON: {{\"answer\":\"A\",\"reason\":\"short explanation\"}}"
        )
        return base_messages + [{"role": "user", "content": retry_msg}]
    
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
    
    def analyze_and_recommend(
        self,
        violation_type: str,
        current_rt: float,
        ema_rt: float,
        structured_state: dict = None,
        candidate_actions: list = None,
        history_entries: list = None,
        # Legacy parameters (kept for backward compat, unused in 8-msg mode)
        cluster_data: dict = None,
        network_data: dict = None,
        monitoring_data: dict = None,
        history: str = "",
        monitoring_data_str: str = "",
        deployments_data: str = "",
        available_nodes: str = ""
    ) -> dict:
        """
        Main method: Analyze system state and recommend an action.
        
        Uses the 8-message conversation format with retry on parse failure.
        
        Args:
            violation_type: "UPPER_THRESHOLD_EXCEEDED" or "LOWER_THRESHOLD_EXCEEDED"
            current_rt: Current response time in seconds
            ema_rt: EMA response time in seconds
            structured_state: Structured JSON from DataCollector.build_structured_state()
            candidate_actions: List of candidate actions from CandidateActionGenerator
            history_entries: List of structured history dicts (rolling window)
            
        Returns:
            Dictionary with 'action' and 'parameters' keys
        """
        logger.info(f"Analyzing {violation_type} violation...")
        logger.info(f"Current RT: {current_rt:.2f}s, EMA: {ema_rt:.2f}s")
        logger.info(f"Thresholds: [{self.lower_threshold}, {self.upper_threshold}]")
        
        # Ensure we have structured data
        if not structured_state or not candidate_actions:
            logger.error("Missing structured_state or candidate_actions for 8-msg mode")
            return self._get_fallback_response("Missing structured data for LLM query")
        
        history_entries = history_entries or []
        
        logger.info(f"Candidates: {len(candidate_actions)} actions, History: {len(history_entries)} entries")
        if self.debug_llm:
            for c in candidate_actions:
                logger.info(f"  [{c['id']}] {c['type']} → {c.get('target', c.get('description', 'n/a'))}")
        
        # === RETRY LOOP ===
        last_response_text = None
        last_error = ""
        
        # Build base 8-message array
        history_text = json.dumps({"history": history_entries})
        part1 = json.dumps({
            "intent": structured_state["intent"],
            "application": structured_state["application"]
        })
        part2 = json.dumps({
            "services": structured_state["services"],
            "nodes": structured_state["nodes"],
            "network": structured_state["network"]
        })
        part3 = json.dumps({"candidate_actions": candidate_actions}) + "\n\nSelect the best action."
        
        base_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": history_text},
            {"role": "assistant", "content": ASSISTANT_BRIDGE},
            {"role": "user", "content": part1},
            {"role": "assistant", "content": ASSISTANT_BRIDGE},
            {"role": "user", "content": part2},
            {"role": "assistant", "content": ASSISTANT_BRIDGE},
            {"role": "user", "content": part3}
        ]
        
        for attempt in range(1, MAX_LLM_RETRIES + 1):
            if attempt == 1:
                messages = base_messages
            else:
                logger.warning(f"Retry {attempt}/{MAX_LLM_RETRIES}: {last_error}")
                messages = self._build_retry_messages(
                    base_messages, last_response_text or "", last_error
                )
            
            response_text = self._send_chat(messages)
            last_response_text = response_text
            
            if not response_text:
                last_error = "No response from LLM"
                continue
            
            action = self._parse_candidate_response(response_text, candidate_actions)
            
            if action and action.get("action") != "none":
                # Validate against config constraints
                is_valid, error_reason = self._validate_action(action)
                
                if is_valid:
                    self._consecutive_failures = 0
                    logger.info(f"Valid action on attempt {attempt}: {action['action']}")
                    logger.info(f"Parameters: {action['parameters']}")
                    return action
                else:
                    last_error = error_reason
                    logger.warning(f"Attempt {attempt}: valid parse but invalid action - {error_reason}")
            else:
                last_error = "Could not parse answer from LLM response"
                logger.warning(f"Attempt {attempt}: parse failed")
        
        # All retries exhausted — smart fallback
        self._consecutive_failures += 1
        logger.error(f"All {MAX_LLM_RETRIES} LLM attempts failed. Consecutive failures: {self._consecutive_failures}")
        
        if self._consecutive_failures <= 3 and candidate_actions:
            # Pick the first candidate as a reasonable default
            fallback = self._candidate_to_executor_action(candidate_actions[0])
            fallback["fallback_reason"] = f"LLM failed after {MAX_LLM_RETRIES} retries, using first candidate"
            logger.warning(f"Fallback: using candidate A ({candidate_actions[0]['type']})")
            return fallback
        else:
            logger.error("Too many consecutive failures, returning no-action")
            return self._get_fallback_response(f"All {MAX_LLM_RETRIES} attempts failed: {last_error}")
    
    def is_healthy(self) -> bool:
        """Check if Ollama is responding."""
        try:
            url = f"{self.ollama_url}/api/tags"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False