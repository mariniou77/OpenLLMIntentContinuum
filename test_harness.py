"""
Test Harness for OpenLLMIntentContinuum

This module provides a testing framework that:
1. Simulates LLM responses with pre-defined outputs
2. Executes actual actions on the cluster
3. Logs before/after state
4. Reverts changes after testing

Usage:
    from test_harness import TestHarness
    
    harness = TestHarness(config)
    harness.run_test(
        action_type="horizontal_scaling",
        parameters={"deployment_name": "microservice1-deployment", "replicas": 2},
        violation_type="LOWER_THRESHOLD_EXCEEDED"
    )
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional

import yaml

from utils.kubernetes_client import KubernetesClient
from utils.onos_client import ONOSClient
from action_executor import ActionExecutor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestHarness:
    """
    Test harness for manually triggering OpenLLMIntentContinuum actions.
    
    This bypasses the LLM and directly executes actions with pre-defined parameters,
    allowing for predictable testing of the action execution pipeline.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the test harness.
        
        Args:
            config_path: Path to the configuration file
        """
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Initialize clients
        self.k8s_client = KubernetesClient(
            master_ip=self.config["endpoints"]["kubernetes_master"]
        )
        
        self.onos_client = ONOSClient(
            base_url=self.config["endpoints"]["onos"],
            username=self.config["endpoints"]["onos_user"],
            password=self.config["endpoints"]["onos_password"]
        )
        
        # Initialize action executor
        self.action_executor = ActionExecutor(self.config)
        
        # Store state for revert
        self.initial_state = {}
        self.changes_made = []
    
    def capture_state(self) -> Dict[str, Any]:
        """
        Capture current system state for comparison and revert.
        
        Returns:
            Dictionary with current state of deployments, nodes, and intents
        """
        logger.info("Capturing current system state...")
        
        state = {
            "timestamp": datetime.now().isoformat(),
            "deployments": {},
            "intents": []
        }
        
        # Capture deployment states
        deployments = self.k8s_client.get_deployments()
        for dep in deployments:
            name = dep.get("name")
            state["deployments"][name] = {
                "replicas": dep.get("replicas_desired", 0),
                "ready": dep.get("replicas_ready", 0)
            }
            
            # Also capture resource limits
            # This requires an additional kubectl call
            output = self.k8s_client._run_kubectl(
                f"get deployment {name} -o jsonpath='{{.spec.template.spec.containers[0].resources.limits}}'"
            )
            if output:
                try:
                    # Clean up the output (remove quotes)
                    clean_output = output.strip().strip("'")
                    if clean_output and clean_output != "{}":
                        state["deployments"][name]["resources"] = clean_output
                except:
                    pass
            
            # Capture node placement
            pods_output = self.k8s_client._run_kubectl(
                f"get pods -l app={name.replace('-deployment', '')} -o jsonpath='{{.items[*].spec.nodeName}}'"
            )
            if pods_output:
                state["deployments"][name]["nodes"] = pods_output.strip().split()
        
        # Capture ONOS intents
        try:
            intents = self.onos_client.get_intents()
            state["intents"] = [
                {"key": i.get("key"), "type": i.get("type"), "state": i.get("state")}
                for i in intents
            ]
        except Exception as e:
            logger.warning(f"Could not capture ONOS intents: {e}")
        
        return state
    
    def print_state(self, state: Dict[str, Any], title: str = "System State"):
        """Pretty print the system state."""
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
        print(f"Timestamp: {state.get('timestamp', 'N/A')}")
        
        print("\nDeployments:")
        for name, info in state.get("deployments", {}).items():
            replicas = info.get("replicas", "?")
            ready = info.get("ready", "?")
            resources = info.get("resources", "default")
            nodes = info.get("nodes", [])
            print(f"  - {name}:")
            print(f"      Replicas: {ready}/{replicas}")
            print(f"      Resources: {resources}")
            print(f"      Nodes: {', '.join(nodes) if nodes else 'N/A'}")
        
        print("\nONOS Intents:")
        intents = state.get("intents", [])
        if intents:
            for intent in intents:
                print(f"  - {intent.get('key')}: {intent.get('type')} ({intent.get('state')})")
        else:
            print("  (none)")
        
        print(f"{'='*60}\n")
    
    def compare_states(self, before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare before and after states to identify changes.
        
        Returns:
            Dictionary describing the changes
        """
        changes = {
            "deployments": {},
            "intents": {"added": [], "removed": []}
        }
        
        # Compare deployments
        for name in set(list(before.get("deployments", {}).keys()) + 
                       list(after.get("deployments", {}).keys())):
            before_dep = before.get("deployments", {}).get(name, {})
            after_dep = after.get("deployments", {}).get(name, {})
            
            dep_changes = {}
            
            # Check replicas
            if before_dep.get("replicas") != after_dep.get("replicas"):
                dep_changes["replicas"] = {
                    "before": before_dep.get("replicas"),
                    "after": after_dep.get("replicas")
                }
            
            # Check resources
            if before_dep.get("resources") != after_dep.get("resources"):
                dep_changes["resources"] = {
                    "before": before_dep.get("resources"),
                    "after": after_dep.get("resources")
                }
            
            # Check nodes
            if set(before_dep.get("nodes", [])) != set(after_dep.get("nodes", [])):
                dep_changes["nodes"] = {
                    "before": before_dep.get("nodes", []),
                    "after": after_dep.get("nodes", [])
                }
            
            if dep_changes:
                changes["deployments"][name] = dep_changes
        
        # Compare intents
        before_intents = {i.get("key") for i in before.get("intents", [])}
        after_intents = {i.get("key") for i in after.get("intents", [])}
        
        changes["intents"]["added"] = list(after_intents - before_intents)
        changes["intents"]["removed"] = list(before_intents - after_intents)
        
        return changes
    
    def print_changes(self, changes: Dict[str, Any]):
        """Pretty print the changes made."""
        print(f"\n{'='*60}")
        print("  CHANGES MADE")
        print(f"{'='*60}")
        
        has_changes = False
        
        # Deployment changes
        for name, dep_changes in changes.get("deployments", {}).items():
            has_changes = True
            print(f"\n{name}:")
            
            if "replicas" in dep_changes:
                print(f"  Replicas: {dep_changes['replicas']['before']} → {dep_changes['replicas']['after']}")
            
            if "resources" in dep_changes:
                print(f"  Resources: {dep_changes['resources']['before']} → {dep_changes['resources']['after']}")
            
            if "nodes" in dep_changes:
                print(f"  Nodes: {dep_changes['nodes']['before']} → {dep_changes['nodes']['after']}")
        
        # Intent changes
        if changes.get("intents", {}).get("added"):
            has_changes = True
            print(f"\nIntents Added: {changes['intents']['added']}")
        
        if changes.get("intents", {}).get("removed"):
            has_changes = True
            print(f"\nIntents Removed: {changes['intents']['removed']}")
        
        if not has_changes:
            print("\n  (No changes detected)")
        
        print(f"\n{'='*60}\n")
    
    def run_test(
        self,
        action_type: str,
        parameters: Dict[str, Any],
        violation_type: str = "UPPER_THRESHOLD_EXCEEDED",
        current_rt: float = 5.0,
        auto_revert: bool = True,
        wait_time: int = 15
    ) -> Dict[str, Any]:
        """
        Run a single test with the specified action.
        
        Args:
            action_type: One of horizontal_scaling, vertical_scaling, service_placement, flow_scheduling
            parameters: Action parameters (deployment_name, replicas, etc.)
            violation_type: UPPER_THRESHOLD_EXCEEDED or LOWER_THRESHOLD_EXCEEDED
            current_rt: Simulated current response time
            auto_revert: Whether to revert changes after test
            wait_time: Seconds to wait for changes to take effect
            
        Returns:
            Dictionary with test results
        """
        print(f"\n{'#'*60}")
        print(f"  TEST: {action_type}")
        print(f"  Violation: {violation_type}")
        print(f"  Parameters: {parameters}")
        print(f"{'#'*60}")
        
        result = {
            "action_type": action_type,
            "violation_type": violation_type,
            "parameters": parameters,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "error": None,
            "changes": {}
        }
        
        # Step 1: Capture initial state
        print("\n[1/5] Capturing BEFORE state...")
        before_state = self.capture_state()
        self.initial_state = before_state
        self.print_state(before_state, "BEFORE STATE")
        
        # Step 2: Simulate LLM recommendation
        print("[2/5] Simulating LLM recommendation...")
        simulated_recommendation = {
            "action": action_type,
            "parameters": parameters
        }
        print(f"  Simulated LLM output: {json.dumps(simulated_recommendation, indent=2)}")
        
        # Step 3: Execute the action
        print(f"\n[3/5] Executing action: {action_type}...")
        try:
            exec_result = self.action_executor.execute(
                action=action_type,
                parameters=parameters,
                analysis=f"Test execution for {violation_type}"
            )
            
            if exec_result.get("success"):
                print(f"  ✅ Action executed successfully: {exec_result.get('message')}")
                result["success"] = True
                self.changes_made.append({
                    "action": action_type,
                    "parameters": parameters
                })
            else:
                print(f"  ❌ Action failed: {exec_result.get('message')}")
                result["error"] = exec_result.get("message")
                
        except Exception as e:
            print(f"  ❌ Exception during execution: {e}")
            result["error"] = str(e)
        
        # Step 4: Wait and capture after state
        if result["success"]:
            print(f"\n[4/5] Waiting {wait_time}s for changes to take effect...")
            time.sleep(wait_time)
            
            after_state = self.capture_state()
            self.print_state(after_state, "AFTER STATE")
            
            # Compare and log changes
            changes = self.compare_states(before_state, after_state)
            result["changes"] = changes
            self.print_changes(changes)
        else:
            print("\n[4/5] Skipping state capture (action failed)")
        
        # Step 5: Revert if requested
        if auto_revert and result["success"]:
            print("[5/5] Reverting changes...")
            self.revert_action(action_type, parameters, before_state)
            
            # Verify revert
            time.sleep(10)
            reverted_state = self.capture_state()
            print("  Revert complete. Verifying...")
            
            # Check if state is restored
            revert_changes = self.compare_states(before_state, reverted_state)
            if not any(revert_changes.get("deployments", {}).values()):
                print("  ✅ State successfully reverted")
            else:
                print("  ⚠️  State may not be fully reverted")
        else:
            print("[5/5] Skipping revert (not requested or action failed)")
        
        return result
    
    def revert_action(
        self,
        action_type: str,
        parameters: Dict[str, Any],
        original_state: Dict[str, Any]
    ):
        """
        Revert an action to restore original state.
        
        Args:
            action_type: The action type that was executed
            parameters: The parameters that were used
            original_state: The state before the action
        """
        deployment_name = parameters.get("deployment_name")
        
        if action_type == "horizontal_scaling":
            # Restore original replica count
            if deployment_name and deployment_name in original_state.get("deployments", {}):
                original_replicas = original_state["deployments"][deployment_name].get("replicas", 1)
                print(f"  Reverting {deployment_name} to {original_replicas} replicas...")
                self.k8s_client.scale_deployment(deployment_name, original_replicas)
        
        elif action_type == "vertical_scaling":
            # Restore original resources (use defaults from config)
            if deployment_name:
                # Find default resources from config
                default_cpu = "300m"
                default_memory = "312Mi"
                
                for dep_config in self.config.get("kubernetes", {}).get("deployments", []):
                    if dep_config.get("name") == deployment_name:
                        default_cpu = dep_config.get("default_cpu", "300m")
                        default_memory = dep_config.get("default_memory", "312Mi")
                        break
                
                print(f"  Reverting {deployment_name} resources to CPU={default_cpu}, Memory={default_memory}...")
                self.k8s_client.set_resources(deployment_name, default_cpu, default_memory)
        
        elif action_type == "service_placement":
            # Remove nodeSelector to restore default scheduling
            if deployment_name:
                print(f"  Removing nodeSelector from {deployment_name}...")
                self.k8s_client._run_kubectl(
                    f"patch deployment {deployment_name} --type=json -p '[{{\"op\":\"remove\",\"path\":\"/spec/template/spec/nodeSelector\"}}]'"
                )
        
        elif action_type == "flow_scheduling":
            # Delete the created intent
            source = parameters.get("source_switch")
            dest = parameters.get("destination_switch")
            
            # Find and delete matching intents
            print("  Removing created intents...")
            intents = self.onos_client.get_intents()
            for intent in intents:
                # Delete recent intents (simple approach)
                key = intent.get("key")
                if key:
                    self.onos_client.delete_intent("org.onosproject.cli", key)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)