#!/usr/bin/env python3
"""
LLM Prompt Quality Test

Run this on the SDN-Controller VM to verify that Qwen2.5:3b produces
valid JSON responses with the improved prompt. No infrastructure needed
— just Ollama running with the model loaded.

Usage:
    python3 test_llm_prompt.py
    python3 test_llm_prompt.py --model qwen2.5:3b
    python3 test_llm_prompt.py --runs 20  # Run 20 tests for statistics
"""

import json
import argparse
import requests
import time
import sys

OLLAMA_URL = "http://localhost:11434"

# Valid deployment names (must match your config.yaml)
VALID_DEPLOYMENTS = {
    "microservice1-deployment",
    "microservice2-deployment",
    "microservice3-deployment",
    "microservice4-deployment",
}

# Simulated deployment states for testing (explicit labeled format)
FAKE_DEPLOYMENTS = """microservice1-deployment: replicas=1, cpu_usage=250m, cpu_limit=300m, memory_usage=200Mi, memory_limit=312Mi
microservice2-deployment: replicas=1, cpu_usage=80m, cpu_limit=300m, memory_usage=100Mi, memory_limit=312Mi
microservice3-deployment: replicas=1, cpu_usage=450m, cpu_limit=500m, memory_usage=400Mi, memory_limit=512Mi
microservice4-deployment: replicas=1, cpu_usage=60m, cpu_limit=300m, memory_usage=80Mi, memory_limit=312Mi"""

# Load the prompt template
def load_prompt_template():
    """Try to load from file, fall back to embedded."""
    try:
        with open("prompts/analysis_prompt.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
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

JSON:"""


def build_test_prompt(template, scenario):
    """Build a test prompt from a scenario dict."""
    return template.format(
        ema_rt=scenario["ema_rt"],
        lower_threshold=scenario["lower"],
        upper_threshold=scenario["upper"],
        status=scenario["status"],
        what_to_do=scenario["rule"],
        deployments_table=scenario["table"],
        bottleneck_hint=scenario.get("bottleneck_hint", ""),
        constraints="Replicas: 1-5 | CPU: 100m-1000m | Memory: 128Mi-1024Mi",
        history_section=scenario.get("history", ""),
    )


def query_ollama(model, prompt, temperature=0.1):
    """Query Ollama and return (response_text, latency_seconds)."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": temperature, "num_predict": 256},
    }
    start = time.time()
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
        elapsed = time.time() - start
        return resp.json().get("response", ""), elapsed
    except Exception as e:
        return f"ERROR: {e}", time.time() - start


def validate_response(response_text):
    """
    Validate LLM response. Returns (is_valid, action_dict, error_msg).
    """
    if not response_text or response_text.startswith("ERROR:"):
        return False, {}, f"No response: {response_text}"

    cleaned = response_text.strip()

    # Try JSON parse
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try extracting JSON from surrounding text
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            try:
                parsed = json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                return False, {}, f"JSON parse failed: {cleaned[:200]}"
        else:
            return False, {}, f"No JSON found: {cleaned[:200]}"

    if not isinstance(parsed, dict):
        return False, {}, f"Not a dict: {type(parsed)}"

    action = parsed.get("action", "")
    params = parsed.get("parameters", {})

    # Normalize common LLM action name variants
    action_normalization = {
        "horizontal_scaling": "horizontal_scaling",
        "vertical_scaling": "vertical_scaling",
        "increase_replicas": "horizontal_scaling",
        "decrease_replicas": "horizontal_scaling",
        "scale_up": "horizontal_scaling",
        "scale_down": "horizontal_scaling",
        "increase_cpu": "vertical_scaling",
        "decrease_cpu": "vertical_scaling",
        "service_placement": "service_placement",
        "flow_scheduling": "flow_scheduling",
    }
    
    normalized_action = action_normalization.get(action, action)
    parsed["action"] = normalized_action
    action = normalized_action

    # Check action type
    valid_actions = {"horizontal_scaling", "vertical_scaling", "service_placement", "flow_scheduling"}
    if action not in valid_actions:
        return False, parsed, f"Invalid action: '{action}'"

    # Check deployment name
    dep_name = params.get("deployment_name", "")
    if action in ("horizontal_scaling", "vertical_scaling"):
        if dep_name not in VALID_DEPLOYMENTS:
            return False, parsed, f"Invalid deployment: '{dep_name}'"

    # Check required params
    if action == "horizontal_scaling":
        replicas = params.get("replicas")
        if replicas is None:
            return False, parsed, "Missing replicas"
        try:
            r = int(replicas)
            if r < 1 or r > 10:
                return False, parsed, f"Replicas out of range: {r}"
        except (ValueError, TypeError):
            return False, parsed, f"Non-integer replicas: {replicas}"

    if action == "vertical_scaling":
        if not params.get("cpu_limit"):
            return False, parsed, "Missing cpu_limit"
        # Auto-fill memory_limit if missing (common with small LLMs)
        if not params.get("memory_limit"):
            try:
                cpu_val = int(str(params["cpu_limit"]).replace("m", "").strip())
                mem_val = max(128, (cpu_val // 100) * 100 + 12)
                params["memory_limit"] = f"{mem_val}Mi"
            except (ValueError, TypeError):
                params["memory_limit"] = "512Mi"

    return True, parsed, ""


# ---- TEST SCENARIOS ----

SCENARIOS = [
    {
        "name": "UPPER violation (too slow, all 1 replica)",
        "ema_rt": "4.50",
        "lower": 1.0,
        "upper": 3.0,
        "status": "TOO SLOW - must speed up",
        "rule": "INCREASE replicas (e.g., 1->2) OR INCREASE cpu_limit (e.g., 300m->500m)",
        "table": FAKE_DEPLOYMENTS,
        "bottleneck_hint": "BOTTLENECK: microservice3-deployment (cpu_usage=450m, cpu_limit=500m) - target this one",
        "expect_action": ["horizontal_scaling", "vertical_scaling"],
        "expect_target": "microservice3-deployment",
    },
    {
        "name": "LOWER violation (too fast, some with 3 replicas)",
        "ema_rt": "0.50",
        "lower": 1.0,
        "upper": 3.0,
        "status": "TOO FAST - must slow down to save resources",
        "rule": "DECREASE replicas (e.g., 2->1, min=1) OR DECREASE cpu_limit (e.g., 500m->300m, min=100m)",
        "table": """microservice1-deployment: replicas=3, cpu_usage=100m, cpu_limit=500m, memory_usage=150Mi, memory_limit=512Mi
microservice2-deployment: replicas=1, cpu_usage=30m, cpu_limit=300m, memory_usage=50Mi, memory_limit=312Mi
microservice3-deployment: replicas=3, cpu_usage=120m, cpu_limit=600m, memory_usage=200Mi, memory_limit=612Mi
microservice4-deployment: replicas=1, cpu_usage=20m, cpu_limit=300m, memory_usage=40Mi, memory_limit=312Mi""",
        "bottleneck_hint": "OVER-PROVISIONED: microservice1-deployment has 3 replicas - target this one",
        "expect_action": ["horizontal_scaling", "vertical_scaling"],
        "expect_target": ["microservice1-deployment", "microservice3-deployment"],
    },
    {
        "name": "UPPER with history (ms1 failed, should try ms3)",
        "ema_rt": "5.00",
        "lower": 1.0,
        "upper": 3.0,
        "status": "TOO SLOW - must speed up",
        "rule": "INCREASE replicas (e.g., 1->2) OR INCREASE cpu_limit (e.g., 300m->500m)",
        "table": FAKE_DEPLOYMENTS,
        "bottleneck_hint": "BOTTLENECK: microservice3-deployment (cpu_usage=450m, cpu_limit=500m) - target this one",
        "history": "\nHISTORY (avoid these - they WORSENED):\n- microservice1-deployment: FAILED\nTRY INSTEAD:\n- microservice3-deployment (only 1 replica - good candidate)\n",
        "expect_action": ["horizontal_scaling", "vertical_scaling"],
        "expect_target": ["microservice3-deployment", "microservice2-deployment", "microservice4-deployment"],
        "avoid_target": "microservice1-deployment",
    },
    {
        "name": "UPPER - need vertical scaling (already at max replicas)",
        "ema_rt": "4.00",
        "lower": 1.0,
        "upper": 3.0,
        "status": "TOO SLOW - must speed up",
        "rule": "INCREASE replicas (e.g., 1->2) OR INCREASE cpu_limit (e.g., 300m->500m)",
        "table": """microservice1-deployment: replicas=5, cpu_usage=290m, cpu_limit=300m, memory_usage=300Mi, memory_limit=312Mi
microservice2-deployment: replicas=5, cpu_usage=280m, cpu_limit=300m, memory_usage=290Mi, memory_limit=312Mi
microservice3-deployment: replicas=5, cpu_usage=490m, cpu_limit=500m, memory_usage=500Mi, memory_limit=512Mi
microservice4-deployment: replicas=5, cpu_usage=270m, cpu_limit=300m, memory_usage=280Mi, memory_limit=312Mi""",
        "bottleneck_hint": "BOTTLENECK: microservice3-deployment (cpu_usage=490m, cpu_limit=500m) - target this one",
        "expect_action": ["vertical_scaling"],
        "expect_target": ["microservice3-deployment", "microservice1-deployment"],
    },
]


def main():
    parser = argparse.ArgumentParser(description="Test LLM prompt quality")
    parser.add_argument("--model", default="qwen2.5:3b", help="Ollama model name")
    parser.add_argument("--runs", type=int, default=1, help="Repeat each scenario N times")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full responses")
    args = parser.parse_args()

    # Check Ollama is running
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"✅ Ollama is running. Available models: {models}")
        if not any(args.model in m for m in models):
            print(f"⚠️  Model '{args.model}' not found. Pull it with: ollama pull {args.model}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Cannot connect to Ollama at {OLLAMA_URL}: {e}")
        sys.exit(1)

    template = load_prompt_template()

    total = 0
    valid = 0
    correct_action = 0
    correct_target = 0
    latencies = []

    print(f"\n{'='*70}")
    print(f"Testing model: {args.model} | Runs per scenario: {args.runs}")
    print(f"{'='*70}\n")

    for scenario in SCENARIOS:
        for run in range(args.runs):
            total += 1
            prompt = build_test_prompt(template, scenario)
            response_text, latency = query_ollama(args.model, prompt)
            latencies.append(latency)

            is_valid, parsed, error = validate_response(response_text)

            run_label = f"  Run {run+1}/{args.runs}" if args.runs > 1 else ""
            
            if is_valid:
                valid += 1
                action = parsed.get("action", "")
                dep = parsed.get("parameters", {}).get("deployment_name", "")
                
                # Check if action matches expected type(s)
                expect_actions = scenario.get("expect_action", [])
                if isinstance(expect_actions, str):
                    expect_actions = [expect_actions]
                action_ok = action in expect_actions
                if action_ok:
                    correct_action += 1
                
                # Check if target deployment is correct
                expect_targets = scenario.get("expect_target", [])
                if isinstance(expect_targets, str):
                    expect_targets = [expect_targets]
                avoid_target = scenario.get("avoid_target", "")
                
                target_ok = True
                if expect_targets and dep not in expect_targets:
                    target_ok = False
                if avoid_target and dep == avoid_target:
                    target_ok = False
                if target_ok and (expect_targets or avoid_target):
                    correct_target += 1
                
                icon = "✅" if (action_ok and target_ok) else ("⚠️ " if action_ok else "🎯")
                
                detail = ""
                if action == "horizontal_scaling":
                    detail = f"replicas={parsed['parameters'].get('replicas')}"
                elif action == "vertical_scaling":
                    detail = f"cpu={parsed['parameters'].get('cpu_limit')}"
                
                target_note = ""
                if not target_ok:
                    target_note = f" ← wrong target (expected: {expect_targets})"
                if avoid_target and dep == avoid_target:
                    target_note = f" ← should have avoided {avoid_target}"
                
                print(f"{icon} {scenario['name']}{run_label}")
                print(f"   → {action} on {dep} ({detail}) [{latency:.1f}s]{target_note}")
            else:
                print(f"❌ {scenario['name']}{run_label}")
                print(f"   → Error: {error} [{latency:.1f}s]")

            if args.verbose:
                print(f"   Raw: {response_text[:300]}")
            print()

    # Summary
    print(f"{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Model:            {args.model}")
    print(f"  Total tests:      {total}")
    print(f"  Valid JSON:       {valid}/{total} ({100*valid/total:.0f}%)")
    print(f"  Correct action:   {correct_action}/{total} ({100*correct_action/total:.0f}%)")
    print(f"  Correct target:   {correct_target}/{total} ({100*correct_target/total:.0f}%)")
    print(f"  Avg latency:      {sum(latencies)/len(latencies):.1f}s")
    print(f"  Min/Max latency:  {min(latencies):.1f}s / {max(latencies):.1f}s")
    print(f"{'='*70}")

    if valid < total:
        print(f"\n⚠️  {total-valid} responses failed validation.")
        print("  Consider: ollama pull qwen2.5:3b  (or try phi3:3.8b or mistral:7b)")
    
    if correct_action == total and correct_target == total:
        print("\n🎉 All responses were valid with correct actions AND targets!")


if __name__ == "__main__":
    main()