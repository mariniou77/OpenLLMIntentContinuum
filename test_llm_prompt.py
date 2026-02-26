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

# Simulated deployment states for testing
FAKE_DEPLOYMENTS_TABLE = """| Name                       | Replicas | CPU Lim | CPU Used | Mem Lim | Mem Used |
|----------------------------|----------|---------|----------|---------|----------|
| microservice1-deployment   | 1        | 300m    | 250m     | 312Mi   | 200Mi    |
| microservice2-deployment   | 1        | 300m    | 80m      | 312Mi   | 100Mi    |
| microservice3-deployment   | 1        | 500m    | 450m     | 512Mi   | 400Mi    |
| microservice4-deployment   | 1        | 300m    | 60m      | 312Mi   | 80Mi     |"""

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
        if not params.get("memory_limit"):
            return False, parsed, "Missing memory_limit"

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
        "table": FAKE_DEPLOYMENTS_TABLE,
        "expect_action": "horizontal_scaling",
        "expect_direction": "up",
    },
    {
        "name": "LOWER violation (too fast, some with 3 replicas)",
        "ema_rt": "0.50",
        "lower": 1.0,
        "upper": 3.0,
        "status": "TOO FAST - must slow down to save resources",
        "rule": "DECREASE replicas (e.g., 2->1, min=1) OR DECREASE cpu_limit (e.g., 500m->300m, min=100m)",
        "table": """| Name                       | Replicas | CPU Lim | CPU Used | Mem Lim | Mem Used |
|----------------------------|----------|---------|----------|---------|----------|
| microservice1-deployment   | 3        | 500m    | 100m     | 512Mi   | 150Mi    |
| microservice2-deployment   | 1        | 300m    | 30m      | 312Mi   | 50Mi     |
| microservice3-deployment   | 3        | 600m    | 120m     | 612Mi   | 200Mi    |
| microservice4-deployment   | 1        | 300m    | 20m      | 312Mi   | 40Mi     |""",
        "expect_action": "horizontal_scaling",
        "expect_direction": "down",
    },
    {
        "name": "UPPER with history (ms1 failed, should try ms3)",
        "ema_rt": "5.00",
        "lower": 1.0,
        "upper": 3.0,
        "status": "TOO SLOW - must speed up",
        "rule": "INCREASE replicas (e.g., 1->2) OR INCREASE cpu_limit (e.g., 300m->500m)",
        "table": FAKE_DEPLOYMENTS_TABLE,
        "history": "\nHISTORY (avoid these - they WORSENED):\n- microservice1-deployment: FAILED\nTRY INSTEAD:\n- microservice3-deployment (only 1 replica - good candidate)\n",
        "expect_action": "horizontal_scaling",
        "expect_direction": "up",
    },
    {
        "name": "UPPER - need vertical scaling (already at max replicas)",
        "ema_rt": "4.00",
        "lower": 1.0,
        "upper": 3.0,
        "status": "TOO SLOW - must speed up",
        "rule": "INCREASE replicas (e.g., 1->2) OR INCREASE cpu_limit (e.g., 300m->500m)",
        "table": """| Name                       | Replicas | CPU Lim | CPU Used | Mem Lim | Mem Used |
|----------------------------|----------|---------|----------|---------|----------|
| microservice1-deployment   | 5        | 300m    | 290m     | 312Mi   | 300Mi    |
| microservice2-deployment   | 5        | 300m    | 280m     | 312Mi   | 290Mi    |
| microservice3-deployment   | 5        | 500m    | 490m     | 512Mi   | 500Mi    |
| microservice4-deployment   | 5        | 300m    | 270m     | 312Mi   | 280Mi    |""",
        "expect_action": "vertical_scaling",
        "expect_direction": "up",
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
                
                # Check if action matches expected type
                action_correct = (action == scenario.get("expect_action", ""))
                if action_correct:
                    correct_action += 1
                
                icon = "✅" if action_correct else "⚠️ "
                
                detail = ""
                if action == "horizontal_scaling":
                    detail = f"replicas={parsed['parameters'].get('replicas')}"
                elif action == "vertical_scaling":
                    detail = f"cpu={parsed['parameters'].get('cpu_limit')}"
                
                print(f"{icon} {scenario['name']}{run_label}")
                print(f"   → {action} on {dep} ({detail}) [{latency:.1f}s]")
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
    print(f"  Model:           {args.model}")
    print(f"  Total tests:     {total}")
    print(f"  Valid JSON:      {valid}/{total} ({100*valid/total:.0f}%)")
    print(f"  Correct action:  {correct_action}/{total} ({100*correct_action/total:.0f}%)")
    print(f"  Avg latency:     {sum(latencies)/len(latencies):.1f}s")
    print(f"  Min/Max latency: {min(latencies):.1f}s / {max(latencies):.1f}s")
    print(f"{'='*70}")

    if valid < total:
        print(f"\n⚠️  {total-valid} responses failed validation.")
        print("  Consider: ollama pull qwen2.5:3b  (or try phi3:3.8b or mistral:7b)")
    
    if correct_action == total:
        print("\n🎉 All responses were valid and picked the expected action type!")


if __name__ == "__main__":
    main()