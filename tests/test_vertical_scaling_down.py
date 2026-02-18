#!/usr/bin/env python3
"""
Test Script: Vertical Scaling - Reduce Resources
Violation Type: LOWER_THRESHOLD_EXCEEDED (response time too fast, over-provisioned)

This test simulates:
- Response time below lower threshold (system has excess resources)
- LLM recommends reducing CPU/memory limits to save resources

Configuration:
- Modify DEPLOYMENT_NAME to test different deployments
- Modify TARGET_CPU and TARGET_MEMORY to set resource limits
- Set AUTO_REVERT to False to keep changes after test
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_harness import TestHarness

# ============================================================
# CONFIGURATION - Modify these values for different test cases
# ============================================================

# Which deployment to modify
DEPLOYMENT_NAME = "microservice1-deployment"

# Target CPU limit (reduce from current)
TARGET_CPU = "200m"

# Target memory limit (reduce from current)
TARGET_MEMORY = "256Mi"

# Simulated response time (below lower threshold of 1.0s)
SIMULATED_RT = 0.5

# Violation type
VIOLATION_TYPE = "LOWER_THRESHOLD_EXCEEDED"

# Whether to automatically revert changes after test
AUTO_REVERT = True

# Time to wait for changes to take effect (seconds)
WAIT_TIME = 20  # Vertical scaling requires pod restart

# ============================================================
# TEST EXECUTION
# ============================================================

def main():
    print("\n" + "="*60)
    print("  VERTICAL SCALING TEST - REDUCE RESOURCES")
    print("  Violation: LOWER_THRESHOLD_EXCEEDED")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Deployment: {DEPLOYMENT_NAME}")
    print(f"  Target CPU: {TARGET_CPU}")
    print(f"  Target Memory: {TARGET_MEMORY}")
    print(f"  Simulated RT: {SIMULATED_RT}s")
    print(f"  Auto Revert: {AUTO_REVERT}")
    print(f"  Wait Time: {WAIT_TIME}s")
    
    # Initialize test harness
    harness = TestHarness()
    
    # Define the simulated LLM response
    parameters = {
        "deployment_name": DEPLOYMENT_NAME,
        "cpu_limit": TARGET_CPU,
        "memory_limit": TARGET_MEMORY
    }
    
    # Run the test
    result = harness.run_test(
        action_type="vertical_scaling",
        parameters=parameters,
        violation_type=VIOLATION_TYPE,
        current_rt=SIMULATED_RT,
        auto_revert=AUTO_REVERT,
        wait_time=WAIT_TIME
    )
    
    # Print summary
    print("\n" + "="*60)
    print("  TEST SUMMARY")
    print("="*60)
    print(f"  Action: vertical_scaling")
    print(f"  Violation: {VIOLATION_TYPE}")
    print(f"  Success: {'✅ Yes' if result['success'] else '❌ No'}")
    if result.get('error'):
        print(f"  Error: {result['error']}")
    print("="*60 + "\n")
    
    return 0 if result['success'] else 1


if __name__ == "__main__":
    sys.exit(main())
