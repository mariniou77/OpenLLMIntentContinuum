#!/usr/bin/env python3
"""
Test Script: Service Placement - Move Pod
Violation Type: UPPER_THRESHOLD_EXCEEDED (response time too slow, needs better placement)

This test simulates:
- Response time above upper threshold (current node may be overloaded)
- LLM recommends moving pod to a different node for better performance

Configuration:
- Modify DEPLOYMENT_NAME to test different deployments
- Modify TARGET_NODE to set the destination node
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

# Which deployment to move
DEPLOYMENT_NAME = "microservice3-deployment"

# Target node to move the pod to
TARGET_NODE = "worker1"

# Simulated response time (above upper threshold of 3.0s)
SIMULATED_RT = 5.0

# Violation type
VIOLATION_TYPE = "UPPER_THRESHOLD_EXCEEDED"

# Whether to automatically revert changes after test
AUTO_REVERT = True

# Time to wait for changes to take effect (seconds)
WAIT_TIME = 25  # Service placement requires pod rescheduling

# ============================================================
# TEST EXECUTION
# ============================================================

def main():
    print("\n" + "="*60)
    print("  SERVICE PLACEMENT TEST - MOVE POD")
    print("  Violation: UPPER_THRESHOLD_EXCEEDED")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Deployment: {DEPLOYMENT_NAME}")
    print(f"  Target Node: {TARGET_NODE}")
    print(f"  Simulated RT: {SIMULATED_RT}s")
    print(f"  Auto Revert: {AUTO_REVERT}")
    print(f"  Wait Time: {WAIT_TIME}s")
    
    # Initialize test harness
    harness = TestHarness()
    
    # Define the simulated LLM response
    parameters = {
        "deployment_name": DEPLOYMENT_NAME,
        "target_node": TARGET_NODE
    }
    
    # Run the test
    result = harness.run_test(
        action_type="service_placement",
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
    print(f"  Action: service_placement")
    print(f"  Violation: {VIOLATION_TYPE}")
    print(f"  Success: {'✅ Yes' if result['success'] else '❌ No'}")
    if result.get('error'):
        print(f"  Error: {result['error']}")
    print("="*60 + "\n")
    
    return 0 if result['success'] else 1


if __name__ == "__main__":
    sys.exit(main())
