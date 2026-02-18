#!/usr/bin/env python3
"""
Test Script: Horizontal Scaling - Scale UP
Violation Type: UPPER_THRESHOLD_EXCEEDED (response time too slow, needs more resources)

This test simulates:
- Response time above upper threshold (system needs more capacity)
- LLM recommends increasing replicas to handle load

Configuration:
- Modify DEPLOYMENT_NAME to test different deployments
- Modify TARGET_REPLICAS to set the desired replica count
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

# Which deployment to scale
DEPLOYMENT_NAME = "microservice3-deployment"

# Target replica count (should be MORE than current for scale up)
TARGET_REPLICAS = 3

# Simulated response time (above upper threshold of 3.0s)
SIMULATED_RT = 5.0

# Violation type
VIOLATION_TYPE = "UPPER_THRESHOLD_EXCEEDED"

# Whether to automatically revert changes after test
AUTO_REVERT = True

# Time to wait for changes to take effect (seconds)
WAIT_TIME = 15

# ============================================================
# TEST EXECUTION
# ============================================================

def main():
    print("\n" + "="*60)
    print("  HORIZONTAL SCALING TEST - SCALE UP")
    print("  Violation: UPPER_THRESHOLD_EXCEEDED")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Deployment: {DEPLOYMENT_NAME}")
    print(f"  Target Replicas: {TARGET_REPLICAS}")
    print(f"  Simulated RT: {SIMULATED_RT}s")
    print(f"  Auto Revert: {AUTO_REVERT}")
    print(f"  Wait Time: {WAIT_TIME}s")
    
    # Initialize test harness
    harness = TestHarness()
    
    # Define the simulated LLM response
    parameters = {
        "deployment_name": DEPLOYMENT_NAME,
        "replicas": TARGET_REPLICAS
    }
    
    # Run the test
    result = harness.run_test(
        action_type="horizontal_scaling",
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
    print(f"  Action: horizontal_scaling")
    print(f"  Violation: {VIOLATION_TYPE}")
    print(f"  Success: {'✅ Yes' if result['success'] else '❌ No'}")
    if result.get('error'):
        print(f"  Error: {result['error']}")
    print("="*60 + "\n")
    
    return 0 if result['success'] else 1


if __name__ == "__main__":
    sys.exit(main())
