#!/usr/bin/env python3
"""
Test Script: Flow Scheduling - Optimize Network Path
Violation Type: LOWER_THRESHOLD_EXCEEDED (response time too fast, can simplify)

This test simulates:
- Response time below lower threshold (system is over-provisioned)
- LLM recommends creating a more direct path or removing unnecessary intents

Note: In practice, LOWER_THRESHOLD for flow scheduling is less common.
This test demonstrates the ability to create/modify intents in response to 
over-provisioning scenarios (e.g., consolidating traffic paths).

Configuration:
- Modify SOURCE_SWITCH and DESTINATION_SWITCH for different paths
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

# Source switch (OpenFlow ID)
SOURCE_SWITCH = "of:0000000000000002"

# Destination switch (OpenFlow ID)
DESTINATION_SWITCH = "of:0000000000000003"

# Optional: Intermediate switches for explicit path
NEW_PATH = []  # Leave empty for ONOS to compute path automatically

# Simulated response time (below lower threshold of 1.0s)
SIMULATED_RT = 0.5

# Violation type
VIOLATION_TYPE = "LOWER_THRESHOLD_EXCEEDED"

# Whether to automatically revert changes after test
AUTO_REVERT = True

# Time to wait for changes to take effect (seconds)
WAIT_TIME = 5  # Flow scheduling is usually fast

# ============================================================
# TEST EXECUTION
# ============================================================

def main():
    print("\n" + "="*60)
    print("  FLOW SCHEDULING TEST - OPTIMIZE NETWORK PATH")
    print("  Violation: LOWER_THRESHOLD_EXCEEDED")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Source Switch: {SOURCE_SWITCH}")
    print(f"  Destination Switch: {DESTINATION_SWITCH}")
    print(f"  New Path: {NEW_PATH if NEW_PATH else '(auto-computed by ONOS)'}")
    print(f"  Simulated RT: {SIMULATED_RT}s")
    print(f"  Auto Revert: {AUTO_REVERT}")
    print(f"  Wait Time: {WAIT_TIME}s")
    
    # Initialize test harness
    harness = TestHarness()
    
    # Define the simulated LLM response
    parameters = {
        "source_switch": SOURCE_SWITCH,
        "destination_switch": DESTINATION_SWITCH,
        "new_path": NEW_PATH
    }
    
    # Run the test
    result = harness.run_test(
        action_type="flow_scheduling",
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
    print(f"  Action: flow_scheduling")
    print(f"  Violation: {VIOLATION_TYPE}")
    print(f"  Success: {'✅ Yes' if result['success'] else '❌ No'}")
    if result.get('error'):
        print(f"  Error: {result['error']}")
    print("="*60 + "\n")
    
    return 0 if result['success'] else 1


if __name__ == "__main__":
    sys.exit(main())
