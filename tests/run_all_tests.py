#!/usr/bin/env python3
"""
Master Test Runner for OpenLLMIntentContinuum

Runs all 8 test scripts sequentially:
- 2 horizontal_scaling tests (up/down)
- 2 vertical_scaling tests (up/down)
- 2 service_placement tests (up/down)
- 2 flow_scheduling tests (up/down)

Usage:
    python3 run_all_tests.py           # Run all tests
    python3 run_all_tests.py --action horizontal_scaling  # Run specific action tests
    python3 run_all_tests.py --violation upper            # Run only UPPER violation tests
"""

import sys
import os
import argparse
import subprocess
from datetime import datetime

# Test scripts
ALL_TESTS = [
    ("horizontal_scaling", "down", "test_horizontal_scaling_down.py"),
    ("horizontal_scaling", "up", "test_horizontal_scaling_up.py"),
    ("vertical_scaling", "down", "test_vertical_scaling_down.py"),
    ("vertical_scaling", "up", "test_vertical_scaling_up.py"),
    ("service_placement", "down", "test_service_placement_down.py"),
    ("service_placement", "up", "test_service_placement_up.py"),
    ("flow_scheduling", "down", "test_flow_scheduling_down.py"),
    ("flow_scheduling", "up", "test_flow_scheduling_up.py"),
]


def run_test(test_file: str) -> bool:
    """Run a single test script and return success status."""
    test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), test_file)
    
    try:
        result = subprocess.run(
            [sys.executable, test_path],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            timeout=180  # 3 minute timeout per test
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  ⏱️  Test timed out: {test_file}")
        return False
    except Exception as e:
        print(f"  ❌ Error running test: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run OpenLLMIntentContinuum action tests")
    parser.add_argument(
        "--action",
        choices=["horizontal_scaling", "vertical_scaling", "service_placement", "flow_scheduling"],
        help="Run only tests for a specific action type"
    )
    parser.add_argument(
        "--violation",
        choices=["upper", "lower"],
        help="Run only tests for a specific violation type (upper=UPPER_THRESHOLD_EXCEEDED, lower=LOWER_THRESHOLD_EXCEEDED)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available tests without running them"
    )
    
    args = parser.parse_args()
    
    # Filter tests based on arguments
    tests_to_run = ALL_TESTS
    
    if args.action:
        tests_to_run = [t for t in tests_to_run if t[0] == args.action]
    
    if args.violation:
        violation_map = {"upper": "up", "lower": "down"}
        tests_to_run = [t for t in tests_to_run if t[1] == violation_map[args.violation]]
    
    if args.list:
        print("\nAvailable Tests:")
        print("="*60)
        for action, direction, filename in ALL_TESTS:
            violation = "UPPER_THRESHOLD_EXCEEDED" if direction == "up" else "LOWER_THRESHOLD_EXCEEDED"
            print(f"  {filename}")
            print(f"    Action: {action}")
            print(f"    Violation: {violation}")
            print()
        return 0
    
    # Run tests
    print("\n" + "#"*60)
    print("  OpenLLMIntentContinuum - Test Suite")
    print("#"*60)
    print(f"\nStarted at: {datetime.now().isoformat()}")
    print(f"Tests to run: {len(tests_to_run)}")
    print()
    
    results = []
    
    for i, (action, direction, filename) in enumerate(tests_to_run, 1):
        violation = "UPPER_THRESHOLD_EXCEEDED" if direction == "up" else "LOWER_THRESHOLD_EXCEEDED"
        
        print(f"\n{'='*60}")
        print(f"  Running Test {i}/{len(tests_to_run)}: {filename}")
        print(f"  Action: {action}")
        print(f"  Violation: {violation}")
        print(f"{'='*60}\n")
        
        success = run_test(filename)
        results.append((filename, action, violation, success))
        
        # Brief pause between tests
        if i < len(tests_to_run):
            print("\n  Waiting 5 seconds before next test...\n")
            import time
            time.sleep(5)
    
    # Print summary
    print("\n" + "#"*60)
    print("  TEST SUITE SUMMARY")
    print("#"*60)
    print(f"\nCompleted at: {datetime.now().isoformat()}")
    print(f"\nResults:")
    
    passed = 0
    failed = 0
    
    for filename, action, violation, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        if success:
            passed += 1
        else:
            failed += 1
        print(f"  {status} - {action} ({violation.split('_')[0].lower()})")
    
    print(f"\nTotal: {passed} passed, {failed} failed out of {len(results)} tests")
    print("#"*60 + "\n")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
