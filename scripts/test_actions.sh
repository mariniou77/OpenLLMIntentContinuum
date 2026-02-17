#!/bin/bash
# test_actions.sh - Manual test script for all IntentContinuum actions
# Run this on the SDN-Controller VM
#
# Usage: ./test_actions.sh
#
# This script tests:
# 1. Horizontal Scaling (change replica count)
# 2. Vertical Scaling (change CPU/memory limits)
# 3. Service Placement (move pod to different node)
# 4. Flow Scheduling (ONOS intent creation)
#
# Each test shows BEFORE state, applies change, shows AFTER state, then REVERTS

set -e

# Configuration - adjust these as needed
MASTER_HOST="10.132.0.14"
MASTER_USER="antonios-icontinuum"
MASTER="$MASTER_USER@$MASTER_HOST"
ONOS_URL="http://localhost:8181"
ONOS_USER="onos"
ONOS_PASS="rocks"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
}

print_section() {
    echo ""
    echo -e "${YELLOW}--- $1 ---${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

run_kubectl() {
    ssh -o StrictHostKeyChecking=no $MASTER "sudo kubectl $1"
}

wait_for_pods() {
    echo "Waiting for pods to stabilize..."
    sleep 5
    run_kubectl "get pods -o wide | grep -E 'microservice|db'"
}

# Store initial state for cleanup
declare -A INITIAL_REPLICAS
declare -A INITIAL_CPU
declare -A INITIAL_MEMORY
declare -A INITIAL_NODE

#############################################################################
# CAPTURE INITIAL STATE
#############################################################################
capture_initial_state() {
    print_header "CAPTURING INITIAL STATE"
    
    echo "Getting deployment info..."
    
    # Get all deployments and their replicas
    DEPLOYMENTS=$(run_kubectl "get deployments -o jsonpath='{range .items[*]}{.metadata.name}:{.spec.replicas} {end}'")
    
    for item in $DEPLOYMENTS; do
        name=$(echo $item | cut -d: -f1)
        replicas=$(echo $item | cut -d: -f2)
        INITIAL_REPLICAS[$name]=$replicas
        echo "  $name: $replicas replicas"
    done
    
    # Get resource limits for microservice1-deployment
    echo ""
    echo "Getting resource limits for microservice1-deployment..."
    RESOURCES=$(run_kubectl "get deployment microservice1-deployment -o jsonpath='{.spec.template.spec.containers[0].resources.limits.cpu},{.spec.template.spec.containers[0].resources.limits.memory}'")
    INITIAL_CPU["microservice1-deployment"]=$(echo $RESOURCES | cut -d, -f1)
    INITIAL_MEMORY["microservice1-deployment"]=$(echo $RESOURCES | cut -d, -f2)
    echo "  CPU: ${INITIAL_CPU["microservice1-deployment"]}"
    echo "  Memory: ${INITIAL_MEMORY["microservice1-deployment"]}"
    
    # Get node placement for microservice3-deployment
    echo ""
    echo "Getting node placement for microservice3-deployment..."
    INITIAL_NODE["microservice3-deployment"]=$(run_kubectl "get pods -l app=microservice3 -o jsonpath='{.items[0].spec.nodeName}'" 2>/dev/null || echo "unknown")
    echo "  Node: ${INITIAL_NODE["microservice3-deployment"]}"
    
    print_success "Initial state captured"
}

#############################################################################
# TEST 1: HORIZONTAL SCALING
#############################################################################
test_horizontal_scaling() {
    print_header "TEST 1: HORIZONTAL SCALING"
    
    DEPLOYMENT="microservice1-deployment"
    ORIGINAL_REPLICAS=${INITIAL_REPLICAS[$DEPLOYMENT]:-2}
    NEW_REPLICAS=3
    
    # BEFORE STATE
    print_section "BEFORE STATE"
    echo "Deployment: $DEPLOYMENT"
    run_kubectl "get deployment $DEPLOYMENT"
    echo ""
    run_kubectl "get pods -l app=microservice1 -o wide"
    
    # APPLY CHANGE
    print_section "APPLYING CHANGE: Scale to $NEW_REPLICAS replicas"
    run_kubectl "scale deployment $DEPLOYMENT --replicas=$NEW_REPLICAS"
    
    echo "Waiting for scaling to complete..."
    sleep 10
    
    # AFTER STATE
    print_section "AFTER STATE"
    run_kubectl "get deployment $DEPLOYMENT"
    echo ""
    run_kubectl "get pods -l app=microservice1 -o wide"
    
    # Verify
    CURRENT=$(run_kubectl "get deployment $DEPLOYMENT -o jsonpath='{.spec.replicas}'")
    if [ "$CURRENT" == "$NEW_REPLICAS" ]; then
        print_success "Horizontal scaling successful: $ORIGINAL_REPLICAS -> $NEW_REPLICAS replicas"
    else
        print_error "Horizontal scaling failed: expected $NEW_REPLICAS, got $CURRENT"
    fi
    
    # REVERT
    print_section "REVERTING: Scale back to $ORIGINAL_REPLICAS replicas"
    run_kubectl "scale deployment $DEPLOYMENT --replicas=$ORIGINAL_REPLICAS"
    sleep 5
    run_kubectl "get deployment $DEPLOYMENT"
    print_success "Reverted to original state"
}

#############################################################################
# TEST 2: VERTICAL SCALING
#############################################################################
test_vertical_scaling() {
    print_header "TEST 2: VERTICAL SCALING"
    
    DEPLOYMENT="microservice1-deployment"
    CONTAINER="microservice1"
    ORIGINAL_CPU=${INITIAL_CPU[$DEPLOYMENT]:-"300m"}
    ORIGINAL_MEMORY=${INITIAL_MEMORY[$DEPLOYMENT]:-"312Mi"}
    NEW_CPU="400m"
    NEW_MEMORY="400Mi"
    
    # BEFORE STATE
    print_section "BEFORE STATE"
    echo "Deployment: $DEPLOYMENT"
    echo "Container: $CONTAINER"
    echo "Current resources:"
    run_kubectl "get deployment $DEPLOYMENT -o jsonpath='  CPU Limit: {.spec.template.spec.containers[0].resources.limits.cpu}'"
    echo ""
    run_kubectl "get deployment $DEPLOYMENT -o jsonpath='  Memory Limit: {.spec.template.spec.containers[0].resources.limits.memory}'"
    echo ""
    
    # APPLY CHANGE
    print_section "APPLYING CHANGE: CPU=$NEW_CPU, Memory=$NEW_MEMORY"
    PATCH="{\"spec\":{\"template\":{\"spec\":{\"containers\":[{\"name\":\"$CONTAINER\",\"resources\":{\"limits\":{\"cpu\":\"$NEW_CPU\",\"memory\":\"$NEW_MEMORY\"},\"requests\":{\"cpu\":\"$NEW_CPU\",\"memory\":\"$NEW_MEMORY\"}}}]}}}}"
    
    run_kubectl "patch deployment $DEPLOYMENT --type=strategic -p '$PATCH'"
    
    echo "Waiting for pods to restart with new limits..."
    sleep 15
    
    # AFTER STATE
    print_section "AFTER STATE"
    echo "New resources:"
    run_kubectl "get deployment $DEPLOYMENT -o jsonpath='  CPU Limit: {.spec.template.spec.containers[0].resources.limits.cpu}'"
    echo ""
    run_kubectl "get deployment $DEPLOYMENT -o jsonpath='  Memory Limit: {.spec.template.spec.containers[0].resources.limits.memory}'"
    echo ""
    run_kubectl "get pods -l app=microservice1 -o wide"
    
    # Verify
    CURRENT_CPU=$(run_kubectl "get deployment $DEPLOYMENT -o jsonpath='{.spec.template.spec.containers[0].resources.limits.cpu}'")
    if [ "$CURRENT_CPU" == "$NEW_CPU" ]; then
        print_success "Vertical scaling successful: CPU $ORIGINAL_CPU -> $NEW_CPU"
    else
        print_error "Vertical scaling failed: expected CPU $NEW_CPU, got $CURRENT_CPU"
    fi
    
    # REVERT
    print_section "REVERTING: CPU=$ORIGINAL_CPU, Memory=$ORIGINAL_MEMORY"
    PATCH_REVERT="{\"spec\":{\"template\":{\"spec\":{\"containers\":[{\"name\":\"$CONTAINER\",\"resources\":{\"limits\":{\"cpu\":\"$ORIGINAL_CPU\",\"memory\":\"$ORIGINAL_MEMORY\"},\"requests\":{\"cpu\":\"$ORIGINAL_CPU\",\"memory\":\"$ORIGINAL_MEMORY\"}}}]}}}}"
    
    run_kubectl "patch deployment $DEPLOYMENT --type=strategic -p '$PATCH_REVERT'"
    sleep 10
    
    echo "Restored resources:"
    run_kubectl "get deployment $DEPLOYMENT -o jsonpath='  CPU Limit: {.spec.template.spec.containers[0].resources.limits.cpu}'"
    echo ""
    run_kubectl "get deployment $DEPLOYMENT -o jsonpath='  Memory Limit: {.spec.template.spec.containers[0].resources.limits.memory}'"
    echo ""
    print_success "Reverted to original state"
}

#############################################################################
# TEST 3: SERVICE PLACEMENT
#############################################################################
test_service_placement() {
    print_header "TEST 3: SERVICE PLACEMENT"
    
    DEPLOYMENT="microservice3-deployment"
    
    # BEFORE STATE
    print_section "BEFORE STATE"
    echo "Deployment: $DEPLOYMENT"
    echo "Available nodes:"
    run_kubectl "get nodes -o wide"
    echo ""
    echo "Current pod placement:"
    run_kubectl "get pods -l app=microservice3 -o wide"
    
    ORIGINAL_NODE=$(run_kubectl "get pods -l app=microservice3 -o jsonpath='{.items[0].spec.nodeName}'" 2>/dev/null)
    echo "Current node: $ORIGINAL_NODE"
    
    # Get list of nodes and pick a different one
    NODES=($(run_kubectl "get nodes -o jsonpath='{.items[*].metadata.name}'"))
    TARGET_NODE=""
    for node in "${NODES[@]}"; do
        if [ "$node" != "$ORIGINAL_NODE" ]; then
            TARGET_NODE=$node
            break
        fi
    done
    
    if [ -z "$TARGET_NODE" ]; then
        print_error "No alternative node found for placement test"
        return
    fi
    
    # APPLY CHANGE
    print_section "APPLYING CHANGE: Move to node $TARGET_NODE"
    PATCH="{\"spec\":{\"template\":{\"spec\":{\"nodeSelector\":{\"kubernetes.io/hostname\":\"$TARGET_NODE\"}}}}}"
    
    run_kubectl "patch deployment $DEPLOYMENT --type=strategic -p '$PATCH'"
    
    echo "Waiting for pod to be rescheduled..."
    sleep 20
    
    # AFTER STATE
    print_section "AFTER STATE"
    echo "New pod placement:"
    run_kubectl "get pods -l app=microservice3 -o wide"
    
    NEW_NODE=$(run_kubectl "get pods -l app=microservice3 -o jsonpath='{.items[0].spec.nodeName}'" 2>/dev/null)
    
    if [ "$NEW_NODE" == "$TARGET_NODE" ]; then
        print_success "Service placement successful: $ORIGINAL_NODE -> $TARGET_NODE"
    else
        print_error "Service placement may still be in progress: expected $TARGET_NODE, got $NEW_NODE"
    fi
    
    # REVERT
    print_section "REVERTING: Remove nodeSelector to restore default scheduling"
    # Remove nodeSelector using JSON patch
    run_kubectl "patch deployment $DEPLOYMENT --type=json -p '[{\"op\":\"remove\",\"path\":\"/spec/template/spec/nodeSelector\"}]'" 2>/dev/null || echo "nodeSelector may not exist"
    
    sleep 15
    echo "Restored pod placement (scheduler will place automatically):"
    run_kubectl "get pods -l app=microservice3 -o wide"
    print_success "Reverted to original state (default scheduling)"
}

#############################################################################
# TEST 4: FLOW SCHEDULING (ONOS)
#############################################################################
test_flow_scheduling() {
    print_header "TEST 4: FLOW SCHEDULING (ONOS)"
    
    # BEFORE STATE
    print_section "BEFORE STATE"
    echo "ONOS Devices:"
    curl -s -u $ONOS_USER:$ONOS_PASS $ONOS_URL/onos/v1/devices | jq -r '.devices[] | "  \(.id) - available: \(.available)"' 2>/dev/null || echo "  Could not connect to ONOS"
    
    echo ""
    echo "Current Intents:"
    INTENTS_BEFORE=$(curl -s -u $ONOS_USER:$ONOS_PASS $ONOS_URL/onos/v1/intents | jq -r '.intents | length' 2>/dev/null || echo "0")
    echo "  Total intents: $INTENTS_BEFORE"
    curl -s -u $ONOS_USER:$ONOS_PASS $ONOS_URL/onos/v1/intents | jq -r '.intents[] | "  \(.key): \(.type) - \(.state)"' 2>/dev/null || echo "  No intents or cannot connect"
    
    # Get first two devices for the test
    DEVICES=($(curl -s -u $ONOS_USER:$ONOS_PASS $ONOS_URL/onos/v1/devices | jq -r '.devices[].id' 2>/dev/null))
    
    if [ ${#DEVICES[@]} -lt 2 ]; then
        print_error "Not enough ONOS devices for flow scheduling test (need at least 2)"
        return
    fi
    
    INGRESS_DEVICE=${DEVICES[0]}
    EGRESS_DEVICE=${DEVICES[1]}
    
    # APPLY CHANGE
    print_section "APPLYING CHANGE: Create point-to-point intent"
    echo "Ingress: $INGRESS_DEVICE"
    echo "Egress: $EGRESS_DEVICE"
    
    INTENT_JSON="{
        \"type\": \"PointToPointIntent\",
        \"appId\": \"org.onosproject.cli\",
        \"ingressPoint\": {\"device\": \"$INGRESS_DEVICE\", \"port\": \"1\"},
        \"egressPoint\": {\"device\": \"$EGRESS_DEVICE\", \"port\": \"1\"}
    }"
    
    RESPONSE=$(curl -s -X POST -u $ONOS_USER:$ONOS_PASS \
        -H "Content-Type: application/json" \
        $ONOS_URL/onos/v1/intents \
        -d "$INTENT_JSON" -w "%{http_code}" -o /tmp/onos_response.txt)
    
    echo "HTTP Response: $RESPONSE"
    
    sleep 3
    
    # AFTER STATE
    print_section "AFTER STATE"
    echo "Intents after creation:"
    INTENTS_AFTER=$(curl -s -u $ONOS_USER:$ONOS_PASS $ONOS_URL/onos/v1/intents | jq -r '.intents | length' 2>/dev/null || echo "0")
    echo "  Total intents: $INTENTS_AFTER"
    curl -s -u $ONOS_USER:$ONOS_PASS $ONOS_URL/onos/v1/intents | jq -r '.intents[] | "  \(.key): \(.type) - \(.state)"' 2>/dev/null
    
    if [ "$INTENTS_AFTER" -gt "$INTENTS_BEFORE" ]; then
        print_success "Flow scheduling successful: Intent created"
    else
        print_error "Flow scheduling may have failed: intent count unchanged"
    fi
    
    # REVERT
    print_section "REVERTING: Delete created intent"
    # Get the newest intent key
    INTENT_KEY=$(curl -s -u $ONOS_USER:$ONOS_PASS $ONOS_URL/onos/v1/intents | jq -r '.intents[-1].key' 2>/dev/null)
    
    if [ ! -z "$INTENT_KEY" ] && [ "$INTENT_KEY" != "null" ]; then
        echo "Deleting intent: $INTENT_KEY"
        curl -s -X DELETE -u $ONOS_USER:$ONOS_PASS "$ONOS_URL/onos/v1/intents/org.onosproject.cli/$INTENT_KEY"
        sleep 2
        
        echo "Intents after deletion:"
        curl -s -u $ONOS_USER:$ONOS_PASS $ONOS_URL/onos/v1/intents | jq -r '.intents | length' 2>/dev/null
        print_success "Reverted to original state"
    else
        echo "No intent to delete"
    fi
}

#############################################################################
# FINAL STATE CHECK
#############################################################################
verify_final_state() {
    print_header "FINAL STATE VERIFICATION"
    
    echo "Deployments:"
    run_kubectl "get deployments"
    
    echo ""
    echo "Pods:"
    run_kubectl "get pods -o wide | grep -E 'microservice|db'"
    
    echo ""
    echo "ONOS Intents:"
    curl -s -u $ONOS_USER:$ONOS_PASS $ONOS_URL/onos/v1/intents | jq -r '.intents | length' 2>/dev/null || echo "Could not connect to ONOS"
    
    print_success "All tests completed and reverted"
}

#############################################################################
# MAIN
#############################################################################
main() {
    print_header "IntentContinuum - Action Executor Test Suite"
    echo "Master Node: $MASTER"
    echo "ONOS URL: $ONOS_URL"
    echo ""
    echo "This script will test all 4 actions:"
    echo "  1. Horizontal Scaling"
    echo "  2. Vertical Scaling"  
    echo "  3. Service Placement"
    echo "  4. Flow Scheduling"
    echo ""
    echo "Each test shows BEFORE/AFTER state and REVERTS changes."
    echo ""
    read -p "Press Enter to continue or Ctrl+C to cancel..."
    
    # Capture initial state
    capture_initial_state
    
    # Run tests
    test_horizontal_scaling
    test_vertical_scaling
    test_service_placement
    test_flow_scheduling
    
    # Verify final state
    verify_final_state
    
    print_header "TEST SUITE COMPLETED"
    echo ""
    echo "Summary:"
    echo "  ✓ Horizontal Scaling - tested and reverted"
    echo "  ✓ Vertical Scaling - tested and reverted"
    echo "  ✓ Service Placement - tested and reverted"
    echo "  ✓ Flow Scheduling - tested and reverted"
    echo ""
}

# Run main
main "$@"