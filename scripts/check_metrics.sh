#!/bin/bash
# =============================================================================
# Script to check available metrics from Kubernetes, sFlow-RT
# Run this on your SDN-Controller VM
# =============================================================================

echo "=============================================="
echo "1. KUBERNETES - NODES INFO"
echo "=============================================="
ssh antonios-icontinuum@10.132.0.14 "kubectl get nodes -o wide"

echo ""
echo "=============================================="
echo "2. KUBERNETES - NODE RESOURCE USAGE (kubectl top)"
echo "=============================================="
ssh antonios-icontinuum@10.132.0.14 "kubectl top nodes 2>/dev/null || echo 'metrics-server not installed'"

echo ""
echo "=============================================="
echo "3. KUBERNETES - PODS INFO"
echo "=============================================="
ssh antonios-icontinuum@10.132.0.14 "kubectl get pods -o wide"

echo ""
echo "=============================================="
echo "4. KUBERNETES - POD RESOURCE USAGE (kubectl top)"
echo "=============================================="
ssh antonios-icontinuum@10.132.0.14 "kubectl top pods 2>/dev/null || echo 'metrics-server not installed'"

echo ""
echo "=============================================="
echo "5. KUBERNETES - DEPLOYMENTS WITH REPLICAS"
echo "=============================================="
ssh antonios-icontinuum@10.132.0.14 "kubectl get deployments -o wide"

echo ""
echo "=============================================="
echo "6. KUBERNETES - DEPLOYMENT DETAILS (resources)"
echo "=============================================="
ssh antonios-icontinuum@10.132.0.14 "kubectl get deployments -o jsonpath='{range .items[*]}{.metadata.name}{\" replicas=\"}{.spec.replicas}{\" cpu=\"}{.spec.template.spec.containers[0].resources.limits.cpu}{\" mem=\"}{.spec.template.spec.containers[0].resources.limits.memory}{\"\\n\"}{end}'"

echo ""
echo "=============================================="
echo "7. SFLOW-RT - AVAILABLE METRICS"
echo "=============================================="
echo "Checking sFlow-RT agents..."
curl -s http://localhost:8008/agents/json | python3 -m json.tool 2>/dev/null | head -50

echo ""
echo "=============================================="
echo "8. SFLOW-RT - CPU METRICS"
echo "=============================================="
curl -s "http://localhost:8008/metric/ALL/cpu_utilization/json" | python3 -m json.tool 2>/dev/null | head -30

echo ""
echo "=============================================="
echo "9. SFLOW-RT - MEMORY METRICS"  
echo "=============================================="
curl -s "http://localhost:8008/metric/ALL/mem_utilization/json" | python3 -m json.tool 2>/dev/null | head -30

echo ""
echo "=============================================="
echo "10. SFLOW-RT - ALL AVAILABLE METRIC KEYS"
echo "=============================================="
curl -s "http://localhost:8008/metrics/json" | python3 -m json.tool 2>/dev/null | head -50

echo ""
echo "=============================================="
echo "11. KUBERNETES - DETAILED POD INFO (JSON)"
echo "=============================================="
ssh antonios-icontinuum@10.132.0.14 "kubectl get pods -o json" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for pod in data.get('items', []):
    name = pod['metadata']['name']
    node = pod['spec'].get('nodeName', 'N/A')
    status = pod['status']['phase']
    containers = pod['spec']['containers']
    for c in containers:
        res = c.get('resources', {})
        limits = res.get('limits', {})
        requests = res.get('requests', {})
        print(f'{name}:')
        print(f'  node: {node}')
        print(f'  status: {status}')
        print(f'  cpu_limit: {limits.get(\"cpu\", \"N/A\")}')
        print(f'  mem_limit: {limits.get(\"memory\", \"N/A\")}')
        print(f'  cpu_request: {requests.get(\"cpu\", \"N/A\")}')
        print(f'  mem_request: {requests.get(\"memory\", \"N/A\")}')
        print()
"

echo ""
echo "=============================================="
echo "DONE - Review the output above"
echo "=============================================="