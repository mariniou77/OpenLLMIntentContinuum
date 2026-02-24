# OpenLLMIntentContinuum - Command Reference Guide

## Table of Contents
1. [Main System Commands](#1-main-system-commands)
2. [Test Suite Commands](#2-test-suite-commands)
3. [Individual Test Scripts](#3-individual-test-scripts)
4. [Infrastructure Commands](#4-infrastructure-commands)
5. [Monitoring & Debugging](#5-monitoring--debugging)
6. [Configuration](#6-configuration)

---

## 1. Main System Commands

### Start the Main System

```bash
# Basic run (continuous monitoring)
python3 main.py

# Run for a specific time window (in minutes)
python3 main.py --time-window 10

# Run with LLM debug output (shows prompts and responses)
python3 main.py --debug-llm

# Combined: 10 minute window with debug
python3 main.py --time-window 10 --debug-llm

# Show help
python3 main.py --help
```

### Main System Options

| Option | Description | Default |
|--------|-------------|---------|
| `--time-window N` | Run for N minutes then stop | None (runs forever) |
| `--debug-llm` | Log full LLM prompts and responses | Off |
| `--config PATH` | Use custom config file | `config.yaml` |

---

## 2. Test Suite Commands

### Run All Tests

```bash
# Run all 8 tests (4 actions × 2 violation types)
python3 tests/run_all_tests.py
```

### Filter Tests by Action Type

```bash
# Run only horizontal scaling tests (2 tests)
python3 tests/run_all_tests.py --action horizontal_scaling

# Run only vertical scaling tests (2 tests)
python3 tests/run_all_tests.py --action vertical_scaling

# Run only service placement tests (2 tests)
python3 tests/run_all_tests.py --action service_placement

# Run only flow scheduling tests (2 tests)
python3 tests/run_all_tests.py --action flow_scheduling
```

### Filter Tests by Violation Type

```bash
# Run only UPPER_THRESHOLD_EXCEEDED tests (4 tests)
python3 tests/run_all_tests.py --violation upper

# Run only LOWER_THRESHOLD_EXCEEDED tests (4 tests)
python3 tests/run_all_tests.py --violation lower
```

### Combine Filters

```bash
# Run horizontal scaling with UPPER violation only (1 test)
python3 tests/run_all_tests.py --action horizontal_scaling --violation upper

# Run vertical scaling with LOWER violation only (1 test)
python3 tests/run_all_tests.py --action vertical_scaling --violation lower
```

### List Available Tests

```bash
python3 tests/run_all_tests.py --list
```

---

## 3. Individual Test Scripts

### Horizontal Scaling

```bash
# Scale DOWN (LOWER_THRESHOLD_EXCEEDED - over-provisioned)
python3 tests/test_horizontal_scaling_down.py

# Scale UP (UPPER_THRESHOLD_EXCEEDED - needs more resources)
python3 tests/test_horizontal_scaling_up.py
```

**Configuration options** (edit at top of script):
```python
DEPLOYMENT_NAME = "microservice1-deployment"  # Which deployment to scale
TARGET_REPLICAS = 2                           # Target replica count
AUTO_REVERT = True                            # Revert after test?
WAIT_TIME = 15                                # Seconds to wait for changes
```

### Vertical Scaling

```bash
# Reduce resources (LOWER_THRESHOLD_EXCEEDED - over-provisioned)
python3 tests/test_vertical_scaling_down.py

# Increase resources (UPPER_THRESHOLD_EXCEEDED - needs more)
python3 tests/test_vertical_scaling_up.py
```

**Configuration options** (edit at top of script):
```python
DEPLOYMENT_NAME = "microservice1-deployment"  # Which deployment
TARGET_CPU = "200m"                           # CPU limit (millicores)
TARGET_MEMORY = "256Mi"                       # Memory limit
AUTO_REVERT = True                            # Revert after test?
WAIT_TIME = 20                                # Seconds to wait
```

### Service Placement

```bash
# Consolidate pods (LOWER_THRESHOLD_EXCEEDED)
python3 tests/test_service_placement_down.py

# Move pod to better node (UPPER_THRESHOLD_EXCEEDED)
python3 tests/test_service_placement_up.py
```

**Configuration options** (edit at top of script):
```python
DEPLOYMENT_NAME = "microservice3-deployment"  # Which deployment
TARGET_NODE = "worker1"                       # Destination node
AUTO_REVERT = True                            # Revert after test?
WAIT_TIME = 25                                # Seconds to wait
```

### Flow Scheduling

```bash
# Optimize network path (LOWER_THRESHOLD_EXCEEDED)
python3 tests/test_flow_scheduling_down.py

# Reroute traffic (UPPER_THRESHOLD_EXCEEDED)
python3 tests/test_flow_scheduling_up.py
```

**Configuration options** (edit at top of script):
```python
SOURCE_SWITCH = "of:0000000000000001"         # Source switch ID
DESTINATION_SWITCH = "of:0000000000000002"    # Destination switch ID
NEW_PATH = ["of:0000000000000001", "of:0000000000000003", "of:0000000000000002"]
AUTO_REVERT = True                            # Revert after test?
WAIT_TIME = 5                                 # Seconds to wait
```

---

## 4. Infrastructure Commands

### Kubernetes Commands (via SSH to master)

```bash
# Check cluster nodes
ssh antonios-icontinuum@10.0.0.100 'sudo kubectl get nodes'

# Check deployments
ssh antonios-icontinuum@10.0.0.100 'sudo kubectl get deployments'

# Check pods
ssh antonios-icontinuum@10.0.0.100 'sudo kubectl get pods -o wide'

# Check pod resources
ssh antonios-icontinuum@10.0.0.100 'sudo kubectl describe deployment microservice1-deployment'

# Manual scaling
ssh antonios-icontinuum@10.0.0.100 'sudo kubectl scale deployment microservice1-deployment --replicas=2'

# Manual resource change
ssh antonios-icontinuum@10.0.0.100 'sudo kubectl set resources deployment microservice1-deployment --limits=cpu=400m,memory=400Mi'

# Check logs of a pod
ssh antonios-icontinuum@10.0.0.100 'sudo kubectl logs <pod-name>'
```

### ONOS Commands

```bash
# Check ONOS devices (switches)
curl -u onos:rocks http://localhost:8181/onos/v1/devices

# Check ONOS links
curl -u onos:rocks http://localhost:8181/onos/v1/links

# Check ONOS hosts
curl -u onos:rocks http://localhost:8181/onos/v1/hosts

# Check ONOS intents
curl -u onos:rocks http://localhost:8181/onos/v1/intents

# Delete all intents
curl -u onos:rocks -X DELETE http://localhost:8181/onos/v1/intents
```

### sFlow-RT Commands

```bash
# Check sFlow-RT status
curl http://localhost:8008/version

# Check agents
curl http://localhost:8008/agents/json

# Check metrics
curl http://localhost:8008/metric/ALL/json
```

### Ollama (LLM) Commands

```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Test LLM generation
curl http://localhost:11434/api/generate -d '{
  "model": "tinyllama",
  "prompt": "Hello",
  "stream": false
}'

# Pull a model
ollama pull tinyllama

# List models
ollama list

# Run interactive chat
ollama run tinyllama
```

---

## 5. Monitoring & Debugging

### View Logs

```bash
# Run with verbose output
python3 main.py --debug-llm 2>&1 | tee experiment.log

# View saved experiment results
cat experiment_results_*.json | python3 -m json.tool
```

### Check Component Health

```bash
# Quick health check (built into main.py)
python3 -c "
from data_collector import DataCollector
import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)

dc = DataCollector(config)
health = dc.get_health_status()
print('Component Health:')
for component, status in health.items():
    symbol = '✅' if status else '❌'
    print(f'  {symbol} {component}: {\"healthy\" if status else \"unhealthy\"}')
"
```

### Test Individual Components

```bash
# Test Kubernetes connection
python3 -c "
from utils.kubernetes_client import KubernetesClient
k8s = KubernetesClient('10.0.0.100')
nodes = k8s.get_nodes()
print(f'Found {len(nodes)} nodes')
for n in nodes:
    print(f'  - {n[\"name\"]}: {n[\"status\"]}')
"

# Test ONOS connection
python3 -c "
from utils.onos_client import ONOSClient
onos = ONOSClient('http://localhost:8181', 'onos', 'rocks')
devices = onos.get_devices()
print(f'Found {len(devices)} devices')
for d in devices:
    print(f'  - {d[\"id\"]}: {\"available\" if d[\"available\"] else \"unavailable\"}')
"

# Test sFlow-RT connection
python3 -c "
from utils.sflow_client import SFlowRTClient
sflow = SFlowRTClient('http://localhost:8008')
print(f'sFlow-RT healthy: {sflow.is_healthy()}')
"
```

---

## 6. Configuration

### Main Configuration File (config.yaml)

```yaml
# Intent thresholds
intent:
  upper_threshold: 3.0      # Max acceptable response time (seconds)
  lower_threshold: 1.0      # Min acceptable response time (seconds)
  ema_alpha: 0.02           # EMA smoothing factor

# Timing
timing:
  check_interval: 5         # Seconds between checks
  wait_after_action: 60     # Seconds to wait after taking action

# History
history:
  max_entries: 3            # Number of past decisions to include in prompt

# Endpoints
endpoints:
  kubernetes_master: "10.0.0.100"
  onos: "http://localhost:8181"
  onos_user: "onos"
  onos_password: "rocks"
  sflow_rt: "http://localhost:8008"
  application: "http://10.132.0.14:5001/resize"

# LLM
llm:
  model: "tinyllama"
  base_url: "http://localhost:11434"

# Enabled actions
actions:
  horizontal_scaling: true
  vertical_scaling: true
  service_placement: true
  flow_scheduling: true

# Deployment constraints
kubernetes:
  deployments:
    - name: "microservice1-deployment"
      min_replicas: 1
      max_replicas: 5
      default_cpu: "300m"
      default_memory: "312Mi"
    - name: "microservice3-deployment"
      min_replicas: 1
      max_replicas: 5
      default_cpu: "500m"
      default_memory: "512Mi"
```

### Environment Variables (optional)

```bash
# Override config file location
export INTENT_CONFIG_PATH=/path/to/custom/config.yaml

# Override LLM endpoint
export OLLAMA_HOST=http://localhost:11434
```

---

## Quick Reference Card

### Most Common Commands

| Task | Command |
|------|---------|
| Start system (10 min) | `python3 main.py --time-window 10` |
| Start with debug | `python3 main.py --time-window 10 --debug-llm` |
| Run all tests | `python3 tests/run_all_tests.py` |
| Test horizontal scaling | `python3 tests/run_all_tests.py --action horizontal_scaling` |
| Test single action | `python3 tests/test_horizontal_scaling_up.py` |
| List tests | `python3 tests/run_all_tests.py --list` |
| Check pods | `ssh antonios-icontinuum@10.0.0.100 'sudo kubectl get pods -o wide'` |
| Check ONOS | `curl -u onos:rocks http://localhost:8181/onos/v1/devices` |

### Test Script Summary

| Script | Action | Violation | Effect |
|--------|--------|-----------|--------|
| `test_horizontal_scaling_down.py` | Scale replicas | LOWER | Reduce replicas |
| `test_horizontal_scaling_up.py` | Scale replicas | UPPER | Increase replicas |
| `test_vertical_scaling_down.py` | Change resources | LOWER | Reduce CPU/memory |
| `test_vertical_scaling_up.py` | Change resources | UPPER | Increase CPU/memory |
| `test_service_placement_down.py` | Move pod | LOWER | Consolidate pods |
| `test_service_placement_up.py` | Move pod | UPPER | Move to better node |
| `test_flow_scheduling_down.py` | Network path | LOWER | Optimize path |
| `test_flow_scheduling_up.py` | Network path | UPPER | Reroute traffic |

---

## Troubleshooting

### Common Issues

**1. "Deployment not found"**
```bash
# Check actual deployment names
ssh antonios-icontinuum@10.0.0.100 'sudo kubectl get deployments'
```

**2. "ONOS connection failed"**
```bash
# Check ONOS is running
curl -u onos:rocks http://localhost:8181/onos/v1/devices
```

**3. "LLM not responding"**
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags
# Restart if needed
sudo systemctl restart ollama
```

**4. "SSH connection failed"**
```bash
# Test SSH to master node
ssh -o StrictHostKeyChecking=no antonios-icontinuum@10.0.0.100 'echo OK'
```

**5. Tests fail with "State may not be fully reverted"**
- This warning is usually harmless - it means node assignments changed during pod recreation
- The actual resources/replicas are correctly reverted
