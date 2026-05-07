#!/bin/bash
# Setup script for IntentContinuum experiments
# Run this once on the SDN controller before running experiments

set -e

echo "=========================================="
echo "IntentContinuum Experiment Setup"
echo "=========================================="

# Check Python
echo ""
echo "🐍 Python version:"
python3 --version

# Install Locust if not present
echo ""
echo "🦗 Checking Locust..."
if python3 -c "import locust" 2>/dev/null; then
    echo "  ✅ Locust is installed"
    python3 -m locust --version
else
    echo "  📦 Installing Locust..."
    pip3 install locust --break-system-packages
    echo "  ✅ Locust installed"
fi

# Check PyYAML (needed by run_experiment.py)
echo ""
echo "📦 Checking PyYAML..."
if python3 -c "import yaml" 2>/dev/null; then
    echo "  ✅ PyYAML is installed"
else
    echo "  📦 Installing PyYAML..."
    pip3 install pyyaml --break-system-packages
    echo "  ✅ PyYAML installed"
fi

# Verify SSH to master node
echo ""
echo "🔑 Testing SSH to master node..."
MASTER_HOST=${MASTER_HOST:-10.56.0.92}
MASTER_USER=${MASTER_USER:-cc}
if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 ${MASTER_USER}@${MASTER_HOST} "echo OK" 2>/dev/null; then
    echo "  ✅ SSH to master node works"
else
    echo "  ❌ Cannot SSH to master node (${MASTER_USER}@${MASTER_HOST})"
    echo "     Fix this before running experiments!"
    exit 1
fi

# Verify local test image exists
echo ""
echo "🖼️  Checking local test image..."
LOCAL_IMAGE=${LOCAL_IMAGE:-images/family.jpg}
if [ -f "${LOCAL_IMAGE}" ]; then
    SIZE=$(ls -lh "${LOCAL_IMAGE}" | awk '{print $5}')
    echo "  ✅ Image exists: ${LOCAL_IMAGE} (${SIZE})"
else
    echo "  ❌ Image not found: ${LOCAL_IMAGE}"
    exit 1
fi

# Verify application is reachable (curl directly from sdn-controller)
echo ""
echo "🌐 Testing application endpoint..."
RT=$(curl -X POST \
    -F "image=@${LOCAL_IMAGE}" \
    -H "X-Special-Object: person" \
    http://10.56.0.92:5001/resize \
    --max-time 30 -w '%{time_total}' -o /dev/null -s 2>/dev/null) || true

if [ -n "$RT" ]; then
    echo "  ✅ Application responded in ${RT}s"
else
    echo "  ❌ Application not reachable at http://10.56.0.92:5001/resize"
    exit 1
fi

# Verify Ollama (on llm-server, not localhost)
echo ""
echo "🤖 Checking Ollama..."
OLLAMA_URL=${OLLAMA_URL:-http://10.56.3.87:11434}
if curl -s ${OLLAMA_URL}/api/tags | python3 -c "import sys,json; models=[m['name'] for m in json.load(sys.stdin)['models']]; print('  ✅ Models:', ', '.join(models))" 2>/dev/null; then
    :
else
    echo "  ❌ Ollama not responding at ${OLLAMA_URL}"
    exit 1
fi

# Create results directory
echo ""
mkdir -p results
echo "📁 Results directory: results/"

echo ""
echo "=========================================="
echo "✅ Setup complete! Ready to run experiments."
echo ""
echo "Quick start:"
echo "  python3 run_experiment.py --dry-run          # Preview the plan"
echo "  python3 run_experiment.py                    # Run default experiment"
echo "  python3 run_experiment.py --debug-llm        # Run with LLM debug logs"
echo "=========================================="
