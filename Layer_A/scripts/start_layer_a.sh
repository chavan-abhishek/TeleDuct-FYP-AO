#!/usr/bin/env bash
# Daily resume script — run anytime the instance starts or resumes
set -e

echo "=================================================="
echo "🚀 TeleDuct Layer A Auto-Start Sequence"
echo "=================================================="

# 1. Verify GPU presence
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ ERROR: nvidia-smi not found. GPU not attached."
    exit 1
fi
echo "✅ GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# 2. Activate Persistent Virtualenv if present
if [ -f "/workspace/teleduct_env/bin/activate" ]; then
    source /workspace/teleduct_env/bin/activate
fi

# 3. Check / Start SGLang on Port 18000
if lsof -i :18000 > /dev/null 2>&1; then
    echo "✅ SGLang server is already running on port 18000."
else
    echo "⏳ Starting SGLang server with openai/gpt-oss-20b..."
    nohup sglang serve \
        --model-path openai/gpt-oss-20b \
        --dtype bfloat16 \
        --trust-remote-code \
        --attention-backend triton \
        --mem-fraction-static 0.85 \
        --download-dir /workspace/models \
        --host 127.0.0.1 \
        --port 18000 \
        --tensor-parallel-size 1 \
        > /workspace/logs/sglang.log 2>&1 &

    echo "⏳ Waiting for SGLang /health (loading weights from cache)..."
    until curl -s http://127.0.0.1:18000/health > /dev/null 2>&1; do
        sleep 3
        echo -n "."
    done
    echo ""
    echo "✅ SGLang server is HEALTHY on port 18000!"
fi

echo "=================================================="
echo "🎯 SGLang Ready! Logs located at /workspace/logs/sglang.log"
echo "=================================================="