#!/usr/bin/env bash
# Clean shutdown script for SGLang and Layer A daemons
echo "=================================================="
echo "🛑 Stopping TeleDuct SGLang & Layer A Daemons..."
echo "=================================================="

pkill -9 -f "sglang" || true
pkill -9 -f "run_daemons.py" || true

# Verify port 18000 is freed
if command -v lsof &> /dev/null; then
    PORT_PID=$(lsof -t -i :18000 2>/dev/null || true)
    if [ -n "$PORT_PID" ]; then
        echo "Killing residual process on port 18000 (PID: $PORT_PID)..."
        kill -9 $PORT_PID || true
    fi
fi

echo "✅ All SGLang server and daemon processes cleanly terminated."
echo "=================================================="
