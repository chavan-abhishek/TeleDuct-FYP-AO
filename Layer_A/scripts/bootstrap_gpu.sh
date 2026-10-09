#!/usr/bin/env bash
# One-time bootstrap on a fresh persistent volume (/workspace)
set -e

echo "=================================================="
echo "⚡ TeleDuct GPU Environment Bootstrap"
echo "=================================================="

mkdir -p /workspace/models
mkdir -p /workspace/scripts
mkdir -p /workspace/logs

if [ ! -d "/workspace/teleduct_env" ]; then
    echo "📦 Creating persistent virtualenv at /workspace/teleduct_env..."
    python3 -m venv /workspace/teleduct_env
fi

source /workspace/teleduct_env/bin/activate
pip install --upgrade pip

echo "📦 Installing Layer A Telemetry & Serving Dependencies..."
pip install \
    "sglang[all]" \
    pynvml==11.5.0 \
    psycopg2-binary==2.9.9 \
    requests==2.31.0 \
    opentelemetry-api==1.24.0 \
    opentelemetry-sdk==1.24.0 \
    opentelemetry-exporter-otlp-proto-http==1.24.0

# CUDA 13 / Ada Lovelace (sm_89) compatibility hardening:
# 1. Remove buggy C++ DLPack extension if present (prevents undefined symbol _ZNK3c106Device3strB5cxx11Ev)
pip uninstall -y torch_c_dlpack_ext 2>/dev/null || true

# 2. Ensure sglang-kernel and nvrtc match the host CUDA architecture
pip install --upgrade sglang-kernel nvidia-cuda-nvrtc

echo "✅ Environment bootstrapped successfully on /workspace!"