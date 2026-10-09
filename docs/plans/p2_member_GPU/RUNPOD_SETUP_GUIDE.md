# Layer A (GPU Provider) - RunPod Setup Guide

When you spawn a new pod on RunPod to act as Layer A, follow this guide to set up the environment from scratch.

## Pod Configuration Requirements
- **GPU:** NVIDIA L4 (or RTX 3090/4090). Minimum 24GB VRAM required for `openai/gpt-oss-20b` + KV Cache.
- **Image/Template:** Official `lmsysorg/sglang:latest` (Preferred) or standard RunPod PyTorch image with CUDA 13.
- **Exposed Ports:** 22 (SSH). Port 18000 (SGLang) and other telemetry ports will be accessed via SSH tunneling.

## 🚀 Setup Commands

After SSH'ing into the new pod, run the following commands sequentially:

### 1. Clone the Repository
```bash
git clone https://github.com/chavan-abhishek/TeleDuct-FYP-AO.git /workspace/repo
cd /workspace/repo
```

### 2. Install Dependencies
> **CRITICAL WARNING:** Do NOT use a python virtual environment (`venv`) for the PyTorch or SGLang installations here. Using a `venv` led to PyTorch ABI mismatches (e.g. `undefined symbol` and `common_ops` C++ crashes). We must use the container's native system Python.

```bash
# Force kill any stale processes just in case
pkill -9 -f sglang

# Install SGLang and its core dependencies into the system python
pip install "sglang[all]"

# Install TeleDuct specific telemetry and daemon dependencies
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http pynvml psycopg2-binary requests
```

### 3. Start Layer A
```bash
# Execute the auto-start script which loads the weights and starts the SGLang server
./Layer_A/scripts/start_layer_a.sh
```

### 4. Monitor Initialization
Open a second SSH session to the pod and monitor the logs to verify when the SGLang server becomes healthy on port 18000:
```bash
tail -f /workspace/logs/sglang.log
```
Look for: `✅ SGLang server is HEALTHY on port 18000!`

---

## 🔒 Communication & Handshake (SSH Tunnel)

As per the latest architectural decision, **we are not using raw TCP sockets for the Layer A <-> Layer B handshake**. 

Instead, use standard **SSH** for secure communication.
To connect your local Mac (Layer B) to the pod (Layer A), you should configure SSH port forwarding. 

Example SSH configuration (`~/.ssh/config`) on your Mac:
```ssh-config
Host runpod-teleduct
    HostName <runpod_ip>
    User root
    Port <runpod_ssh_port>
    IdentityFile ~/.ssh/id_ed25519
    # Forward local port 18000 to pod's 18000 for inference
    LocalForward 18000 127.0.0.1:18000
    # Add other port forwards as needed for OTel / Telemetry
```
