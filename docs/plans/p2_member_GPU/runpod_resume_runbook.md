# RunPod Resume Runbook: Starting Up After Deleting a Pod

> **Purpose:** Step-by-step guide to resume work on a fresh RunPod GPU instance after terminating/deleting the previous pod to save GPU credits.  
> **Persistent Storage:** RunPod Network Volume mounted at `/workspace`.  

---

## 1. What Persists vs. What Is Ephemeral

When you terminate/delete a pod on RunPod with an attached Network Volume:

| Resource | Status After Pod Deletion | Details |
|---|---|---|
| **`/workspace/models`** | 💾 **PRESERVED** | The 13.05 GB `openai/gpt-oss-20b` weights remain cached. You do **not** need to re-download. |
| **`/workspace/repo`** | 💾 **PRESERVED** | Your TeleDuct repository clone and scripts are intact. |
| **`/workspace/teleduct_env`** | 💾 **PRESERVED** | Your Python virtual environment and installed wheels remain intact on the network volume. |
| **`/workspace/logs`** | 💾 **PRESERVED** | Previous execution logs are preserved for post-mortem analysis. |
| **Root filesystem (`/`)** | ❌ **WIPED** | Any packages installed into system Python outside `/workspace` or changes to `/root` are reset. |
| **Process State** | ❌ **TERMINATED** | Background servers and daemons are stopped. |

---

## 2. Step-by-Step Terminal Execution Next Time

When you launch a new pod on RunPod (e.g. 1x NVIDIA L4 or RTX 4090/3090, 24GB VRAM) and attach your existing Network Volume to `/workspace`:

### Step 1: Open the Web Terminal or SSH
Once the new pod status changes to **Running**, connect via SSH or open the Web Terminal.

### Step 2: Verify GPU & Persistent Volume
Verify that the GPU is attached and `/workspace` is mounted:
```bash
nvidia-smi
df -h /workspace
ls -la /workspace
```
*Expected: GPU recognized (e.g., NVIDIA L4, 24GB VRAM), `/workspace` shows available space and contains `models`, `repo` (or `TeleDuct`), and `teleduct_env`.*

---

### Step 3: Navigate to Repo & Pull Latest Updates
```bash
cd /workspace/repo
# (or cd /workspace/TeleDuct depending on directory name)
git status
git pull origin main
```

---

### Step 4: Activate Virtual Environment
Activate the persistent virtual environment:
```bash
source /workspace/teleduct_env/bin/activate
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available(), 'Torch:', torch.__version__)"
```

> [!NOTE]
> If for any reason the python symlinks in `/workspace/teleduct_env` complain about a mismatched python path on a different base image (e.g. Python 3.10 vs 3.12), simply re-link or bootstrap:
> ```bash
> pip install "sglang[all]" opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http pynvml psycopg2-binary requests
> pip uninstall -y torch_c_dlpack_ext 2>/dev/null || true
> pip install --upgrade sglang-kernel nvidia-cuda-nvrtc
> ```

---

### Step 5: Start SGLang Inference Server
Launch the automated startup script:
```bash
chmod +x ./Layer_A/scripts/*.sh
./Layer_A/scripts/start_layer_a.sh
```

**What this script does:**
1. Checks GPU availability.
2. Checks if SGLang is already running on port `18000`.
3. If not running, launches `sglang serve` with `openai/gpt-oss-20b` pointing to `/workspace/models`.
4. Polls `http://127.0.0.1:18000/health` until healthy (prints `.` every 3s).
5. Completes with `✅ SGLang server is HEALTHY on port 18000!`.

*(Optional: In a second terminal tab, monitor internal logs)*:
```bash
tail -f /workspace/logs/sglang.log
```

---

### Step 6: Test Native Generation (Quick Sanity Check)
Run a fast curl test to confirm tokens are generating:
```bash
curl -s http://127.0.0.1:18000/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, TeleDuct is online!", "sampling_params": {"max_new_tokens": 10, "temperature": 0.0}}'
```
*Expected:*
```json
{"text":" Hello, TeleDuct is online! Let's get started...","meta_info":{...}}
```

---

### Step 7: Export Member 1 Tunnel Endpoints & Run Smoke Test
Ask Member 1 for their active ngrok tunnel URLs and export them in your terminal:
```bash
# Replace with the actual ngrok URLs provided by Member 1:
export OTEL_ENDPOINT="https://xxxx-xxxx.ngrok-free.app"
export PG_DSN="postgresql://teleduct_admin:teleduct_secure_pass_2026@0.tcp.ngrok.io:12345/teleduct_telemetry"

# Run end-to-end remote-to-local pipeline verification:
python3 ./Layer_A/scripts/test_remote_to_local_pipeline.py
```
*Expected:*
```
🔍 1. Checking SGLang Health at http://127.0.0.1:18000/health...
   ✅ SGLang /health responded with status 200
🔍 2. Testing Live Generation on SGLang (gpt-oss-20b)...
   ✅ Generated in 24.5ms: "The capital of France is Paris."
🔍 3. Testing DualWriteEmitter Telemetry Dispatch...
   ✅ Telemetry packets emitted through tunnel.
==================================================
🎉 PHASE 1 REMOTE SMOKE TEST COMPLETE!
==================================================
```

---

### Step 8: Start Background Telemetry Daemons
Once the smoke test passes, start the background NVML GPU poller and SGLang scraper:
```bash
python3 ./Layer_A/scripts/run_daemons.py
```
*Leave this running in its own terminal or background it with `nohup`:*
```bash
nohup python3 ./Layer_A/scripts/run_daemons.py > /workspace/logs/daemons.log 2>&1 &
```

---

## 3. How to Cleanly Shut Down Before Deleting the Pod

Before terminating or stopping the RunPod instance at the end of a session to conserve credits:

```bash
# 1. Stop SGLang server and daemons cleanly
./Layer_A/scripts/stop_layer_a.sh

# 2. Commit and push any code or log modifications to git
cd /workspace/repo
git status
git add .
git commit -m "chore: save session checkpoint"
git push origin main
```
Then safely terminate the pod in the RunPod dashboard. Your Network Volume `/workspace` will remain safely stored.

---

## 4. Quick Command Summary Cheatsheet (Next Time Copy-Paste)

```bash
# 1. Enter repo & pull
cd /workspace/repo && git pull origin main

# 2. Activate environment
source /workspace/teleduct_env/bin/activate

# 3. Start SGLang server
./Layer_A/scripts/start_layer_a.sh

# 4. Set tunnels & smoke test (get URLs from Member 1)
export OTEL_ENDPOINT="https://<OTEL_URL>.ngrok-free.app"
export PG_DSN="postgresql://teleduct_admin:teleduct_secure_pass_2026@<TCP_HOST>:<PORT>/teleduct_telemetry"
python3 ./Layer_A/scripts/test_remote_to_local_pipeline.py

# 5. Launch daemons
python3 ./Layer_A/scripts/run_daemons.py
```
