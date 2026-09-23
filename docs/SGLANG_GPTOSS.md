# SGLang + GPT-OSS 20B — Fresh Instance Runbook

> **Target Stack:** NVIDIA RTX 3090 (24 GB) · SGLang · `openai/gpt-oss-20b` · Port `18000`
>
> **Goal:** Go from a fresh GPU instance to a working SGLang inference server in ~5 minutes (model cached).

---

## Table of Contents

1. [Connect to the Instance](#1-connect-to-the-instance)
2. [Verify GPU](#2-verify-gpu)
3. [Verify SGLang Environment Variables](#3-verify-sglang-environment-variables)
4. [Verify Workspace](#4-verify-workspace)
5. [Check Whether SGLang Is Already Running](#5-check-whether-sglang-is-already-running)
6. [Start SGLang Manually (If Not Running)](#6-start-sglang-manually-if-not-running)
7. [Successful Startup Sequence](#7-successful-startup-sequence)
8. [Verify `/health`](#8-verify-health)
9. [Verify Native SGLang `/generate`](#9-verify-native-sglang-generate)
10. [Verify OpenAI-Compatible API](#10-verify-openai-compatible-api)
11. [Clone TeleDuct-PoC](#11-clone-teleduct-poc)
12. [Final "SGLang Ready" Checklist](#12-final-sglang-ready-checklist)
13. [Continue With TeleDuct Telemetry](#13-continue-with-teleduct-telemetry)
14. [Quick Fresh-Instance Cheatsheet](#quick-fresh-instance-cheatsheet)
15. [Important Notes & Gotchas](#important-notes--gotchas)

---

## 1. Connect to the Instance

Open the instance's Jupyter terminal or SSH terminal. You should land in:

```bash
pwd
# Expected: /workspace
```

---

## 2. Verify GPU

```bash
nvidia-smi
```

**Expected output:**

| Field          | Value                       |
| -------------- | --------------------------- |
| GPU Name       | NVIDIA GeForce RTX 3090     |
| Total Memory   | 24576 MiB                   |
| Used (clean)   | ~17 MiB / 24576 MiB        |
| Processes      | No running processes found  |

> [!NOTE]
> The small amount of memory used on a clean GPU is normal.

---

## 3. Verify SGLang Environment Variables

```bash
env | grep SGLANG
```

**Expected:**

```
SGLANG_MODEL=openai/gpt-oss-20b
SGLANG_ARGS=--dtype bfloat16 --trust-remote-code --attention-backend triton --mem-fraction-static 0.85 --download-dir /workspace/models --host 127.0.0.1 --port 18000
```

**Configuration breakdown:**

| Setting               | Value                  |
| --------------------- | ---------------------- |
| Model                 | `openai/gpt-oss-20b`  |
| dtype                 | `bfloat16`             |
| Attention backend     | `triton`               |
| GPU memory fraction   | `0.85`                 |
| Model directory       | `/workspace/models`    |
| Host                  | `127.0.0.1`            |
| Port                  | `18000`                |

---

## 4. Verify Workspace

```bash
df -h /workspace
# Expected: ~25G total

ls -la /workspace
```

The important directory is `/workspace/models` — this is where the Hugging Face model snapshot is stored.

On a cached instance, the model lives at:

```
/workspace/models/models--openai--gpt-oss-20b/
```

SGLang will report:

```
Found local HF snapshot for openai/gpt-oss-20b ...; skipping download.
```

This is why subsequent startups are fast.

---

## 5. Check Whether SGLang Is Already Running

```bash
ps aux | grep sglang
lsof -i :18000
```

> [!IMPORTANT]
> If SGLang is already running, **do NOT start another server.** Test it directly instead.

```bash
timeout 30 curl -s http://localhost:18000/health
# Expected: 200
```

```bash
timeout 30 curl -s http://localhost:18000/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "What is 2+2?", "sampling_params": {"max_new_tokens": 20, "temperature": 0}}'
```

**Expected:** JSON containing generated text, e.g.:

```json
{"text":" 4\n\nWhat is 3+5? 8\n\nWhat is 7+1?", "...": "..."}
```

If this works — **SGLang is ready**, skip to [Step 8](#8-verify-health).

---

## 6. Start SGLang Manually (If Not Running)

> [!WARNING]
> Do **not** depend on the Supervisor SGLang service. Start SGLang directly.

```bash
sglang serve \
  --model-path openai/gpt-oss-20b \
  --dtype bfloat16 \
  --trust-remote-code \
  --attention-backend triton \
  --mem-fraction-static 0.85 \
  --download-dir /workspace/models \
  --host 127.0.0.1 \
  --port 18000 \
  --tensor-parallel-size 1
```

> [!CAUTION]
> This command runs in the **foreground**. Do **not** press `Ctrl+C`. Keep this terminal open and use a **second terminal** for testing and TeleDuct work.

---

## 7. Successful Startup Sequence

A successful startup progresses through these stages:

1. **Model detection**
   ```
   Detected GPT-OSS model, enabling triton_kernels MOE kernel.
   ```

2. **Weight loading**
   ```
   Load weight begin.
   Found local HF snapshot for openai/gpt-oss-20b ...; skipping download.
   Load weight end.
   ```
   Uses approximately **13.05 GB** of GPU memory for model weights.

3. **KV Cache allocation**
   ```
   KV Cache is allocated.
   ```

4. **CUDA graph capture**
   ```
   Capture target prefill CUDA graph end.
   Capture target decode CUDA graph end.
   ```

5. **Server ready** ✅
   ```
   Application startup complete.
   Uvicorn running on http://127.0.0.1:18000
   ```

---

## 8. Verify `/health`

From the **second terminal**:

```bash
timeout 30 curl -s -o /dev/null -w "%{http_code}\n" http://localhost:18000/health
```

**Expected:** `200`

---

## 9. Verify Native SGLang `/generate`

```bash
timeout 30 curl -s http://localhost:18000/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "What is 2+2?", "sampling_params": {"max_new_tokens": 20, "temperature": 0}}'
```

**Expected:** JSON with a `"text"` field containing generated output:

```json
{"text":" 4\n\nWhat is 3+5? 8\n\nWhat is 7+1?", "...": "..."}
```

> [!NOTE]
> The exact generated continuation may differ. The important thing is that `"text"` contains generated output.

---

## 10. Verify OpenAI-Compatible API

```bash
timeout 30 curl -s http://localhost:18000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"openai/gpt-oss-20b","prompt":"What is 2+2?","max_tokens":20,"temperature":0}'
```

**Expected structure:**

```json
{
  "id": "...",
  "object": "text_completion",
  "model": "openai/gpt-oss-20b",
  "choices": [
    {
      "index": 0,
      "text": " 4\n\nWhat is 3+5? 8...",
      "finish_reason": "length"
    }
  ]
}
```

If this works, the **OpenAI-compatible interface is ready** for TeleDuct.

---

## 11. Clone TeleDuct-PoC

```bash
mkdir -p /workspace/repo
cd /workspace/repo
git clone https://github.com/CHIRAL-CENTER/TeleDuct-POC.git
cd TeleDuct-POC
git checkout abhishek-layer-a
```

**Verify:**

```bash
git branch --show-current
# Expected: abhishek-layer-a
```

---

## 12. Final "SGLang Ready" Checklist

Before moving to telemetry, confirm **all** of the following:

- [ ] `nvidia-smi` → RTX 3090, 24576 MiB
- [ ] `env | grep SGLANG` → `openai/gpt-oss-20b`
- [ ] `ps aux | grep sglang` → SGLang process running
- [ ] `lsof -i :18000` → Port bound
- [ ] `curl http://localhost:18000/health` → `200`
- [ ] Native `/generate` → Returns generated text
- [ ] OpenAI `/v1/completions` → Returns JSON with `"model":"openai/gpt-oss-20b"` and generated `"text"`

---

## 13. Continue With TeleDuct Telemetry

Once the above checks pass, the infrastructure is ready. The next stage architecture:

```
SGLang (Port 18000)
    │
    ├── gpu_poller.py
    │
    └── sglang_scraper.py
            │
            ▼
       TeleDuct-PoC
            │
            ▼
        ngrok tunnel
```

Move on to the telemetry setup.

---

## Quick Fresh-Instance Cheatsheet

> For when everything is already baked into the instance and the model is cached.

**1. Verify the environment:**

```bash
nvidia-smi
env | grep SGLANG
df -h /workspace
ps aux | grep sglang
lsof -i :18000
```

**2. If SGLang is not running, start it:**

```bash
sglang serve \
  --model-path openai/gpt-oss-20b \
  --dtype bfloat16 \
  --trust-remote-code \
  --attention-backend triton \
  --mem-fraction-static 0.85 \
  --download-dir /workspace/models \
  --host 127.0.0.1 \
  --port 18000 \
  --tensor-parallel-size 1
```

> Leave that terminal running. Open another terminal.

**3. Clone the repo (if needed):**

```bash
mkdir -p /workspace/repo && cd /workspace/repo
git clone https://github.com/CHIRAL-CENTER/TeleDuct-POC.git
cd TeleDuct-POC && git checkout abhishek-layer-a
```

**4. Smoke test:**

```bash
timeout 30 curl -s -o /dev/null -w "%{http_code}\n" http://localhost:18000/health
# Expected: 200

timeout 30 curl -s http://localhost:18000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"openai/gpt-oss-20b","prompt":"What is 2+2?","max_tokens":20,"temperature":0}'
```

If that returns generated text → **SGLang is ready. Move to TeleDuct telemetry.**

---

## Important Notes & Gotchas

### ⚠️ Do Not Use the `pty` Wrapper

The instance's Supervisor script attempts to run `pty sglang serve ...`, but the `pty` command was unavailable in our environment. The reliable method is to run `sglang serve ...` directly.

### ⚠️ Do Not Rely on Supervisor's SGLang Status

We observed `sglang EXITED` under `supervisorctl status` while a manually launched SGLang server was fully functional.

> [!WARNING]
> `Supervisor: sglang EXITED` does **NOT** mean SGLang inference is unavailable.
> The authoritative test is always:
> ```bash
> curl http://localhost:18000/health
> ```
> and an actual generation request.

### 🖥️ Keep the SGLang Terminal Open

When launched manually, `sglang serve` occupies the terminal — this is expected. Use a **second terminal** for:

- curl tests
- Git operations
- Telemetry scripts
- Monitoring
- ngrok
- Other TeleDuct commands

### 💾 Model Caching

The model is cached at `/workspace/models/models--openai--gpt-oss-20b/`. When this cache exists, SGLang skips the download. A completely fresh `/workspace/models` will take **substantially longer** because the model must first be downloaded.

### 🧠 GPU Memory

| Parameter                | Value          |
| ------------------------ | -------------- |
| RTX 3090 Total           | 24576 MiB      |
| `--mem-fraction-static`  | `0.85`         |

> [!CAUTION]
> Do **not** change the memory configuration unnecessarily. The successful run loaded the model, allocated KV cache, and captured CUDA graphs without running out of GPU memory with these settings.