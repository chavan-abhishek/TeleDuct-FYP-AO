# Phase 1: Team Member 2 Execution & Verification Log

> **Subsystem:** TeleDuct Layer A GPU Serving & Host Telemetry  
> **Role:** Team Member 2 (Remote GPU / Serving & Telemetry Lead)  
> **Environment:** RunPod / Vast.ai Cloud GPU with Network Volume (`/workspace`)  
> **GPU Hardware:** NVIDIA L4 (24 GB GDDR6 VRAM, Ada Lovelace `sm_89`, CUDA 13.0)  
> **Model:** `openai/gpt-oss-20b` (13.05 GB weights cached)  
> **Inference Engine:** SGLang (v0.5.21) on Port `18000`  
> **Status:** ✅ Completed & Verified  

---

## 1. Remote GPU Architecture Overview

Team Member 2 operates the high-performance GPU tier of TeleDuct. The primary responsibility is running the heavy language model inference engine (**SGLang** with `openai/gpt-oss-20b`) and harvesting low-level telemetry (NVML GPU hardware metrics, engine throughput/latencies, and token logit margins) before streaming them across encrypted tunnels to Member 1's local ingestion hub.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TEAM MEMBER 2 (RunPod NVIDIA L4 GPU)                     │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Persistent Network Volume (/workspace)                             │   │
│   │                                                                     │   │
│   │  ┌──────────────────────┐   ┌───────────────────────────────────┐   │   │
│   │  │   Model Cache        │   │   Layer A Scripts & Repo          │   │   │
│   │  │   /workspace/models/ │   │   - start_layer_a.sh              │   │   │
│   │  │   (gpt-oss-20b)      │   │   - stop_layer_a.sh               │   │   │
│   │  │   [13.05 GB cached]  │   │   - bootstrap_gpu.sh              │   │   │
│   │  └──────────┬───────────┘   └─────────────────┬─────────────────┘   │   │
│   │             │                                 │                     │   │
│   │             ▼                                 ▼                     │   │
│   │  ┌──────────────────────────────────────────────────────────────┐   │   │
│   │  │  SGLang Inference Engine (Port 18000)                        │   │   │
│   │  │  - Model: openai/gpt-oss-20b                                 │   │   │
│   │  │  - UnifiedRadixCache, Triton Attention, CUDA Graph Capture   │   │   │
│   │  │  - /health (200 OK), /generate (200 OK)                      │   │   │
│   │  └──────────────────────────────┬───────────────────────────────┘   │   │
│   │                                 │ (Metrics & Hooks)                 │   │
│   │  ┌──────────────────────────────▼───────────────────────────────┐   │   │
│   │  │  Layer A Background Daemons & Emitter                        │   │   │
│   │  │  - gpu_poller.py     (NVML Hardware: VRAM, SM%, Temp, Watts) │   │   │
│   │  │  - sglang_scraper.py (Throughput, TTFT, Queue Depth)         │   │   │
│   │  │  - emitter.py        (DualWrite: OTel HTTP + PG TCP)         │   │   │
│   │  └──────────────────────────────┬───────────────────────────────┘   │   │
│   └─────────────────────────────────┼───────────────────────────────────┘   │
└─────────────────────────────────────┼───────────────────────────────────────┘
                                      │ (OTel HTTP :4318 / PG TCP :5432)
                                      ▼
                   To Team Member 1 Local Hub (via ngrok)
```

---

## 2. Configured Scripts & Source Files

The following operational scripts were implemented under `Layer_A/scripts/` to ensure full reproducibility and one-click resumption across ephemeral pod restarts:

1. **`Layer_A/scripts/start_layer_a.sh`**:
   - Inspects GPU hardware via `nvidia-smi`.
   - Activates persistent virtualenv `/workspace/teleduct_env`.
   - Checks if SGLang is already alive on port `18000`.
   - Launches SGLang in the background with `openai/gpt-oss-20b`, `--attention-backend triton`, `--mem-fraction-static 0.85`, reading weights from `/workspace/models`.
   - Polls `http://127.0.0.1:18000/health` with visual progress dots until healthy.
2. **`Layer_A/scripts/stop_layer_a.sh`**:
   - Clean shutdown utility killing SGLang and daemon background processes and verifying port `18000` is freed.
3. **`Layer_A/scripts/bootstrap_gpu.sh`**:
   - One-time persistent volume initialization script.
   - Creates `/workspace/models`, `/workspace/scripts`, `/workspace/logs`.
   - Configures the virtualenv and installs SGLang, OpenTelemetry, PyNVML, and PostgreSQL drivers.
   - Includes CUDA 13 / Ada Lovelace (`sm_89`) hardening (removes `torch_c_dlpack_ext` and updates `sglang-kernel`).
4. **`Layer_A/scripts/run_daemons.py`**:
   - Spawns `GPUPoller` (2.0s interval) and `SGLangMetricsScraper` (3.0s interval) in parallel.
   - Dispatches metrics to Member 1 via `DualWriteEmitter`.
5. **`Layer_A/scripts/test_remote_to_local_pipeline.py`**:
   - End-to-end smoke test validating SGLang health, live generation, and tunnel telemetry delivery.

---

## 3. Actual Verified Startup Log (NVIDIA L4)

The startup sequence executed successfully on RunPod:

```
==================================================
🚀 TeleDuct Layer A Auto-Start Sequence
==================================================
✅ GPU: NVIDIA L4
⏳ Starting SGLang server with openai/gpt-oss-20b...
⏳ Waiting for SGLang /health (loading weights from cache)...
............................................................................................
✅ SGLang server is HEALTHY on port 18000!
==================================================
🎯 SGLang Ready! Logs located at /workspace/logs/sglang.log
```

### SGLang Internal Engine Log (`/workspace/logs/sglang.log`):
```text
Tree cache initialized: source=default impl=UnifiedRadixCache hybrid_swa=True hybrid_ssm=False hicache_attached=False streaming_wrapped=False
Engine startup timings (s): load_weight=154.93, kv_cache_allocation=0.44, scheduler_e2e=243.89, cuda_graph={prefill=64.55, decode=4.58, target_verify=0.00, draft_prefill=0.00, draft_decode=0.00, draft_extend=0.00}, tokenizer_e2e=257.64
INFO:     Started server process [8643]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:18000 (Press CTRL+C to quit)
INFO:     127.0.0.1:41178 - "GET /model_info HTTP/1.1" 200 OK
INFO:     127.0.0.1:41198 - "GET /health HTTP/1.1" 503 Service Unavailable
Triton kernel '_fwd_kernel' took 1.31 s to compile after serving started.
Prefill batch, #new-seq: 1, #new-token: 6, #cached-token: 0, full token usage: 0.00, swa token usage: 0.00, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 0.54
INFO:     127.0.0.1:41184 - "POST /generate HTTP/1.1" 200 OK
INFO:     127.0.0.1:37790 - "POST /freeze_gc HTTP/1.1" 200 OK
The server is fired up and ready to roll!
```

---

## 4. Key Engineering Challenges & Resolutions

| Challenge / Symptom | Root Cause | Resolution |
|---|---|---|
| `torch_c_dlpack_ext` undefined symbol `_ZNK3c106Device3strB5cxx11Ev` | Incompatible prebuilt DLPack extension compiled against older PyTorch ABI. | Executed `pip uninstall -y torch_c_dlpack_ext`. Extension was unused for SGLang LLM serving. |
| `sglang-kernel` missing precompiled kernels for Ada Lovelace (`sm_89`) under CUDA 13.0 | Older `sglang-kernel 0.4.7` lacked SM89 binaries for the host CUDA driver. | Upgraded to `sglang-kernel 0.4.9` and `torch 2.14.1` with native CUDA 13 support (`pip install --upgrade sglang-kernel nvidia-cuda-nvrtc`). |
| Incomplete process termination leaving zombie sockets on port 18000 | Terminating parent process didn't tear down worker processes immediately. | Created `stop_layer_a.sh` with `pkill -9 -f sglang` and verified port release via `lsof -i :18000`. |
| Initial `/health` returning 503 before engine ready | Normal SGLang initialization lifecycle: weights load first, followed by Triton compilation and CUDA graph capture warmup. | Implemented polling loop in `start_layer_a.sh` until `/health` returns `200 OK`. |

---

## 5. Verification Checklist

- [x] **GPU Detection:** NVIDIA L4 with 24 GB VRAM detected by `nvidia-smi`.
- [x] **Model Cache:** `openai/gpt-oss-20b` weights loaded directly from `/workspace/models` without redownloading.
- [x] **Engine Initialization:** UnifiedRadixCache and Triton attention initialized.
- [x] **CUDA Graph Capture:** Prefill (64.55s) and Decode (4.58s) graph captures completed cleanly.
- [x] **Warmup Inference:** `POST /generate` succeeded with 200 OK.
- [x] **Server State:** "The server is fired up and ready to roll!" confirmed on `127.0.0.1:18000`.
