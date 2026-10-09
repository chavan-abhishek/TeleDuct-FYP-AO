# 05 — Project Engineering Changelog & Post-Mortem Lessons

This document provides a chronological record of major project milestones, architectural iterations, and engineering lessons learned—with special emphasis on failures encountered and their systematic resolution.

---

## Chronological Development Log

### Phase 0: Research, Problem Formulation & Literature Foundation
- **Foundational Survey:** Reviewed core literature on agent failure taxonomies (MAST study, NeurIPS 2025), system-level agent tracing (AgentSight, eunomia-bpf), memory management in LLM serving (PagedAttention, SOSP 2023), and batch-invariant execution runtimes (SGLang, NeurIPS 2024).
- **Regulatory Framework Alignment:** Mapped technical requirements to **Article 12 (Record-Keeping)** and **Article 13 (Transparency)** of the **EU AI Act (Regulation (EU) 2024/1689)**, establishing the need for tamper-evident, silicon-grounded decision audit trails.
- **Problem Formulation:** Defined the 4 core sub-problems:
  - `SP-1`: Evaluation under non-determinism (floating-point non-associativity, MoE routing instability).
  - `SP-2`: Lack of "Chain of Intent" auditing at the pre-sampling layer.
  - `SP-3`: OpenTelemetry-compatible unified telemetry schema.
  - `SP-4`: Programmatic 6-category failure attribution.

---

### Phase 1: Core Design & Pipeline Implementation
- **Relational Schema (`layer_b/storage/init.sql`):** Designed a 7-table schema (`sessions`, `intent_events`, `logit_telemetry`, `moe_telemetry`, `hardware_telemetry`, `sglang_metrics`, `failure_reports`) supporting UUID foreign keys and indexed `JSONB` payloads for token distributions and expert weights.
- **Layer A Telemetry Core (`Layer_A/`):**
  - Implemented `ChainOfIntentLogitProcessor` to compute softmax distributions, sampling margins ($p_1 - p_2$), and fragility flags without mutating logits.
  - Implemented `MoERoutingCapture` and `patch_moe_router_gates` to intercept PyTorch forward passes and retain gate router logits.
  - Implemented `GPUPoller` (NVML background daemon) and `SGLangMetricsScraper` (Prometheus text parser).
- **OpenTelemetry & Database Emitter (`Layer_A/emitter.py`):**
  - Configured OpenTelemetry OTLP/HTTP export to the local OpenTelemetry Collector for distributed tracing and metric aggregation.
  - Integrated `psycopg2` direct writes to PostgreSQL for relational queryability.
- **Layer B SDK (`layer_b/sdk/`):**
  - Implemented `AgentSessionCorrelator` for node-level intent tracking and session lifecycle management.
  - Implemented `ObservabilityRepository` for relational querying.

---

### Phase 2: Local Verification & Diagnostic Discoveries
- **Offline Dry-Run Suite (`Layer_A/test_local_dryrun.py`):**
  - Built a 6-suite CPU dry-run test with `StubEmitter` to validate tensor math, margin bounds, gate patching, and Prometheus parsing without GPU requirements.
- **Layer B Mock Pipeline (`layer_b/demo_agent/run_mock.py`):**
  - Created `MockLLM` and validated 5 automated agent sessions writing 10 `intent_events` to local PostgreSQL with 100% relational integrity.
- **Explicit Histogram Bucketing Fix (OTel Metric Aggregation):**
  - *Bug Discovered:* Probability metrics ($[0.0, 1.0]$) collapsed into a single bucket under default OTel latency-oriented histograms.
  - *Fix Applied:* Configured explicit `View` aggregations with custom bucket boundaries (`[0.0, 0.05, 0.10, ..., 1.0]`), restoring fine-grained probability distribution visualization across OTel consumers.
- **Remote Tunnel Transport (`Layer_A/test_tunnel_span.py`, `test_real_emitter_local.py`):**
  - Verified span and metric transmission across ngrok HTTP/TCP tunnels to the local OpenTelemetry Collector and PostgreSQL instances.

---

### Phase 3: GPU Deployment Rental Session 1 (July 23, 2026)
*Target: Deploy quantized Qwen1.5-MoE on rented Vast.ai instance.*

- **Failures Encountered & Diagnosed:**
  1. *Invalid CLI flag:* `--quantization int4` failed because SGLang requires kernel names (`compressed-tensors`, `gptq`), not bit-widths.
  2. *Native BF16 OOM:* 14.3B total parameters in Qwen1.5-MoE required ~28.6GB VRAM, exceeding the 24GB card limit. (Lesson: MoE memory is governed by total params, not active params).
  3. *Online FP8 Staging Buffer OOM:* Online quantization failed identically because weights are staged in full precision before compression.
  4. *Stale GPTQ Checkpoint Format:* Official early-2024 GPTQ checkpoint lacked `w2_bias` tensors expected by modern SGLang Marlin kernels.
  5. *Legacy GPTQ Kernel MoE Rejection:* SGLang's legacy GPTQ method explicitly rejected MoE architectures (`TypeError: GPTQ Method does not support MoE`).
  6. *Working Quantization Found:* Neural Magic's modern `nm-testing/Qwen1.5-MoE-A2.7B-Chat-quantized.w4a16` with `--quantization compressed-tensors` loaded cleanly (7.98GB weights, 5.89GB KV cache).
  7. *CUDA Graph Capture Stall on Degraded Host:* Graph capture hung at `0/42` for 30+ minutes due to host I/O contention. (Lesson: Use `timeout` and monitor `nvidia-smi dmon` to distinguish hardware degradation from software bugs).
  8. *Watchdog Timeout on Eager Mode:* Disabling CUDA graphs forced slow eager-mode generation that exceeded the 300s default watchdog timeout.

---

### Phase 4: GPU Deployment Rental Session 2 (August 4, 2026)
*Target: Verify actual generation and logit interception on fresh healthy instance.*

- **Failures Encountered & Diagnosed:**
  1. *Health Check 503:* `/health` returned 503 despite loaded weights because SGLang's probe is tied to internal scheduler readiness.
  2. *Hypothesis Testing on Completion Hangs:* Ruled out FlashAttention-3 (`fa3`), Triton attention backends, and deterministic kernels—all produced identical hangs with 0% SM utilization.
  3. *Ghost Process Trap:* Identified that incomplete process cleanup between restarts left zombie servers on port 18000. Established mandatory `pkill -9` + `lsof -i :18000` verification.
  4. *Persistent Deadlock on MoE Generation:* Clean restarts showed `Qwen1.5-MoE` booted cleanly and `/model_info` returned `200 OK`, but generation requests (`/v1/completions` and `/generate`) hung indefinitely with 15.1% CPU consumption and 0% GPU SM utilization.

---

### Phase 5: Root Cause Discovery & Architectural Pivot (August 15, 2026)
- **Root Cause Identified:** Qwen1.5-MoE uses the `Qwen2MoeForCausalLM` architecture class, which is **not supported in current SGLang builds**. The architecture-agnostic paths (memory allocation, weights loading) succeeded, while MoE token dispatch deadlocked.
- **Architectural Pivot:** Decided to pivot to `gpt-oss-20b` as the primary model, which has verified SGLang compatibility and provides the architectural features required for TeleDuct's logit interception and routing telemetry.

---

### Phase 6: RunPod GPU Deployment & SGLang Serving Verification (October 10, 2026)
*Target: Establish a healthy, running SGLang inference server with `openai/gpt-oss-20b` on RunPod.*

- **Environment Provisioned:** RunPod instance with 1x NVIDIA L4 (24GB GDDR6, Ada Lovelace `sm_89`, CUDA 13.0) and persistent network volume mounted at `/workspace`.
- **Failures Diagnosed & Resolved:**
  1. *DLPack Symbol Collision (`torch_c_dlpack_ext`):* SGLang CLI crashed during backend autodetection due to undefined symbol `_ZNK3c106Device3strB5cxx11Ev`. Resolved by uninstalling `torch_c_dlpack_ext` (`pip uninstall -y torch_c_dlpack_ext`), which is unnecessary for standard text inference.
  2. *CUDA 13 / Ada Lovelace SM89 Kernel Mismatch:* Older `sglang-kernel 0.4.7` lacked precompiled CUDA 13 binaries for SM89 architecture. Upgraded to `sglang-kernel 0.4.9` with `torch 2.14.1` and `triton 3.8.0`, restoring native kernel dispatch.
  3. *Virtualenv Isolation:* Hardened bootstrap logic to purge cross-pod artifact contamination and cleanly manage packages on the persistent volume.
- **Milestone Verified:**
  - `openai/gpt-oss-20b` model weights (13.05 GB) loaded from `/workspace/models` cache in 154.93s.
  - CUDA graph captures completed: prefill (64.55s), decode (4.58s).
  - Internal `/model_info` returned `200 OK`.
  - Health check probe (`/health`) returned `200 OK`.
  - Triton attention kernel compiled; initial warmup prefill batch (`POST /generate`) succeeded with `200 OK`.
  - GC frozen (`POST /freeze_gc 200 OK`) and server reported: *"The server is fired up and ready to roll!"* on `127.0.0.1:18000`.

---

## Summary of Key Engineering Lessons

1. **MoE VRAM Arithmetic:** Memory capacity planning for Mixture-of-Experts models must always be computed against *total parameters*, never *active parameters*.
2. **Online vs. Pre-Quantized Storage:** Online quantization provides zero relief for load-time VRAM constraints because full-precision tensors are staged in memory first.
3. **Artifact Recency vs. Engine Evolution:** Quantized checkpoint formats age rapidly. A model quantized with legacy tooling often breaks modern optimized CUDA kernels.
4. **Diagnostic Discipline (`nvidia-smi dmon`):** Sampling SM utilization during hangs is the definitive tool to distinguish genuine software deadlocks (0% SM, high CPU) from slow hardware execution ($>0\%$ SM).
5. **Process Hygiene in GPU Environments:** Never assume terminating a parent process frees the listening socket immediately; always verify with `lsof -i :PORT` before launching new servers.
6. **Decoupled Local Dry-Runs:** Developing Layer B and telemetry sinks against a deterministic `MockLLM` decoupled application progress from GPU availability, saving extensive cloud rental costs.
7. **Cloud Volume Caching Strategy:** Caching large model snapshots (`/workspace/models`) on network storage prevents burning expensive GPU compute minutes on network downloads during fresh pod spins.
8. **C++ ABI & CUDA 13 Pinning:** When using cutting-edge host drivers (CUDA 13.0+), ensure extensions like `sglang-kernel` are aligned with the host compute capability (`sm_89`) to prevent obscure dynamic linker errors.
