# 04 — Concrete Next Steps & Action Plan

This document outlines the prioritized, step-by-step roadmap for subsequent development sessions. 

---

## Session 1: Remote GPU Setup & SGLang Serving Validation
**Status:** ✅ **COMPLETED & VERIFIED (October 10, 2026)**

- [x] **Step 1.1: Provision Fresh GPU Instance & Clean Environment**
  - Deployed RunPod instance with 1x NVIDIA L4 (24GB VRAM, Ada Lovelace `sm_89`, CUDA 13.0) and attached persistent network volume (`/workspace`).
- [x] **Step 1.2: Deploy SGLang with Supported Model (`openai/gpt-oss-20b`)**
  - Fixed CUDA 13 compatibility (`sglang-kernel 0.4.9`, uninstalled conflicting `torch_c_dlpack_ext`).
  - Cached weights in `/workspace/models` (13.05 GB).
  - Started SGLang on port `18000` with Triton attention and UnifiedRadixCache.
- [x] **Step 1.3: Verify Real-Time Generation & Engine Health**
  - SGLang `/health` returned `200 OK`.
  - CUDA graph captures completed: prefill (64.55s), decode (4.58s).
  - Warmup generation (`POST /generate`) succeeded with `200 OK`.
  - Engine confirmed ready: *"The server is fired up and ready to roll!"*

---

## Session 2: Telemetry Pipeline Deployment & Remote Hook Verification (CURRENT MILESTONE)

**Goal:** Execute the joint Phase 1 handshake, stream live hardware & engine telemetry to Member 1's local hub, and wire `ChainOfIntentLogitProcessor` to the live SGLang token loop.

### Step 2.1: Phase 1 Joint End-to-End Handshake
- **Member 1 (Local):** Run `./infra/local_hub/start_local_hub.sh` and open ngrok tunnels:
  ```bash
  ngrok http 4318   # OTel Collector
  ngrok tcp 5432    # PostgreSQL
  ```
- **Member 2 (GPU):** Export the public tunnel URLs and execute the smoke test:
  ```bash
  export OTEL_ENDPOINT="https://xxxx.ngrok-free.app"
  export PG_DSN="postgresql://teleduct_admin:teleduct_secure_pass_2026@0.tcp.ngrok.io:12345/teleduct_telemetry"
  python3 ./Layer_A/scripts/test_remote_to_local_pipeline.py
  ```
- **Verification:** Confirm rows appear in PostgreSQL `hardware_telemetry` and `logit_telemetry` tables on Member 1's local machine.

### Step 2.2: Launch Persistent Background Daemons
- On GPU host, start the continuous hardware poller and Prometheus scraper:
  ```bash
  python3 ./Layer_A/scripts/run_daemons.py
  ```
- Verify NVML GPU metrics (VRAM, SM utilization, power, temperature) and SGLang engine metrics (throughput, cache usage) continuously stream into Member 1's Grafana dashboard.

### Step 2.3: Wire Live Logit Interception (`ChainOfIntentLogitProcessor`)
- Connect `ChainOfIntentLogitProcessor` to SGLang via custom logit processor flags (`--enable-custom-logit-processor`).
- Validate real-time calculation of top-1 / top-2 probability margins ($p_1 - p_2$) and fragility flags during token generation.
- Confirm live logit records stream to PostgreSQL `logit_telemetry` during active generation.

---

## Session 3: Live Agent Integration & Experimental Benchmarks

**Goal:** Connect the LangGraph compliance agent to the live engine, execute 50-run benchmarks under deterministic vs. non-deterministic conditions, configure the Grafana SRE Cockpit, and perform automated failure attribution.

### Step 3.1: Connect Layer B Agent to Live Engine & Langfuse Baseline
- In `layer_b/demo_agent/agents.py`, replace `MockLLM` with LangChain's `ChatOpenAI` pointing to SGLang's OpenAI-compatible endpoint:
  ```python
  from langchain_openai import ChatOpenAI
  llm = ChatOpenAI(
      base_url=f"{SGLANG_TUNNEL_URL}/v1",
      api_key="dummy",
      model="openai/gpt-oss-20b",
      temperature=0.0
  )
  ```
- Wrap agent reasoning steps with `AgentSessionCorrelator`.
- Send standard API traces in parallel to self-hosted Langfuse baseline.

### Step 3.2: Execute Benchmark Experiments (SP-1 through SP-4)
- **SP-1 Evaluation:** Measure trajectory consistency across 50 consecutive runs under deterministic vs. default serving.
- **SP-2 Auditing:** Verify pre-sampling logit margin capture detects decision branch flips before token emission.
- **SP-3 Telemetry:** Confirm unified OTel schema correlates hardware pressure spikes with latency anomalies.
- **SP-4 Attribution:** Run automated SQL failure classifier to categorize edge-case deviations across the 6 failure categories.
