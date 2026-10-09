# 04 — Concrete Next Steps & Action Plan

This document outlines the prioritized, step-by-step roadmap for the next **1 to 3 development sessions**. The immediate objective is to eliminate the inference blocking issue, connect the verified telemetry pipeline to live model inference, and execute the core experimental benchmarks (SP-1 through SP-4).

---

## Session 1: Remote GPU Setup & SGLang Serving Validation

**Goal:** Establish a healthy, running SGLang server with a supported model family that generates tokens cleanly without hangs or deadlocks.

### Step 1.1: Provision Fresh GPU Instance & Clean Environment
- Rent a fresh GPU instance on Vast.ai (1x RTX 3090/4090, 24GB VRAM, Ubuntu 22.04, CUDA 12.4+).
- Establish the standing process-cleanup routine to eliminate ghost processes:
  ```bash
  pkill -9 -f sglang
  lsof -i :18000
  ps aux | grep sglang
  ```

### Step 1.2: Deploy SGLang with Supported Model
- **Primary Model Candidate:** Deploy `gpt-oss-20b`:
  ```bash
  python3 -m sglang.launch_server \
      --model-path gpt-oss-20b \
      --port 18000 \
      --host 0.0.0.0 \
      --mem-fraction-static 0.85 \
      --enable-deterministic-inference \
      --enable-custom-logit-processor \
      --trust-remote-code
  ```
- Once baseline is verified, evaluate MoE routing telemetry if the model architecture supports sparse expert routing.

### Step 1.3: Verify Real-Time Generation
- Check `/health` returns `200 OK`.
- Send a test generation request and monitor GPU SM utilization with `nvidia-smi dmon -c 5 -d 1`:
  ```bash
  curl -X POST http://localhost:18000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model": "gpt-oss-20b", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 16}'
  ```
- **Success Criteria:** Generation completes in $< 1.0\text{s}$; `nvidia-smi dmon` confirms non-zero SM utilization; no deadlock or timeout.

---

## Session 2: Telemetry Pipeline Deployment & Remote Hook Verification

**Goal:** Connect Layer A instrumentation (`ChainOfIntentLogitProcessor`, `GPUPoller`, `SGLangMetricsScraper`, `DualWriteEmitter`) to the live inference engine and stream real data to the local OpenTelemetry Collector and PostgreSQL.

### Step 2.1: Establish Local Ingestion & Tunnels
- On laptop: Ensure Docker containers for the local OpenTelemetry Collector, PostgreSQL 16, and self-hosted Langfuse baseline are running.
- Open ngrok tunnels for both services:
  ```bash
  ngrok http 4318   # OpenTelemetry Collector OTLP/HTTP
  ngrok tcp 5432    # PostgreSQL TCP
  ```
- Copy the public ngrok URLs into the remote instance environment variables (`OTEL_EXPORTER_OTLP_ENDPOINT`, `PG_DSN`).

### Step 2.2: Deploy & Start Layer A Daemons on GPU Instance
- Deploy `Layer_A/` files to the GPU host.
- Start the background hardware poller and Prometheus scraper:
  ```python
  from emitter import DualWriteEmitter
  from gpu_poller import GPUPoller
  from sglang_scraper import SGLangMetricsScraper

  emitter = DualWriteEmitter(otel_endpoint=OTEL_NGROK_URL, pg_dsn=PG_NGROK_DSN)
  GPUPoller(emitter, interval=2.0).start()
  SGLangMetricsScraper(emitter, sglang_url="http://localhost:18000", interval=3.0).start()
  ```

### Step 2.3: Verify Live Logit Interception
- Wire `ChainOfIntentLogitProcessor` into the request path.
- Trigger inference requests and inspect outputs:
  - Verify `logit_capture` spans appear in the OpenTelemetry Collector stream.
  - Verify `llm.sampling_margin` histograms populate with distinct values in $[0.0, 1.0]$.
  - Verify PostgreSQL `logit_telemetry` table receives inserted records with computed `sampling_margin` and `fragility_flag`.
- **Success Criteria:** Real inference requests produce simultaneous, correlated rows in PostgreSQL and metric points in the OTel Collector stream.

---

## Session 3: Live Agent Integration & Experimental Benchmarks

**Goal:** Connect the LangGraph compliance agent to the live engine, execute 50-run benchmarks under deterministic vs. non-deterministic conditions, configure the Grafana SRE Cockpit, and perform automated failure attribution.

### Step 3.1: Connect Layer B Agent to Live Engine & Langfuse Baseline
- In `layer_b/demo_agent/agents.py`, replace `MockLLM` with LangChain’s `ChatOpenAI`:
  ```python
  from langchain_openai import ChatOpenAI
  llm = ChatOpenAI(
      base_url=f"{SGLANG_NGROK_URL}/v1",
      api_key="dummy",
      model="gpt-oss-20b",
      temperature=0.0
  )
  ```
- Wrap both agent reasoning steps (`analyze_transaction`, `flag_or_approve`) with `AgentSessionCorrelator`.
- Send standard API traces in parallel to self-hosted Langfuse (Component B5) to establish the comparison baseline.

### Step 3.2: Execute Benchmark Experiments (SP-1)
- **Experiment 1A (Batch-Invariant SGLang):** Run 50 consecutive compliance evaluations on identical transaction inputs with `--enable-deterministic-inference`.
- **Experiment 1B (Unconstrained Baseline):** Run 50 consecutive compliance evaluations without deterministic kernels (or on unconstrained vLLM).
- Record all runs in PostgreSQL.

### Step 3.3: Analyze Results & Failure Attribution (SP-2 & SP-4)
- Open the **Grafana SRE & Compliance Cockpit** (connected to PostgreSQL datasource) to visualize:
  - Real-time trace execution and logit confidence distributions.
  - Decision flip alerts and MoE routing heatmaps.
  - Side-by-side comparison cards: Langfuse API-boundary trace vs. TeleDuct hardware-grounded Chain of Intent.
- Query the database using `ObservabilityRepository`:
  - Calculate **Trajectory Consistency Score** (% identical tool call sequences across runs).
  - Calculate **Decision Flip Rate** (% of runs where final decision flipped between `APPROVE` and `FLAG`).
  - Correlate decision flips with steps where `fragility_flag = true` (sampling margin $< 0.05$).
- Implement the automated failure classifier in `layer_b/sdk/repository.py` to categorize runs into the 6 failure modes and insert records into `failure_reports`.
- **Success Criteria:** Quantitative proof that hardware-level logit margin telemetry predicts semantic decision fragility across multi-step agent runs.
