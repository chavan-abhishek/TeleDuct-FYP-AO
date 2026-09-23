# 00 — Product Vision & System Requirements Specification (SRS)

## 1. Executive Summary & Product Vision

### 1.1 The Vision
**TeleDuct 3.0** is an out-of-process, hardware-grounded telemetry and observability middleware designed to resolve the **"Inference Wall"** in autonomous, multi-agent AI systems.

Existing application-layer observability frameworks (e.g., Langfuse, LangSmith, Arize Phoenix) operate strictly at the post-sampling API boundary. They capture text prompts, tool calls, and post-hoc JSON outputs, treating the underlying GPU runtime as an opaque black box.

TeleDuct bridges high-level agent reasoning spans (LangGraph DAG execution states) directly with physical, silicon-level execution parameters:
- **Pre-sampling logits & candidate token probability distributions**
- **Top-2 sampling confidence margins** ($p_1 - p_2$) and decision fragility flags
- **Mixture-of-Experts (MoE) routing weights and expert selection margins**
- **Dynamic batch composition, padding overhead, and RadixAttention KV-cache hit rates**
- **Kernel-space memory allocation byte sizes and device synchronization latencies**

By fusing semantic agent execution context with in-engine hardware signals via an OpenTelemetry pipeline, TeleDuct produces a **hardware-grounded, tamper-evident Chain of Intent audit record**. This enables engineers to eliminate execution non-determinism, debug silent logical failures, and satisfy the strict record-keeping mandates of **Article 12 (Record-Keeping)** and **Article 13 (Transparency)** of the **EU AI Act (Regulation (EU) 2024/1689)** with under 4% runtime CPU overhead.

```
API-Boundary Observers (LangSmith, Langfuse)
|================== THE INFERENCE WALL ==================|
|                                                        |
[LangGraph] ---> [HTTP Client] ---> [GPU INFERENCE ENGINE] ---> [JSON Output]
                                     - Logit Margins (Discarded)
                                     - MoE Expert Routing (Discarded)
                                     - VRAM Allocation (Unmonitored)
                                     ^
                                     | TeleDuct 3.0 Observes Here (In-Engine)
```

---

## 2. The Core Problem Statement & 4 Sub-Problems (SP-1 to SP-4)

### 2.1 The Meta-Problem: The Semantic-to-Hardware Disconnect
Production inference platforms rely on dynamic request batching, non-contiguous memory management (PagedAttention), and iteration-level scheduling to maximize throughput. However, this introduces physical sources of non-determinism:
1. **Floating-Point Non-Associativity:** Parallel GPU reduction blocks execute non-associative additions ($(a+b)+c \neq a+(b+c)$). Dynamic thread scheduling shifts reduction order, introducing floating-point variations ($\sim 10^{-5}$) that flip token selection near decision boundaries even at `temperature = 0` (greedy decoding).
2. **MoE Routing Instability:** Small numerical deviations in router scores cause discrete expert swaps (e.g., swapping a Logic expert for a Creative expert), radically altering downstream agent trajectories.
3. **Batch Padding & KV-Cache Contention:** Variable sequence lengths require padding and alter attention matrix dimensions, triggering prefix cache evictions and memory bandwidth thrashing.

Because these failures occur at the semantic layer, the inference server still returns `HTTP 200 OK`. Legacy APMs show green health checks while the agent silently enters infinite loops or violates compliance policies.

### 2.2 The 4 Sub-Problems
- **SP-1: Evaluation Under Non-Determinism:** Measuring agent quality when identical inputs produce divergent execution paths across runs. TeleDuct measures Trajectory Consistency Score, Decision Flip Rates, and run variance as physical signals.
- **SP-2: Lack of "Chain of Intent" Auditing:** Compliance teams cannot prove *why* an agent made a high-stakes decision. TeleDuct records pre-execution intent, alternative candidate tool probabilities, top-k logit margins, and MoE routing weights.
- **SP-3: Standardized Audit Log Schema:** Creating an OpenTelemetry-compatible audit schema where low-level hardware contexts (VRAM allocation, kernel execution counts, power consumption) are structured, mandatory, and queryable fields.
- **SP-4: Failure Attribution Across Agent Steps:** Programmatically attributing multi-step agent failures to one of 6 discrete categories:
  1. *Model Reasoning Failure*
  2. *Tool Execution Failure*
  3. *Parameter Generation Failure*
  4. *Input Data Failure*
  5. *Guardrail Over-Trigger*
  6. *Batch-Induced Hardware Variance*

---

## 3. Target User Personas

| Persona | Primary Objective | Current Pain Point | How TeleDuct Solves It |
| :--- | :--- | :--- | :--- |
| **ML Infrastructure / SRE** | Maximize GPU utilization, manage VRAM occupancy, and prevent Out-Of-Memory (OOM) faults. | Cannot identify which specific agent reasoning step caused a sudden VRAM spike, KV-cache thrashing, or kernel latency bottleneck. | Provides per-span VRAM attribution via low-frequency eBPF memory probes and tracks RadixAttention cache hit rates. |
| **AI Systems Architect** | Ensure 100% execution consistency and predictability of multi-step agent graphs. | Lacks visibility into decision fragility, token-level probability margins, and MoE routing shifts. | Intercepts pre-sampling logits, extracts candidate token margins, and flags fragile decisions where top-2 probability margin $< 0.05$. |
| **GRC & Compliance Auditor** | Verify that autonomous agent decisions comply with legal and organizational policies (EU AI Act). | Relies on post-hoc self-reported agent summaries, which may hallucinate rationale rather than reflect actual execution logic. | Generates immutable, hardware-grounded audit trails proving exact mathematical confidence and physical system state of every decision. |

---

## 4. Functional Requirements (FRs)

### 4.1 Layer A: Hardware-Instrumented Inference
- **FR-1.1 (Engine Runtime):** Deploy SGLang hosting an open-weights model (e.g., `gpt-oss-20b`) with `--enable-deterministic-inference` backed by batch-invariant CUDA kernels.
- **FR-1.2 (Pre-Sampling Logit Interception):** `ChainOfIntentLogitProcessor` shall intercept unnormalized logits before sampling, extract top-10 candidate token IDs with softmax probabilities, compute rank-1 vs. rank-2 sampling margin ($p_1 - p_2$), and assert `fragility_flag = true` when margin $< 0.05$.
- **FR-1.3 (MoE Router Telemetry):** PyTorch forward hooks (`register_forward_hook`) registered on router modules shall capture expert IDs, routing weights, and routing margins per token pass.
- **FR-1.4 (Batch Composition Profiling):** `InstrumentedScheduler` shall capture dynamic batch sizes, sequence padding lengths, memory allocated, and calculate a `batch_stability_score` (flagging variance risks when stability $< 0.5$ or memory pressure $> 85\%$).
- **FR-1.5 (KV-Cache Hit Rate Tracking):** Monitor SGLang’s RadixAttention prefix reuse to measure cache hit rates and evaluate Time to First Token (TTFT) performance.
- **FR-1.6 (Low-Frequency eBPF Probes):** A C-based `libbpf` program attaches uprobes to `cuMemAlloc`, `cuMemFree`, and `cudaStreamSynchronize` on `libcudart.so`, capturing memory churn and synchronization latency while avoiding hot-path kernel launch overhead.

### 4.2 Layer B: Observability & Explainability
- **FR-2.1 (Agent Execution Graph):** Orchestrate multi-step agent workflows using LangGraph, modeling reasoning nodes, tool execution edges, and conditional state transitions explicitly (supporting Financial Compliance, Research Summarization, and Policy Drafting agent types).
- **FR-2.2 (Out-of-Process Context Propagation):** `AgentSessionCorrelator` SDK generates unique `session_id` and `step_id` pairs before each node executes, writing the active OTel `TraceId`/`SpanId` to Thread-Local Storage / Unix Domain Socket (UDS) `/dev/shm`, and setting Python `ContextVar`.
- **FR-2.3 (Guardrail Logging):** NVIDIA NeMo Guardrails logs every evaluation event (pass, block, escalate, matched rule ID, version) as a first-class field in the audit schema.
- **FR-2.4 (Replay-Based Evaluation Harness):** Re-execute recorded agent decision logic deterministically offline against mocked tool outputs to evaluate trajectory consistency without live external API dependencies.
- **FR-2.5 (Automated Failure Attribution):** Query the persistent store and classify failed agent runs into the 6 discrete categories: model reasoning failure, tool execution failure, parameter generation failure, input data failure, guardrail over-trigger, or batch-induced variance.

### 4.3 Telemetry Pipeline & Storage
- **FR-3.1 (OTel Ingestion):** A local OpenTelemetry Collector shall ingest gRPC and HTTP OTLP streams from Layer A (logit/MoE/eBPF) and Layer B (Chain of Intent SDK), plus Prometheus scrapes from DCGM, merging them by `session_id` and `step_id`.
- **FR-3.2 (Isolated Database Persistence):** Fused records shall be written to an isolated PostgreSQL 16 database (and/or ClickHouse container pinned via `taskset -c 4` to prevent database writes from competing with inference CPU cores).
- **FR-3.3 (Visual SRE & Compliance Cockpit):** A Grafana dashboard (with native PostgreSQL datasource) shall render real-time logit probability margin histograms, MoE routing heatmaps, hardware gauge panels, and threshold-based alert rules that fire when a `fragility_flag` or batch variance event occurs. Self-hosted Langfuse runs in parallel as the agent-level tracing baseline.
- **FR-3.4 (Baseline Comparison):** Deploy self-hosted Langfuse in parallel as the comparative baseline to visually demonstrate the observability gap between API-boundary tracing and in-engine hardware-grounded tracing.

---

## 5. Non-Functional Requirements (NFRs)

### 5.1 Performance & Overhead ("The Agent Tax")
- **NFR-1.1:** Telemetry interception pipeline shall introduce $\le 5\%$ latency overhead to baseline Time Per Output Token (TPOT).
- **NFR-1.2:** Low-frequency eBPF kernel probes and hardware polling shall consume $< 4\%$ total host CPU utilization.
- **NFR-1.3:** Context propagation over Unix Domain Socket (UDS) / Thread-Local Storage shall execute in $< 2\text{ ms}$ per node transition.

### 5.2 Stability & Reliability
- **NFR-2.1:** Telemetry failures (e.g., buffer exhaustion, network glitch, or collector crash) must fail silently in user space without terminating or interrupting primary SGLang inference.
- **NFR-2.2:** SGLang instrumentation must rely strictly on public Python extension points (`CustomLogitProcessor`, PyTorch hooks) to ensure compatibility across minor engine updates without forking engine source code.

### 5.3 Compliance & Auditability
- **NFR-3.1:** All stored audit logs must include immutable timestamps, session UUIDs, logit confidence margins, and physical memory allocation bytes to satisfy Article 12 (Record-Keeping) and Article 13 (Transparency) of the EU AI Act.

---

## 6. Unified Data Schema Specification

Every reasoning step produces a single, immutable, OTel-compatible record stored in the telemetry database:

```json
{
  "intent_context": {
    "session_id": "sess_89a7f2e1_20260825",
    "step_id": "step_03_policy_compliance",
    "node_name": "verify_transaction_limit",
    "parent_goal": "audit_incoming_wire_transfers",
    "current_objective": "verify_if_amount_exceeds_aml_threshold",
    "available_tools": [
      "flag_transaction",
      "request_human_review",
      "bypass_alert"
    ],
    "selected_action": "flag_transaction",
    "selection_rationale": "Transaction amount of $50,000 exceeds strict AML threshold.",
    "guardrail_checks": [
      {
        "id": "nemo_financial_v1",
        "rule": "wire_limit",
        "status": "passed"
      }
    ]
  },
  "logit_telemetry": {
    "top_10_candidates": [
      "flag_transaction",
      "request_human_review",
      "bypass_alert"
    ],
    "top_10_probs": [0.510001, 0.489999, 0.000000],
    "sampling_margin": 0.020002,
    "fragility_flag": true
  },
  "moe_telemetry": {
    "expert_ids": [1, 5],
    "expert_weights": [0.51, 0.49],
    "routing_margin": 0.020002
  },
  "batch_telemetry": {
    "batch_id": "batch_20260825_1200_001",
    "batch_size": 4,
    "padding_applied": 184,
    "batch_stability_score": 0.48,
    "memory_pressure": "87%"
  },
  "ebpf_telemetry": {
    "allocation_size_bytes": 1048576,
    "sync_latency_us": 1420
  },
  "hardware_telemetry": {
    "gpu_sm_utilization": 87,
    "gpu_power_watts": 284,
    "temperature_celsius": 74
  }
}
```

---

## 7. Experimental Validation & Evaluation Plan

1. **Experiment 1 (Trajectory Variance Baseline - SP-1):** Run an identical 3-step compliance agent 50 times on SGLang (`--enable-deterministic-inference`) vs. 50 times on unconstrained vLLM. Calculate Trajectory Consistency Score and Decision Flip Rate to prove batch-invariant kernels stabilize agent reasoning paths.
2. **Experiment 2 (Hardware-Induced Anomaly - SP-4):** Execute identical agent prompts under synthetic CPU/memory noise applied via `stress-ng --cpu 4 --io 2 --vm 2`. Document how physical scheduling jitter alters pre-sampling logit margins ($< 0.05$), inducing trajectory divergence.
3. **Experiment 3 (Batch Padding & KV-Cache Eviction Analysis):** Generate concurrent background requests with variable sequence lengths (50 to 2,000 tokens). Capture the drop in `batch_stability_score` and spike in TTFT (from $< 5\text{ ms}$ to $> 500\text{ ms}$) caused by prefix eviction in RadixAttention.
