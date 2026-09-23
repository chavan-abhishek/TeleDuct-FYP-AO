# 03 — Architecture Decision Records (ADRs)

This document records the key architectural, technical, and infrastructure decisions made during the design and implementation of TeleDuct 3.0. It includes both successful architectural foundations and decisions that caused failures, along with their subsequent remediations.

---

## Index of Decisions
- **ADR-01:** Unified OpenTelemetry Collector & Relational Storage Pipeline (OTel Collector + PostgreSQL 16)
- **ADR-02:** User-Space Telemetry (NVML + Prometheus) for MVP Instead of In-Kernel eBPF
- **ADR-03:** Decoupled Multi-Host Development via ngrok Tunnels
- **ADR-04:** Explicit Bucket Histogram Aggregation for Probability Distributions in OpenTelemetry
- **ADR-05:** Selection of `Qwen1.5-MoE-A2.7B-Chat` as Initial Test Model *(Problematic / Superseded)*
- **ADR-06:** Architectural Pivot to `gpt-oss-20b`
- **ADR-07:** Offline Mock LLM Pipeline for Independent Layer B Development
- **ADR-08:** Pre-Quantized `compressed-tensors` Checkpoints vs. Online Quantization
- **ADR-09:** Fail-Safe Non-Blocking Exception Handling in Telemetry Emitters
- **ADR-10:** Relational PostgreSQL Schema with JSONB Over Pure NoSQL

---

## ADR-01: Unified OpenTelemetry Collector & Relational Storage Pipeline (OTel Collector + PostgreSQL 16)
- **Status:** Accepted & Implemented
- **Context:**
  The project requires both high-throughput distributed tracing (joining pre-intent states, inference forward passes, and post-intent outcomes) and durable, structured audit records for complex analytical evaluation queries (e.g., joining multi-step session graphs with logit confidence margins and hardware states to satisfy EU AI Act compliance).
- **Decision:**
  Implement a vendor-neutral telemetry pipeline centered on a local **OpenTelemetry Collector daemon** coupled with **PostgreSQL 16**:
  1. Layer B emits Chain of Intent spans via OTLP/gRPC.
  2. Layer A emits pre-sampling logit distributions, MoE routing weights, and eBPF events via OTLP/HTTP.
  3. Hardware metrics (DCGM / NVML) and SGLang engine metrics are ingested via Prometheus scrape.
  4. The OTel Collector joins streams by `session_id` and `step_id`, writing unified records to PostgreSQL 16 (and/or ClickHouse).
  5. A **Grafana SRE & Compliance Cockpit** (with native PostgreSQL datasource) reads from the database for real-time histogram, heatmap, gauge, and alert rule visualization, and a self-hosted **Langfuse** instance runs in parallel as an API-boundary comparison baseline.
- **Consequences:**
  - *Positive:* Standardizes all telemetry on open-source, vendor-neutral OpenTelemetry standards. Combines high-resolution distributed tracing with robust SQL analytical queryability for regulatory audits.
  - *Negative:* Requires managing the OTel Collector pipeline configuration and database schema synchronization.

---

## ADR-02: User-Space Telemetry (NVML + Prometheus) for MVP Instead of In-Kernel eBPF
- **Status:** Accepted (Pragmatic Trade-off)
- **Context:**
  The original research proposal specified C-based eBPF uprobes attached to `cuMemAlloc`, `cuMemFree`, and `cudaStreamSynchronize` on `libcudart.so`. However, developing and debugging eBPF programs on rented ephemeral cloud GPUs introduces severe kernel compatibility risks, root permission hurdles, and lengthy debugging cycles.
- **Decision:**
  Scope eBPF uprobes out of the core MVP. Use user-space `pynvml` (polling every 2.0s) and SGLang’s native `/metrics` Prometheus scraper (polling every 3.0s) to collect hardware and engine metrics.
- **Consequences:**
  - *Positive:* Zero kernel compilation requirements; runs cleanly in user space on any standard Linux/CUDA host. Avoided project-halting kernel debugging traps.
  - *Negative:* Polling frequency is lower (2-3 seconds vs. microsecond-level uprobes); does not capture individual per-allocation memory pointers.

---

## ADR-03: Decoupled Multi-Host Development via ngrok Tunnels
- **Status:** Accepted & Implemented
- **Context:**
  Team members develop on laptops (MacBook Air M3, ASUS VivoBook) lacking local discrete NVIDIA GPUs, while GPU inference runs on rented remote cloud instances (Vast.ai / Google Colab). A mechanism was needed to connect the distributed components securely without complex VPC peering.
- **Decision:**
  Deploy Layer B (agent orchestration), the local OpenTelemetry Collector, and PostgreSQL on developer laptops, deploy Layer A (SGLang + emitters) on the remote GPU instance, and bridge all communication using ngrok HTTP and TCP tunnels.
- **Consequences:**
  - *Positive:* Enables local development of agents, dashboards, and database queries without paying for continuous GPU uptime.
  - *Negative:* Tunnels must be re-established upon instance restart; introduces slight network latency (~50-100ms) on telemetry export.

---

## ADR-04: Explicit Bucket Histogram Aggregation for Probability Distributions in OpenTelemetry
- **Status:** Accepted & Implemented
- **Context:**
  Default OpenTelemetry histogram bucket boundaries are calibrated for request latencies (e.g., 0ms to 10,000ms+). Sampling margins ($p_1 - p_2$) and MoE routing margins are probability values strictly bounded between $0.0$ and $1.0$. Using default buckets caused all probability data to collapse into the single lowest bucket in OTel metric readers, rendering the distribution invisible.
- **Decision:**
  Explicitly define OpenTelemetry `View` aggregations in the emitter with custom bucket boundaries:
  `boundaries=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.75, 1.0]`.
- **Consequences:**
  - *Positive:* Restores high-resolution visualization of decision confidence distributions across all downstream metric consumers.
  - *Negative:* Requires explicit MeterProvider configuration before instrument instantiation.

---

## ADR-05: Selection of `Qwen1.5-MoE-A2.7B-Chat` as Initial Test Model
- **Status:** **Rejected / Problematic (Superseded by ADR-06)**
- **Context:**
  The team needed a sparse Mixture-of-Experts model that could fit on a single 24GB GPU to demonstrate MoE routing capture and expert swap dynamics.
- **Decision:**
  Selected `Qwen/Qwen1.5-MoE-A2.7B-Chat` (14.3B total parameters, 2.7B active per token).
- **Consequences (Why it failed):**
  - Native BF16 precision required ~28.6GB VRAM, causing out-of-memory crashes on 24GB cards.
  - Pre-quantized GPTQ checkpoints had tensor naming mismatches with modern SGLang Marlin kernels (`w2_bias` KeyError).
  - Most critically, `Qwen1.5-MoE` uses the `Qwen2MoeForCausalLM` architecture class, which is **not supported in current SGLang builds**, causing silent Python-level deadlocks during generation (0% GPU utilization, 15% CPU spin).

---

## ADR-06: Architectural Pivot to `gpt-oss-20b`
- **Status:** Accepted (In Progress)
- **Context:**
  Following the root-cause diagnosis of the `Qwen1.5-MoE` deadlock (ADR-05), the team needed a model family with documented, actively-maintained SGLang support.
- **Decision:**
  Pivot to `gpt-oss-20b` as the primary model, which has verified SGLang compatibility and provides the architectural features needed for TeleDuct's inference telemetry.
- **Consequences:**
  - *Positive:* Eliminates engine-level dispatch deadlocks; verified SGLang compatibility; suitable architecture for logit interception and routing telemetry.
  - *Negative:* Requires validating quantized checkpoint sizes and VRAM footprint on single-GPU rental instances.

---

## ADR-07: Offline Mock LLM Pipeline for Independent Layer B Development
- **Status:** Accepted & Implemented
- **Context:**
  Layer B agent development (session tracking, intent logging, database queries) was frequently blocked whenever remote GPU instances were stopped or undergoing engine debugging.
- **Decision:**
  Create a drop-in `MockLLM` in `layer_b/demo_agent/mock_llm.py` providing the exact same `.invoke(prompt)` interface as LangChain's `ChatOpenAI`, along with an automated test script `run_mock.py`.
- **Consequences:**
  - *Positive:* Allowed complete development, verification, and regression testing of Layer B SDK, foreign-key relationships, and database repositories on local CPU with zero cloud costs.
  - *Negative:* Mock responses are static heuristics and do not generate real logit distributions.

---

## ADR-08: Pre-Quantized `compressed-tensors` Checkpoints vs. Online Quantization
- **Status:** Accepted & Implemented
- **Context:**
  MoE models exceeding 10B total parameters cannot load in unquantized BF16 on 24GB GPUs. Attempting online FP8 quantization (`--quantization fp8`) failed with identical OOM errors because full-precision weights are staged in VRAM during initial file reading before compression occurs.
- **Decision:**
  Mandate the use of pre-quantized disk checkpoints using modern quantization formats (e.g., Neural Magic's `compressed-tensors` format with `w4a16` precision) and `--quantization compressed-tensors`.
- **Consequences:**
  - *Positive:* Peak memory during weight loading is bounded by the compressed file size (~8GB), leaving ample VRAM (~14GB) for KV-cache and CUDA buffers.
  - *Negative:* Dependent on the availability of actively-maintained community or vendor-provided pre-quantized checkpoints.

---

## ADR-09: Fail-Safe Non-Blocking Exception Handling in Telemetry Emitters
- **Status:** Accepted & Implemented
- **Context:**
  Telemetry emission occurs synchronously or asynchronously within model forward hooks and agent node wrappers. Transient network failures, database connection resets, or OTel collector timeouts could crash active LLM generation.
- **Decision:**
  Wrap all database queries and OpenTelemetry metric/span emissions in robust `try...except` blocks with warning logs, preventing downstream observability failures from interrupting primary LLM serving (satisfying NFR-2.1).
- **Consequences:**
  - *Positive:* High inference resilience; LLM continues serving even if observability sinks are temporarily unreachable.
  - *Negative:* Requires monitoring logs for silent dropped-telemetry warnings.

---

## ADR-10: Relational PostgreSQL Schema with JSONB Over Pure NoSQL
- **Status:** Accepted & Implemented
- **Context:**
  The audit trail requires relational integrity (linking parent sessions to ordered intent events and token-level logit records) while accommodating high-dimensional, dynamic arrays (top-10 candidate token strings, probability arrays, expert load distributions).
- **Decision:**
  Use PostgreSQL 16 with standard relational tables linked by UUID foreign keys, storing variable-length candidate arrays and metadata as indexed `JSONB` columns.
- **Consequences:**
  - *Positive:* Strict foreign key enforcement guarantees audit integrity for EU AI Act compliance; `JSONB` eliminates complex child-table joins for token arrays while supporting JSON operators.
  - *Negative:* Requires PostgreSQL client drivers (`psycopg2`) across both emitter and consumer nodes.
