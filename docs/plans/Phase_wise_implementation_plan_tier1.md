# TeleDuct: Tier-1 Implementation & Execution Plan
## Full Schema Visibility, Deep Intent Audit & Hardware Thermal Grounding

> **Milestone Status:** Phase 1 (Foundations), Phase 2 (Logit Interception), and Phase 3 (MoE Routing) Successfully Verified (October 2026)  
> **Target Subsystem:** Tier 1 — Complete Schema Visibility, Chain of Intent Deep Audit, GPU Thermal Telemetry, and Candidate Logit Competition  
> **Team Structure:** Two-Member Split (Member 1: Local Hub, Analytical SQL & Grafana Lead | Member 2: Remote GPU, NVML & SGLang Serving Lead)  
> **Documentation Target:** `docs/plans/Phase_wise_implementation_plan_tier1.md`

---

## Table of Contents
1. [Executive Context & Multi-Tier Roadmap](#1-executive-context--multi-tier-roadmap)
2. [Tier-1 Architecture & Telemetry Data Flow](#2-tier-1-architecture--telemetry-data-flow)
3. [Mathematical & Theoretical Foundations](#3-mathematical--theoretical-foundations)
4. [Team Member 1 Execution Guide (Local Hub & Analytical Observability)](#4-team-member-1-execution-guide-local-hub--analytical-observability)
   - [Task 1.1: Local Infrastructure Resumption & Port Verification](#task-11-local-infrastructure-resumption--port-verification)
   - [Task 1.2: Database Migration for Tier-1 Analytical Views (`views_tier1_audit.sql`)](#task-12-database-migration-for-tier-1-analytical-views-views_tier1_auditsql)
   - [Task 1.3: Provision Grafana SRE & Intent Audit Cockpit (`sre_intent_cockpit.json`)](#task-13-provision-grafana-sre--intent-audit-cockpit-sre_intent_cockpitjson)
   - [Task 1.4: Enhanced Compliance Agent with Full Intent Metadata (`run_audit_agent.py`)](#task-14-enhanced-compliance-agent-with-full-intent-metadata-run_audit_agentpy)
5. [Team Member 2 Execution Guide (Remote GPU & Hardware Instrumentation)](#5-team-member-2-execution-guide-remote-gpu--hardware-instrumentation)
   - [Task 2.1: GPU Pod Resumption & SSH Triple Tunnel Verification](#task-21-gpu-pod-resumption--ssh-triple-tunnel-verification)
   - [Task 2.2: Thermal & NVML Hardware Poller Verification](#task-22-thermal--nvml-hardware-poller-verification)
   - [Task 2.3: Top-10 Candidate Logit Interception Engine](#task-23-top-10-candidate-logit-interception-engine)
   - [Task 2.4: Tier-1 Trajectory Verification Harness](#task-24-tier-1-trajectory-verification-harness)
6. [Joint End-to-End Verification Protocol](#6-joint-end-to-end-verification-protocol)
7. [Troubleshooting & Gotchas Matrix](#7-troubleshooting--gotchas-matrix)
8. [Tier-1 Definition of Done & Sign-off Checklist](#8-tier-1-definition-of-done--sign-off-checklist)

---

## 1. Executive Context & Multi-Tier Roadmap

### 1.1 The Core Visibility Gap
Our telemetry database (`teleduct_telemetry`) currently ingests high-frequency physical and semantic data from both the GPU and the local agent. However, an analysis of the unified schema against the active Grafana cockpits revealed an **observability gap**:
* **Deep Intent Context is Hidden:** `parent_goal`, `current_objective`, `selection_rationale`, `available_tools`, and `guardrail_checks` are recorded in PostgreSQL `intent_events`, but were omitted from SQL correlation views and dashboard tables. Compliance auditors cannot see *why* an agent selected an action.
* **Thermal Grounding is Omitted:** GPU temperature ($T_{\text{GPU}}$) is logged by NVML into `hardware_telemetry.gpu_temperature_c`, but missing a visual gauge alongside VRAM and Power Watts.
* **Candidate Logit Competition is Unrendered:** SGLang provides top-10 candidate logprobs, but they are stored as JSONB without a visual distribution bar chart showing the margin between the winner and runner-up tokens.

#### Unified Schema Field Audit & Compliance Matrix

| Telemetry Subsystem | Field Name | Data Type | PostgreSQL Table & Column | Grafana Panel Representation | Regulatory Compliance Mandate |
|---|---|---|---|---|---|
| **Intent Context** | `session_id` | `UUID` | `sessions.session_id` / `intent_events.session_id` | Session Filter Variable | Traceability (ISO 42001) |
| **Intent Context** | `step_id` | `VARCHAR(128)` | `intent_events.step_id` | Primary Key in Audit Table | Deterministic Step Replay |
| **Intent Context** | `node_name` | `VARCHAR(64)` | `intent_events.node_name` | Flow State in Audit Table | Graph Node Attribution |
| **Intent Context** | `parent_goal` | `VARCHAR(128)` | `intent_events.parent_goal` | High-Level Objective Column | EU AI Act Art. 12 (Record-Keeping) |
| **Intent Context** | `current_objective` | `VARCHAR(256)` | `intent_events.current_objective` | Micro-Step Target Column | EU AI Act Art. 13 (Transparency) |
| **Intent Context** | `available_tools` | `JSONB` (Array) | `intent_events.available_tools` | Action Space Inspection | Tool Sandboxing & Bound Checks |
| **Intent Context** | `selected_action` | `VARCHAR(64)` | `intent_events.selected_action` | Executed Decision Column | Chain of Custody & Audit Trail |
| **Intent Context** | `selection_rationale` | `TEXT` | `intent_events.selection_rationale` | Explanatory Justification Column | EU AI Act Art. 13 (Explainability) |
| **Intent Context** | `guardrail_checks` | `JSONB` (Object/Array) | `intent_events.guardrail_checks` | Compliance Verification Badge | Safety Pre-Execution Assurance |
| **Logit Telemetry** | `top_10_candidates` | `JSONB` (Array) | `logit_telemetry.top_k_token_ids` | Token Names in Bar Chart | Decision Boundary Analysis |
| **Logit Telemetry** | `top_10_probs` | `JSONB` (Array) | `logit_telemetry.top_k_probs` | Probabilities in Bar Chart | Pre-Sampling Confidence Curve |
| **Logit Telemetry** | `sampling_margin` | `FLOAT` | `logit_telemetry.sampling_margin` | Top-2 Margin Gauge & Histogram | Fragility Detection ($M < 0.05$) |
| **Hardware State** | `gpu_temperature_c` | `FLOAT` | `hardware_telemetry.gpu_temperature_c` | Thermal Gauge (Alert $\ge 82^\circ\text{C}$) | Thermal Throttling Resilience |

### 1.2 Multi-Tier Implementation Roadmap
To achieve 100% compliance with the Product Vision & System Requirements Specification (SRS), remaining work is divided into 4 prioritized tiers:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ TIER 1: FULL SCHEMA VISIBILITY & AUDIT GROUNDING (< 45 mins) [THIS DOCUMENT]          │
│ - Unpack deep intent context (goals, objectives, rationale, tools, guardrail checks)   │
│ - Surface NVML GPU Temperature (°C) gauge with thermal throttling alerts               │
│ - Render Top-K candidate logit probability competition charts in Grafana               │
└────────────────────────────────────────┬───────────────────────────────────────────────┘
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ TIER 2: NEMO GUARDRAILS & 6-CATEGORY FAILURE ATTRIBUTION (1 - 2 hours)                 │
│ - NVIDIA NeMo Guardrails (`nemoguardrails`) Colang financial policies                  │
│ - Automated 6-category failure classifier (Reasoning, Tool, Schema, Data, Guardrail, HW│
│ - Deterministic offline replay harness (`replay_harness.py`)                           │
└────────────────────────────────────────┬───────────────────────────────────────────────┘
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ TIER 3: BATCH COMPOSITION & SGLANG KV-CACHE TRACKING (2 - 3 hours)                     │
│ - Dynamic batch size, padding token count & `batch_stability_score`                    │
│ - SGLang RadixAttention prefix cache hit/miss tracking vs TTFT latency                 │
│ - New `batch_telemetry` PostgreSQL table and Grafana timeline                          │
└────────────────────────────────────────┬───────────────────────────────────────────────┘
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ TIER 4: LOW-FREQUENCY eBPF CUDA PROBES & 50-RUN BENCHMARKS (4 - 6 hours)               │
│ - C `libbpf` uprobes on `cuMemAlloc`, `cuMemFree`, and `cudaStreamSynchronize`         │
│ - Self-hosted Langfuse comparative baseline container in Docker Compose                │
│ - 50-run determinism benchmark (Experiment 1) & `stress-ng` contention (Experiment 2)  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Tier-1 Architecture & Telemetry Data Flow

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ TEAM MEMBER 1 (Local Mac Workstation)                                                 │
│                                                                                        │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │ Layer B: Audit-Grounded Compliance Agent (`run_audit_agent.py`)                  │  │
│  │ - Session ID: UUID per trajectory                                                │  │
│  │ - Populates 100% intent fields: parent_goal, rationale, tools, guardrail_checks  │  │
│  │ - Emits intent events to PostgreSQL (host port 5433)                             │  │
│  └──────────────────────────────────────┬───────────────────────────────────────────┘  │
│                                         │                                              │
│  ┌──────────────────────────────────────┼───────────────────────────────────────────┐  │
│  │ Local Ingestion Hub (Docker Compose) │                                           │  │
│  │                                      │                                           │  │
│  │  ┌───────────────────┐    ┌──────────▼────────┐    ┌──────────────────────────┐  │  │
│  │  │ PostgreSQL 16     │    │ OTel Collector    │    │ Prometheus 2.51          │  │  │
│  │  │ (Host Port 5433)  │    │ (Host Port 4318)  │    │ (Host Port 9090)         │  │  │
│  │  │ - views_tier1_    │    │ - Span:           │    │ - teleduct_gpu_          │  │  │
│  │  │   audit.sql       │    │   intent_capture  │    │   temperature_c          │  │  │
│  │  └─────────▲─────────┘    └─────────▲─────────┘    └────────────┬─────────────┘  │  │
│  │            │                        │                           │                │  │
│  │            │                        │              ┌────────────▼─────────────┐  │  │
│  │            │                        │              │ Grafana SRE & Intent     │  │  │
│  │            │                        │              │ Cockpit (Host Port 3000) │  │  │
│  │            │                        │              │ - Thermal Gauge (°C)     │  │  │
│  │            │                        │              │ - Top-K Probs Bar Chart  │  │  │
│  │            │                        │              │ - Full Audit Table       │  │  │
│  │            │                        │              └──────────────────────────┘  │  │
│  │  └─────────┼────────────────────────┼────────────────────────────────────────────┘  │
│  └────────────┼────────────────────────┼───────────────────────────────────────────────┘
│               ▲                        ▲                        │
│               │ -R 5432:localhost:5433 │ -R 4318:localhost:4318 │ -L 18000:localhost:18000
│               │                        │                        ▼
┌───────────────┴────────────────────────┴───────────────────────────────────────────────┐
│ TEAM MEMBER 2 (Remote RunPod GPU Instance)                                             │
│                                                                                        │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │ Layer A Engine & Poller Daemons                                                  │  │
│  │                                                                                  │  │
│  │  1. `gpu_poller.py` (NVML Hardware Daemon)                                       │  │
│  │     - Samples: VRAM (GB), SM (%), Power (W), Temperature (°C) every 2.0s         │  │
│  │                                                                                  │  │
│  │  2. `sglang` Inference Server (openai/gpt-oss-20b on port 18000)                 │  │
│  │     - Extracts top-10 candidate tokens & logprobs per decision step              │  │
│  │                                                                                  │  │
│  │  3. `DualWriteEmitter`                                                           │  │
│  │     - Streams hardware and logit records to local hub over SSH tunnel            │  │
│  └──────────────────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Mathematical & Theoretical Foundations

```mermaid
flowchart TD
    subgraph DecisionStep ["1. Agent Decision & Logit Sampling"]
        PROMPT["Agent Prompt Context<br/>(Goal + Available Tools)"]
        LLM["SGLang Forward Pass<br/>Unnormalized Logits <b>z</b> ∈ ℝ<sup>|V|</sup>"]
        PROMPT --> LLM
        LLM --> SOFTMAX["Softmax Normalization<br/><b>p<sub>i</sub> = exp(z<sub>i</sub>) / ∑ exp(z<sub>j</sub>)</b>"]
        SOFTMAX --> TOPK["Top-K Candidate Extraction (K = 10)<br/>p<sub>(1)</sub> ≥ p<sub>(2)</sub> ≥ ... ≥ p<sub>(10)</sub>"]
    end

    subgraph MarginAnalysis ["2. Decision Margin & Fragility Criterion"]
        TOPK --> MARGIN["Sampling Margin Calculation<br/><b>M = p<sub>(1)</sub> − p<sub>(2)</sub></b>"]
        MARGIN --> THRESHOLD{"<b>M < 0.05 ?</b>"}
        THRESHOLD -->|Yes: Tie State| FRAGILE["🚨 <b>fragility_flag = TRUE</b><br/>High Vulnerability to Jitter"]
        THRESHOLD -->|No: Decisive| STABLE["✅ <b>fragility_flag = FALSE</b><br/>Deterministic Action"]
    end

    subgraph HardwareCoupling ["3. Silicon State Coupling (NVML)"]
        NVML["NVML Physical Sensors<br/>VRAM, SM%, Power, Temp (°C)"]
        TEMP{"<b>T<sub>GPU</sub> ≥ 82°C ?</b>"}
        NVML --> TEMP
        TEMP -->|Yes| THROTTLE["⚠️ Thermal Throttling Risk<br/>Clock Frequency Scaling"]
        TEMP -->|No| NORMAL["🟢 Optimal Thermal State"]
    end

    style DecisionStep fill:#161b22,stroke:#58a6ff,stroke-width:2px,color:#fff
    style MarginAnalysis fill:#161b22,stroke:#f59e0b,stroke-width:2px,color:#fff
    style HardwareCoupling fill:#161b22,stroke:#10b981,stroke-width:2px,color:#fff
    style FRAGILE fill:#3e1f24,stroke:#ef4444,stroke-width:2px,color:#fff
    style STABLE fill:#173322,stroke:#2ea043,stroke-width:2px,color:#fff
```

### 3.1 Mathematical Notation & Tensor Reference

| Symbol | Mathematical Domain | Physical Interpretation in TeleDuct | Threshold / Nominal Value |
|:---:|:---|:---|:---|
| $\mathbf{z}$ | $\mathbb{R}^{|V|}$ ($|V| \approx 32000$) | Unnormalized logit vector exiting the transformer head | Real values $[-\infty, +\infty]$ |
| $\mathbf{p}$ | $\Delta^{|V|-1} \subset [0, 1]^{|V|}$ | Token probability distribution over the vocabulary | $\sum_{i} p_i = 1.000$ |
| $p_{(1)}$ | $\mathbb{R} \in (0, 1)$ | Probability of the top candidate token (selected action) | e.g. $0.510$ |
| $p_{(2)}$ | $\mathbb{R} \in (0, 1)$ | Probability of the runner-up candidate token | e.g. $0.490$ |
| $M$ | $\mathbb{R} \in [0, 1]$ | Sampling Margin: $M = p_{(1)} - p_{(2)}$ | Threshold $\tau = 0.05$ |
| $T_{\text{GPU}}$ | $\mathbb{R}^+$ (°C) | Physical GPU silicon die temperature reported by NVML | Alert $> 82^\circ\text{C}$, Critical $> 87^\circ\text{C}$ |
| $P_{\text{GPU}}$ | $\mathbb{R}^+$ (Watts) | Real-time GPU total board power consumption | $0 - 450\text{ W}$ (L4 / RTX 4090) |
| $U_{\text{SM}}$ | $[0, 100]\%$ | Percentage of time streaming multiprocessors were active | Target: $60\% - 95\%$ during batch |

---

### 3.2 Formal Equation System

#### Equation 1: Top-K Sampling Margin & Fragility Decision

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  FORMULA 1: PRE-SAMPLING CANDIDATE COMPETITION & SAMPLING MARGIN                       │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│  1. Top-2 Ranked Candidate Probabilities:                                              │
│     p₍₁₎  =  maxᵢ ( pᵢ )                 [Winning candidate token]                     │
│     p₍₂₎  =  maxⱼ≠arg max p ( pⱼ )       [Runner-up candidate token]                   │
│                                                                                        │
│  2. Sampling Margin:                                                                   │
│     M     =  p₍₁₎  −  p₍₂₎               [M ∈ [0.0, 1.0]]                              │
│                                                                                        │
│  3. Fragility Decision Boundary:                                                       │
│                     ┌                                                                  │
│                     │  TRUE   (🚨 Fragile),   if  M < 0.050                            │
│     fragility_flag = ┤                                                                 │
│                     │  FALSE  (✅ Stable),    if  M ≥ 0.050                            │
│                     └                                                                  │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

$$M = p_{(1)} - p_{(2)}, \quad \text{fragility\_flag} = \begin{cases} \mathbf{true} & \text{if } M < 0.05 \\ \mathbf{false} & \text{if } M \ge 0.05 \end{cases}$$

---

#### Equation 2: Candidate Probability Dispersion & Shannon Decision Entropy

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  FORMULA 2: TOP-K CANDIDATE DECISION ENTROPY & DISPERSION                              │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│  1. Truncated Candidate Distribution Normalization (K = 10):                           │
│                 p₍ᵢ₎                                                                   │
│     qᵢ  =  ───────────────       where  ∑ᵢ₌₁ᴷ qᵢ = 1.000                               │
│             ∑ⱼ₌₁ᴷ p₍ⱼ₎                                                                 │
│                                                                                        │
│  2. Candidate Shannon Entropy:                                                         │
│     H_cand  =  − ∑ᵢ₌₁ᴷ  qᵢ · ln( qᵢ )            [ H_cand ∈ [0, ln(K)] ]               │
│                                                                                        │
│  3. Decision Dispersion Ratio:                                                         │
│                 H_cand                                                                 │
│     η_disp  =  ─────────                         [ η_disp ∈ [0.0, 1.0] ]               │
│                 ln( K )                                                                │
│                                                                                        │
│     • η_disp → 0.0 : Highly confident, winner token dominates                          │
│     • η_disp → 1.0 : Severe decision confusion across multiple candidate actions       │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

$$q_i = \frac{p_{(i)}}{\sum_{j=1}^K p_{(j)}}, \quad H_{\text{cand}} = - \sum_{i=1}^K q_i \ln q_i, \quad \eta_{\text{disp}} = \frac{H_{\text{cand}}}{\ln(K)}$$

---

#### Equation 3: Physical GPU Thermal-Power Coupling

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  FORMULA 2: THERMAL EFFICIENCY & THROTTLING RESILIENCE                                 │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│  1. Thermal Headroom:                                                                  │
│     ΔT_headroom  =  T_max_throttle  −  T_GPU         [Nominal: ΔT > 15°C]              │
│                                                                                        │
│  2. Dynamic Power Efficiency Ratio:                                                    │
│               Tokens_per_sec                                                           │
│     η_power = ──────────────                         [Tokens / Watt-second]            │
│                 P_GPU (Watts)                                                          │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

$$\Delta T_{\text{headroom}} = T_{\text{max\_throttle}} - T_{\text{GPU}}, \quad \eta_{\text{power}} = \frac{\text{Throughput}_{\text{tokens/s}}}{P_{\text{GPU}}}$$

---

## 4. Team Member 1 Execution Guide (Local Hub & Analytical Observability)

* **Owner:** Team Member 1 (Local Hub / Analytics Lead)
* **Environment:** Local Mac Workstation
* **Working Directory:** `/Users/abhishekchavan/Documents/VAST/TeleDuct`

### Task 1.1: Local Infrastructure Resumption & Port Verification

Ensure the local Docker Compose stack is running with PostgreSQL mapped to port `5433`:

#### Execution Directory:
```bash
cd /Users/abhishekchavan/Documents/VAST/TeleDuct/infra/local_hub
```

#### Terminal Commands:
```bash
# 1. Start containers in detached mode
docker compose up -d

# 2. Verify all 4 containers are healthy
docker compose ps
```

#### Port Check Command:
```bash
lsof -i :5433 -i :4318 -i :9090 -i :3000
```
*Expected Output:* Port 5433 (Postgres), 4318 (OTel), 9090 (Prometheus), and 3000 (Grafana) in `LISTEN` state.

---

### Task 1.2: Database Migration for Tier-1 Analytical Views (`views_tier1_audit.sql`)

Create relational views that expose the complete `intent_context` block, unpack candidate token logprobs into tabular distributions, and aggregate GPU thermal health.

#### File to Create:
`infra/local_hub/views_tier1_audit.sql`

```sql
-- infra/local_hub/views_tier1_audit.sql
-- Tier-1 Analytical Views: Full Intent Context, Candidate Logits, and Thermal Telemetry

-- 1. Full Chain of Intent Audit View (Satisfies EU AI Act Articles 12 & 13)
CREATE OR REPLACE VIEW v_chain_of_intent_audit AS
SELECT 
    i.session_id,
    i.step_id,
    i.step_index,
    i.node_name,
    i.parent_goal,
    i.current_objective,
    i.available_tools,
    i.selected_action,
    i.selection_rationale,
    i.guardrail_checks,
    l.sampling_margin,
    l.fragility_flag,
    h.gpu_vram_used_gb,
    h.gpu_sm_utilization,
    h.gpu_power_watts,
    h.gpu_temperature_c,
    i.created_at AS step_timestamp
FROM intent_events i
LEFT JOIN logit_telemetry l ON i.step_id = l.step_id
LEFT JOIN LATERAL (
    SELECT gpu_vram_used_gb, gpu_sm_utilization, gpu_power_watts, gpu_temperature_c
    FROM hardware_telemetry
    ORDER BY timestamp_ns DESC
    LIMIT 1
) h ON true
ORDER BY i.created_at DESC;

-- 2. Top-K Candidate Token Logit Distribution View (Unpacks JSONB candidate probabilities)
CREATE OR REPLACE VIEW v_candidate_token_distribution AS
SELECT 
    l.session_id,
    l.step_id,
    l.sampling_margin,
    l.fragility_flag,
    tok.ordinality AS candidate_rank,
    tok.value::text AS candidate_token,
    prob.value::text::double precision AS candidate_prob,
    TO_TIMESTAMP(l.timestamp_ns / 1000000000.0) AS event_time
FROM logit_telemetry l,
LATERAL jsonb_array_elements_text(l.top_k_token_ids) WITH ORDINALITY AS tok(value, ordinality)
JOIN LATERAL jsonb_array_elements_text(l.top_k_probs) WITH ORDINALITY AS prob(value, ordinality) 
    ON tok.ordinality = prob.ordinality
ORDER BY l.timestamp_ns DESC, tok.ordinality ASC;

-- 3. Hardware Thermal & Power Health Summary
CREATE OR REPLACE VIEW v_hardware_thermal_summary AS
SELECT 
    ROUND(AVG(gpu_temperature_c)::numeric, 1) AS mean_temp_c,
    MAX(gpu_temperature_c) AS peak_temp_c,
    ROUND(AVG(gpu_power_watts)::numeric, 1) AS mean_power_w,
    MAX(gpu_power_watts) AS peak_power_w,
    ROUND(AVG(gpu_vram_used_gb)::numeric, 2) AS mean_vram_gb,
    COUNT(CASE WHEN gpu_temperature_c >= 82 THEN 1 END) AS thermal_throttle_warnings
FROM hardware_telemetry
WHERE timestamp_ns >= (EXTRACT(EPOCH FROM (NOW() - INTERVAL '15 minutes')) * 1000000000);
```

#### Apply Migration to PostgreSQL:
```bash
cd /Users/abhishekchavan/Documents/VAST/TeleDuct/infra/local_hub
docker exec -i teleduct-postgres psql -U teleduct_admin -d teleduct_telemetry < views_tier1_audit.sql
```

#### Verify Views in PostgreSQL:
```bash
docker exec teleduct-postgres psql -U teleduct_admin -d teleduct_telemetry -c "\dv v_*"
```
*Expected Output:* `v_chain_of_intent_audit`, `v_candidate_token_distribution`, and `v_hardware_thermal_summary` listed as active views.

---

### Task 1.3: Provision Grafana SRE & Intent Audit Cockpit (`sre_intent_cockpit.json`)

Provision an upgraded dashboard that adds the **GPU Temperature Gauge**, the **Top-K Candidate Probability Bar Chart**, and the **Complete Chain of Intent Audit Table**.

#### File to Create:
`infra/local_hub/grafana/dashboards/sre_intent_cockpit.json`

```json
{
    "annotations": { "list": [] },
    "editable": true,
    "fiscalYearStartMonth": 0,
    "graphTooltip": 1,
    "id": null,
    "links": [],
    "liveNow": true,
    "panels": [
        {
            "datasource": { "type": "postgres", "uid": "postgresql" },
            "fieldConfig": {
                "defaults": {
                    "color": { "mode": "thresholds" },
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            { "color": "green", "value": null },
                            { "color": "yellow", "value": 1 },
                            { "color": "red", "value": 3 }
                        ]
                    },
                    "unit": "short"
                }
            },
            "gridPos": { "h": 5, "w": 6, "x": 0, "y": 0 },
            "id": 1,
            "title": "🚨 Fragile Decisions Detected (Margin < 0.05)",
            "type": "stat",
            "targets": [
                {
                    "format": "table",
                    "rawSql": "SELECT count(*) FROM logit_telemetry WHERE fragility_flag = true;",
                    "refId": "A"
                }
            ]
        },
        {
            "datasource": { "type": "prometheus", "uid": "prometheus" },
            "fieldConfig": {
                "defaults": {
                    "color": { "mode": "thresholds" },
                    "max": 24, "min": 0,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            { "color": "green", "value": null },
                            { "color": "yellow", "value": 18 },
                            { "color": "red", "value": 22 }
                        ]
                    },
                    "unit": "decbytes"
                }
            },
            "gridPos": { "h": 5, "w": 6, "x": 6, "y": 0 },
            "id": 2,
            "title": "GPU VRAM Occupancy (NVML)",
            "type": "gauge",
            "targets": [
                {
                    "expr": "teleduct_gpu_vram_used_gb_GB * 1073741824",
                    "refId": "A"
                }
            ]
        },
        {
            "datasource": { "type": "prometheus", "uid": "prometheus" },
            "fieldConfig": {
                "defaults": {
                    "color": { "mode": "thresholds" },
                    "max": 400, "min": 0,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            { "color": "green", "value": null },
                            { "color": "yellow", "value": 200 },
                            { "color": "red", "value": 320 }
                        ]
                    },
                    "unit": "watt"
                }
            },
            "gridPos": { "h": 5, "w": 6, "x": 12, "y": 0 },
            "id": 3,
            "title": "GPU Power Draw (NVML)",
            "type": "gauge",
            "targets": [
                {
                    "expr": "teleduct_gpu_power_watts",
                    "refId": "A"
                }
            ]
        },
        {
            "datasource": { "type": "postgres", "uid": "postgresql" },
            "fieldConfig": {
                "defaults": {
                    "color": { "mode": "thresholds" },
                    "max": 100, "min": 20,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            { "color": "green", "value": null },
                            { "color": "yellow", "value": 75 },
                            { "color": "red", "value": 85 }
                        ]
                    },
                    "unit": "celsius"
                }
            },
            "gridPos": { "h": 5, "w": 6, "x": 18, "y": 0 },
            "id": 4,
            "title": "GPU Die Temperature (NVML)",
            "type": "gauge",
            "targets": [
                {
                    "format": "table",
                    "rawSql": "SELECT COALESCE(gpu_temperature_c, 0) FROM hardware_telemetry ORDER BY timestamp_ns DESC LIMIT 1;",
                    "refId": "A"
                }
            ]
        },
        {
            "datasource": { "type": "postgres", "uid": "postgresql" },
            "fieldConfig": {
                "defaults": {
                    "color": { "mode": "palette-classic" },
                    "unit": "percentunit"
                }
            },
            "gridPos": { "h": 8, "w": 12, "x": 0, "y": 5 },
            "id": 5,
            "title": "Candidate Token Probability Competition (Top-K Distribution)",
            "type": "barchart",
            "targets": [
                {
                    "format": "table",
                    "rawSql": "SELECT candidate_token, candidate_prob FROM v_candidate_token_distribution WHERE step_id = (SELECT step_id FROM logit_telemetry ORDER BY timestamp_ns DESC LIMIT 1) ORDER BY candidate_rank ASC LIMIT 8;",
                    "refId": "A"
                }
            ]
        },
        {
            "datasource": { "type": "postgres", "uid": "postgresql" },
            "fieldConfig": {
                "defaults": {
                    "color": { "mode": "palette-classic" },
                    "unit": "short"
                }
            },
            "gridPos": { "h": 8, "w": 12, "x": 12, "y": 5 },
            "id": 6,
            "title": "Sampling Margin Distribution by Confidence Tier",
            "type": "barchart",
            "targets": [
                {
                    "format": "table",
                    "rawSql": "SELECT CASE WHEN sampling_margin < 0.05 THEN '🚨 Fragile (< 0.05)' WHEN sampling_margin <= 0.30 THEN '🟡 Medium (0.05 - 0.30)' ELSE '🟢 High (> 0.30)' END AS tier, COUNT(*) AS count FROM logit_telemetry WHERE sampling_margin IS NOT NULL GROUP BY 1 ORDER BY 2 DESC;",
                    "refId": "A"
                }
            ]
        },
        {
            "datasource": { "type": "postgres", "uid": "postgresql" },
            "gridPos": { "h": 10, "w": 24, "x": 0, "y": 13 },
            "id": 7,
            "title": "📜 Chain of Intent Full Audit Record (EU AI Act Article 12/13 Compliant)",
            "type": "table",
            "targets": [
                {
                    "format": "table",
                    "rawSql": "SELECT step_id, node_name, parent_goal, current_objective, selected_action, selection_rationale, guardrail_checks, ROUND(sampling_margin::numeric, 4) AS margin, fragility_flag, gpu_temperature_c AS temp_c, gpu_power_watts AS watts, step_timestamp FROM v_chain_of_intent_audit ORDER BY step_timestamp DESC LIMIT 25;",
                    "refId": "A"
                }
            ]
        }
    ],
    "refresh": "5s",
    "schemaVersion": 39,
    "style": "dark",
    "tags": ["teleduct", "tier1", "audit", "sre"],
    "time": { "from": "now-15m", "to": "now" },
    "title": "TeleDuct Tier 1 — SRE & Chain of Intent Audit Cockpit",
    "uid": "teleduct-tier1-audit"
}
```

#### Reload Dashboards in Grafana:
```bash
docker compose restart grafana
```
**Access URL:** Open `http://localhost:3000` → **Dashboards** → **TeleDuct** → **TeleDuct Tier 1 — SRE & Chain of Intent Audit Cockpit**.

---

### Task 1.4: Enhanced Compliance Agent with Full Intent Metadata (`run_audit_agent.py`)

Run an enhanced agent trajectory generator that populates all 9 intent context fields on every step.

#### File to Create:
`layer_b/demo_agent/run_audit_agent.py`

```python
#!/usr/bin/env python3
"""
layer_b/demo_agent/run_audit_agent.py
Executes financial compliance agent runs while populating 100% of the intent schema fields:
parent_goal, current_objective, available_tools, selected_action, selection_rationale, guardrail_checks.
"""
import os
import sys
import time
import uuid
import json
import psycopg2
import requests

PG_DSN = os.environ.get(
    "PG_DSN", 
    "postgresql://teleduct_admin:teleduct_secure_pass_2026@localhost:5433/teleduct_telemetry"
)
SGLANG_URL = os.environ.get("SGLANG_URL", "http://localhost:18000/v1")

def log_intent(conn, d):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO intent_events 
            (intent_id, session_id, step_id, step_index, node_name, parent_goal, 
             current_objective, available_tools, selected_action, selection_rationale, 
             guardrail_checks, goal_alignment_check)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (intent_id) DO NOTHING;
        """, (
            str(uuid.uuid4()), d["session_id"], d["step_id"], d["step_index"],
            d["node_name"], d["parent_goal"], d["current_objective"],
            json.dumps(d["available_tools"]), d["selected_action"],
            d["selection_rationale"], json.dumps(d["guardrail_checks"]),
            d.get("goal_alignment_check", "aligned")
        ))
    conn.commit()

def log_logits(conn, d):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO logit_telemetry
            (session_id, step_id, request_id, timestamp_ns, top_k_token_ids, 
             top_k_probs, sampling_margin, fragility_flag, confidence_label)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
        """, (
            d["session_id"], d["step_id"], d.get("request_id", "req_audit"),
            time.time_ns(), json.dumps(d["top_k_token_ids"]),
            json.dumps(d["top_k_probs"]), d["sampling_margin"],
            d["fragility_flag"], d["confidence_label"]
        ))
    conn.commit()

def main():
    print("==================================================")
    print("🏦 TeleDuct Tier 1: Deep Intent Audit Agent Runner")
    print(f"   - Database Target : {PG_DSN.split('@')[-1]}")
    print("==================================================")

    conn = psycopg2.connect(PG_DSN)

    test_trajectories = [
        {
            "tx": "Wire transfer $49,500 to offshore account with undisclosed ultimate beneficial owner.",
            "is_suspicious": True,
            "action": "flag_transaction",
            "rationale": "Transaction amount of $49,500 is directly below the $50,000 threshold with missing UBO declaration, indicating potential AML structuring.",
            "rule": "aml_structuring_threshold",
            "candidates": ["flag_transaction", "request_human_review", "bypass_alert"],
            "probs": [0.495, 0.480, 0.025],  # Margin 0.015 -> FRAGILE!
        },
        {
            "tx": "Standard retail cash deposit of $2,400 for local bakery weekly operating float.",
            "is_suspicious": False,
            "action": "approve_transaction",
            "rationale": "Transaction amount of $2,400 aligns with historical weekly cash float profile for verified merchant category code 5462.",
            "rule": "merchant_deposit_normal",
            "candidates": ["approve_transaction", "flag_transaction", "request_kyc_refresh"],
            "probs": [0.820, 0.110, 0.070],  # Margin 0.710 -> STABLE!
        }
    ]

    for idx, t in enumerate(test_trajectories, 1):
        session_id = str(uuid.uuid4())
        step_id = f"{session_id}__step_01_compliance"

        # 1. Register Session
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO sessions (session_id, parent_goal, agent_type, total_steps)
                VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING;
            """, (session_id, "audit_incoming_wire_transfers", "financial_compliance", 1))
        conn.commit()

        # 2. Log Full Intent Context
        margin = round(t["probs"][0] - t["probs"][1], 4)
        fragile = margin < 0.05

        log_intent(conn, {
            "session_id": session_id,
            "step_id": step_id,
            "step_index": 1,
            "node_name": "verify_transaction_limit",
            "parent_goal": "audit_incoming_wire_transfers",
            "current_objective": "verify_if_amount_exceeds_aml_threshold",
            "available_tools": ["flag_transaction", "request_human_review", "bypass_alert"],
            "selected_action": t["action"],
            "selection_rationale": t["rationale"],
            "guardrail_checks": [{"id": "nemo_financial_v1", "rule": t["rule"], "status": "passed"}]
        })

        # 3. Log Top-K Candidate Distribution
        log_logits(conn, {
            "session_id": session_id,
            "step_id": step_id,
            "top_k_token_ids": t["candidates"],
            "top_k_probs": t["probs"],
            "sampling_margin": margin,
            "fragility_flag": fragile,
            "confidence_label": "FRAGILE" if fragile else "CONFIDENT"
        })

        status = "🚨 FRAGILE DECISION" if fragile else "✅ CONFIDENT DECISION"
        print(f"Trajectory {idx}/2 | {status} | Action: {t['action']} | Margin: {margin:.4f}")

    conn.close()
    print("\n🎉 Tier-1 Trajectories Successfully Stored in Database!")

if __name__ == "__main__":
    main()
```

#### Run Audit Agent Trajectories:
```bash
source .venv-local-test/bin/activate
python3 layer_b/demo_agent/run_audit_agent.py
```

---

## 5. Team Member 2 Execution Guide (Remote GPU & Hardware Instrumentation)

* **Owner:** Team Member 2 (Remote GPU & Serving Lead)
* **Environment:** RunPod / Vast.ai GPU Container
* **Working Directory:** `/workspace/repo`

### Task 2.1: GPU Pod Resumption & SSH Triple Tunnel Verification

Member 2 reconnects to the preserved persistent volume and verifies the triple tunnel.

#### SSH Tunnel Command (Run from Member 1 or Member 2 Laptop):
```bash
ssh root@213.173.105.21 -p 13472 \
    -L 18000:localhost:18000 \
    -R 4318:localhost:4318 \
    -R 5432:localhost:5433 \
    -i ~/.ssh/id_rsa
```

#### Verification on GPU Terminal:
```bash
# Verify SGLang server is active
curl -s http://127.0.0.1:18000/health
# Expected: {"status":"healthy"} or HTTP 200
```

---

### Task 2.2: Thermal & NVML Hardware Poller Verification

Ensure `gpu_poller.py` is actively reading `gpu_temperature_c` via NVML and streaming it to PostgreSQL.

#### Code Reference in `Layer_A/gpu_poller.py`:
Verify that `gpu_poller.py` extracts die temperature:
```python
# Lines extracting temperature from NVML:
temp_c = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
```

#### Start Hardware Polling Daemon on GPU:
```bash
cd /workspace/repo
export OTEL_ENDPOINT="http://localhost:4318"
export PG_DSN="postgresql://teleduct_admin:teleduct_secure_pass_2026@localhost:5432/teleduct_telemetry"
python3 Layer_A/scripts/run_daemons.py
```

---

### Task 2.3: Top-10 Candidate Logit Interception Engine

Test SGLang's live logprob extraction with `"logprobs": True, "top_logprobs": 10` to confirm candidate token competition is captured directly from `openai/gpt-oss-20b`.

#### Test Option A: Direct cURL Command on GPU
```bash
curl -X POST http://127.0.0.1:18000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [{"role": "user", "content": "Return FLAG or APPROVE for an offshore wire transfer."}],
    "max_tokens": 1,
    "temperature": 0.0,
    "logprobs": true,
    "top_logprobs": 10
  }'
```
*Expected Output:* Response JSON contains `"top_logprobs"` with 10 candidate token strings and their respective log-probabilities.

#### Test Option B: Automated Python Validation Script
Create `Layer_A/scripts/test_sglang_logprobs.py` on the GPU:

```python
#!/usr/bin/env python3
"""
Layer_A/scripts/test_sglang_logprobs.py
Extracts top-10 candidate tokens and logprobs from SGLang on port 18000,
verifying candidate competition and sampling margin calculation.
"""
import requests
import json
import math

def test_sglang_topk():
    url = "http://127.0.0.1:18000/v1/chat/completions"
    payload = {
        "model": "gpt-oss-20b",
        "messages": [{"role": "user", "content": "Return FLAG or APPROVE for an offshore wire transfer of $50,000."}],
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 10
    }

    print("Connecting to SGLang endpoint at", url)
    res = requests.post(url, json=payload, timeout=10)
    if res.status_code != 200:
        print("❌ Error:", res.status_code, res.text)
        return

    data = res.json()
    choice = data["choices"][0]
    logprobs_content = choice.get("logprobs", {}).get("content", [])
    if not logprobs_content:
        print("⚠️ No logprobs returned in response!")
        return

    top_candidates = logprobs_content[0].get("top_logprobs", [])
    print(f"\n✅ Extracted {len(top_candidates)} Candidate Tokens from SGLang:")
    print(f"{'Rank':<5} | {'Token':<22} | {'LogProb':<10} | {'Probability':<12}")
    print("-" * 55)

    probs = []
    tokens = []
    for rank, cand in enumerate(top_candidates, 1):
        tok = cand.get("token", "").replace("\n", "\\n")
        lp = cand.get("logprob", -99.0)
        prob = math.exp(lp)
        probs.append(prob)
        tokens.append(tok)
        print(f"{rank:<5} | {tok:<22} | {lp:<10.4f} | {prob:<12.4f}")

    if len(probs) >= 2:
        margin = probs[0] - probs[1]
        fragile = margin < 0.05
        badge = "🚨 FRAGILE (M < 0.05)" if fragile else "✅ STABLE (M >= 0.05)"
        print("\n" + "=" * 55)
        print(f"Top-2 Sampling Margin : {margin:.4f} -> {badge}")
        print("=" * 55)

if __name__ == "__main__":
    test_sglang_topk()
```

#### Run Verification Script on GPU:
```bash
python3 Layer_A/scripts/test_sglang_logprobs.py
```


---

### Task 2.4: Tier-1 Trajectory Verification Harness

Member 2 executes `benchmark_moe_stress.py` to continuously feed both hardware thermal events and MoE routing metrics across the tunnel:

```bash
cd /workspace/repo
python3 Layer_A/scripts/benchmark_moe_stress.py
```
*Expected Output:* 20 runs completed, streaming 80 layer records into PostgreSQL.

---

## 6. Joint End-to-End Verification Protocol

Execute the joint test to prove that **100% of Tier-1 schema fields** are captured, correlated, and rendered on the dashboard.

### Step 1: Member 2 Triggers Stress & Polling
On the GPU instance:
```bash
python3 Layer_A/scripts/benchmark_moe_stress.py
```

### Step 2: Member 1 Triggers Audit Trajectories
On the Mac laptop:
```bash
python3 layer_b/demo_agent/run_audit_agent.py
```

### Step 3: Member 1 Verifies Full Chain of Intent in PostgreSQL
```bash
docker exec -i teleduct-postgres psql -U teleduct_admin -d teleduct_telemetry -c "
SELECT 
    step_id, 
    node_name, 
    selected_action, 
    parent_goal, 
    sampling_margin, 
    fragility_flag, 
    gpu_temperature_c, 
    selection_rationale 
FROM v_chain_of_intent_audit 
LIMIT 2;
"
```
*Expected Result:* Two rows returned with populated `selection_rationale`, `parent_goal`, `sampling_margin`, and live `gpu_temperature_c`.

### Step 4: Member 1 Inspects Candidate Logit Unpacking
```bash
docker exec -i teleduct-postgres psql -U teleduct_admin -d teleduct_telemetry -c "
SELECT 
    candidate_rank, 
    candidate_token, 
    candidate_prob 
FROM v_candidate_token_distribution 
ORDER BY event_time DESC, candidate_rank ASC 
LIMIT 3;
"
```
*Expected Result:*
```
 candidate_rank |  candidate_token   | candidate_prob 
----------------+--------------------+----------------
              1 | flag_transaction   |          0.495
              2 | request_human_rev  |          0.480
              3 | bypass_alert       |          0.025
```

### Step 5: Verify Live Grafana Dashboard
Open `http://localhost:3000/d/teleduct-tier1-audit/teleduct-tier-1-e28094-sre-and-chain-of-intent-audit-cockpit` in your browser.  
Verify that:
1. **GPU Temperature Gauge** displays real NVML die temperature (e.g. `45°C - 75°C`).
2. **Top-K Candidate Competition Bar Chart** plots candidate tokens with runner-up margins.
3. **Chain of Intent Table** displays rationale, tools, and guardrails for every step.

---

## 7. Troubleshooting & Gotchas Matrix

| Symptom / Error | Root Cause | Exact Solution |
|---|---|---|
| `LATERAL jsonb_array_elements_text error` | Column `top_k_token_ids` contains JSON object instead of JSON array. | Ensure `top_k_token_ids` and `top_k_probs` are arrays (`["tok1", "tok2"]`). The lateral query relies on array indexing. |
| GPU Temperature displays `0°C` in Grafana | `hardware_telemetry` table empty or `gpu_poller.py` not running. | Run `ps aux \| grep gpu_poller.py` on the GPU. Check `SELECT gpu_temperature_c FROM hardware_telemetry;`. |
| Audit view returns NULL for `gpu_temperature_c` | No hardware events within the lateral join window. | The view uses `LEFT JOIN LATERAL ... LIMIT 1` which handles missing rows gracefully. Ensure daemons are running. |
| Grafana shows old dashboard layout | Browser cached previous dashboard version or provisioner did not reload. | Run `docker compose restart grafana` on Mac and do a hard reload in browser (`Cmd + Shift + R`). |

---

## 8. Tier-1 Definition of Done & Sign-off Checklist

Before transitioning to **Tier 2 (NVIDIA NeMo Guardrails & Automated Failure Attribution)**, verify:

- [ ] **Member 1 (Local Hub & Analytics):**
  - [ ] `views_tier1_audit.sql` applied to PostgreSQL (`v_chain_of_intent_audit`, `v_candidate_token_distribution`, `v_hardware_thermal_summary`).
  - [ ] Grafana dashboard `TeleDuct Tier 1 — SRE & Chain of Intent Audit Cockpit` provisioned and live at `http://localhost:3000`.
  - [ ] GPU Temperature Gauge active alongside VRAM and Power Watts.
  - [ ] Candidate Logit Distribution bar chart actively plots winner vs runner-up token competition.
  - [ ] Full Chain of Intent Audit table displays `parent_goal`, `current_objective`, `selection_rationale`, and `guardrail_checks`.
- [ ] **Member 2 (Remote GPU & Hardware Instrumentation):**
  - [ ] `gpu_poller.py` active and streaming real NVML temperatures (`nvmlDeviceGetTemperature`) across tunnel.
  - [ ] SGLang logprob extraction tested with `"top_logprobs": 10`.
  - [ ] Benchmark stress harness active with zero dropped packets.
- [ ] **Joint Integration Handshake:**
  - [ ] Running `run_audit_agent.py` populates the complete 9-field intent record in PostgreSQL.
  - [ ] All records render live on Grafana with zero schema warnings.
