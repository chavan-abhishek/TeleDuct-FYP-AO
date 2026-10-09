# TeleDuct Project Master Status & Tracking Board

> **Last Updated:** October 10, 2026  
> **Overall Status:** 🟢 Phase 1 Core Milestones Achieved (Member 1 Local Hub & Member 2 SGLang Serving Verified)  
> **Current Focus:** Phase 1 Joint End-to-End Tunnel Handshake ➔ Phase 2 Live Logit Interception  

---

## 1. Executive Summary

TeleDuct is a silicon-to-agent observability pipeline designed for safety-critical agent systems under Article 12/13 of the EU AI Act. It unifies pre-sampling logit margin fragility, MoE gate routing entropy, and GPU physical state into an OpenTelemetry + PostgreSQL pipeline.

As of this update, **both primary foundational tracks have been successfully executed and independently verified:**
1. **Team Member 1 (Local Ingestion Hub):** Complete Docker Compose stack (PostgreSQL 16, OTel Collector Contrib, Prometheus, Grafana) operational with schema tables and datasource provisioning.
2. **Team Member 2 (Remote GPU Serving):** SGLang inference engine successfully serving `openai/gpt-oss-20b` on an NVIDIA L4 GPU (24GB VRAM) under CUDA 13.0, verified with healthy `/health` 200, Triton kernels compiled, and warmup token generation.

---

## 2. Subsystem & Track Status Matrix

| Subsystem / Track | Lead | Environment / Target | Status | Verification Reference |
|---|---|---|---|---|
| **Local Ingestion Hub** | Team Member 1 | Docker Compose (macOS/Linux) | 🟢 **VERIFIED** | [member_1_execution.md](file:///Users/abhishekchavan/Documents/VAST/TeleDuct/docs/plans/p1_Team_member_1/member_1_execution.md) |
| **Relational Storage Schema** | Team Member 1 | PostgreSQL 16 (7 core tables) | 🟢 **VERIFIED** | `init.sql` (sessions, logit_telemetry, etc.) |
| **OTel Metrics & Traces Pipeline** | Team Member 1 | OTel Collector 0.100.0 | 🟢 **VERIFIED** | OTLP/HTTP `:4318`, Prometheus exporter `:8889` |
| **SRE Visualization** | Team Member 1 | Grafana 10.4.0 (`:3000`) | 🟢 **VERIFIED** | Datasources auto-provisioned |
| **GPU Inference Serving** | Team Member 2 | RunPod / NVIDIA L4 (24GB) | 🟢 **VERIFIED** | [member_2_execution.md](file:///Users/abhishekchavan/Documents/VAST/TeleDuct/docs/plans/p2_member_GPU/member_2_execution.md) |
| **Model Weight Caching** | Team Member 2 | Persistent Volume `/workspace/models` | 🟢 **VERIFIED** | `openai/gpt-oss-20b` (13.05 GB cached) |
| **SGLang Engine Health** | Team Member 2 | Port `18000` | 🟢 **VERIFIED** | `/health` 200 OK, warmup generation OK |
| **GPU Host Automation** | Team Member 2 | `Layer_A/scripts/*.sh` | 🟢 **VERIFIED** | `start_layer_a.sh`, `stop_layer_a.sh`, `run_daemons.py` |
| **Phase 1 Joint Handshake** | Joint (M1 + M2) | ngrok Tunnels (HTTP/TCP) | 🟡 **READY TO RUN** | `test_remote_to_local_pipeline.py` |
| **Phase 2: Logit Interception** | Joint (M1 + M2) | `ChainOfIntentLogitProcessor` | ⚪ **UPCOMING** | Next development milestone |
| **Phase 3: MoE Routing** | Joint (M1 + M2) | `patch_moe_router_gates` | ⚪ **UPCOMING** | Milestone 3 |
| **Phase 4: Agent Benchmarks** | Joint (M1 + M2) | LangGraph Compliance Agent | ⚪ **UPCOMING** | Milestone 4 |

---

## 3. Environment & Port Configuration Reference

| Component | Default Host/Port | Purpose | Access Mode |
|---|---|---|---|
| **SGLang Inference** | `127.0.0.1:18000` | LLM token generation & completions | GPU Internal (`/generate`, `/health`) |
| **PostgreSQL** | `localhost:5432` | Relational audit trail & JSONB storage | Member 1 Internal + ngrok TCP |
| **OTel Collector HTTP** | `localhost:4318` | Ingests OTLP traces & metric packets | Member 1 Internal + ngrok HTTP |
| **Prometheus** | `localhost:9090` | Timeseries metrics scraping | Member 1 Internal (`:9090`) |
| **Grafana UI** | `localhost:3000` | Real-time observability dashboard | Member 1 Internal (`admin/admin`) |
| **OTel Health Check** | `localhost:13133` | Collector pipeline health | Member 1 Internal |

---

## 4. Work Breakdown by Team Member

### 4.1 Team Member 1 (Local Hub Lead)
- **Completed:**
  - Designed and tested `docker-compose.yml`, `init.sql`, `otel-collector-config.yaml`, `prometheus.yml`.
  - Executed end-to-end database schema checks (7 tables, indexes).
  - Validated local OTel ingestion and Prometheus metrics pipeline.
- **Next Actions:**
  - Launch local stack (`./infra/local_hub/start_local_hub.sh`).
  - Open ngrok tunnels (`ngrok http 4318` & `ngrok tcp 5432`).
  - Provide public URLs to Member 2 for the live handshake test.
  - Build initial Grafana panels for GPU VRAM and Logit Sampling Margin histograms.

### 4.2 Team Member 2 (Remote GPU Lead)
- **Completed:**
  - Provisioned RunPod instance with NVIDIA L4 (24GB VRAM) and Network Volume (`/workspace`).
  - Solved CUDA 13 / Ada Lovelace (`sm_89`) dependency challenges (`sglang-kernel 0.4.9`, uninstalled conflicting `torch_c_dlpack_ext`).
  - Cached `openai/gpt-oss-20b` weights on persistent volume.
  - Successfully brought SGLang online with verified `/health` 200, Triton compiler execution, and warmup prefill batch.
  - Built automation scripts (`start_layer_a.sh`, `stop_layer_a.sh`, `bootstrap_gpu.sh`).
- **Next Actions:**
  - Resume instance according to [runpod_resume_runbook.md](file:///Users/abhishekchavan/Documents/VAST/TeleDuct/docs/plans/p2_member_GPU/runpod_resume_runbook.md).
  - Execute `Layer_A/scripts/test_remote_to_local_pipeline.py` using Member 1's tunnel URLs.
  - Start background telemetry daemons via `Layer_A/scripts/run_daemons.py`.
  - Integrate `ChainOfIntentLogitProcessor` with SGLang for real-time logit interception.

---

## 5. Next Session Kick-Off Action Plan

When resuming work in the next session:
1. **Member 1 launches local hub & ngrok:**
   ```bash
   ./infra/local_hub/start_local_hub.sh
   ngrok http 4318   # Share HTTPS URL with Member 2
   ngrok tcp 5432    # Share TCP URL with Member 2
   ```
2. **Member 2 spins up pod & starts SGLang:**
   Follow [runpod_resume_runbook.md](file:///Users/abhishekchavan/Documents/VAST/TeleDuct/docs/plans/p2_member_GPU/runpod_resume_runbook.md):
   ```bash
   cd /workspace/repo && git pull origin main
   source /workspace/teleduct_env/bin/activate
   ./Layer_A/scripts/start_layer_a.sh
   ```
3. **Execute Joint Handshake:**
   ```bash
   export OTEL_ENDPOINT="<MEMBER_1_OTEL_NGROK>"
   export PG_DSN="<MEMBER_1_PG_NGROK>"
   python3 ./Layer_A/scripts/test_remote_to_local_pipeline.py
   ```
4. **Begin Phase 2:** Wire `ChainOfIntentLogitProcessor` to live SGLang token loop.
