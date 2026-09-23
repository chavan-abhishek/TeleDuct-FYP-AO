# Phase-Wise Implementation Plan: TeleDuct Layer A Observability

> **Document Status:** Active Execution Blueprint  
> **Target Subsystem:** Layer A (GPU Hardware, SGLang Inference Engine & Logit Interception)  
> **Team Structure:**  
> - **Team Member 1:** Local Ingestion Hub, Docker Compose, Storage, Observability Dashboards (MacBook Air M3)  
> - **Team Member 2:** Remote GPU Instance, SGLang Serving, Persistent Volume Automation, Layer A Daemons (Asus Vivobook + Rented GPU)  
> - **Team Member 3:** Research & eBPF Theory *(Layer B & eBPF work deferred to subsequent milestones)*  

---

## 1. Context & Architectural Strategy

### 1.1 The Rented GPU Challenge & Sandboxing Strategy
When renting GPU instances on **Vast.ai** or **RunPod.io**, instances are billed hourly. Pausing or stopping instances to conserve budget means ephemeral root filesystem changes can be lost upon reconnecting to a new host machine, while the attached network/host volume (`/workspace`) is preserved.

| Environment | Optimal Sandboxing Strategy | Why It Was Chosen |
|---|---|---|
| **Local Laptop (Member 1 / Member 2)** | **Docker Compose** | Bundles PostgreSQL 16, OpenTelemetry Collector Contrib, Prometheus, and Grafana in a single command (`docker compose up -d`). Completely reproducible with zero manual dependency hell. |
| **Rented GPU (Vast.ai / RunPod)** | **Hermetic Persistent Volume Sandbox (`/workspace`)** | Rented GPU instances are *already* Docker containers running on host NVIDIA drivers. Running Docker-in-Docker inside rental pods creates cgroup and CUDA driver injection conflicts. Furthermore, pulling 30 GB container images on every instance pause/resume wastes 10–15 minutes. Storing model cache (`/workspace/models`), isolated virtualenv (`/workspace/teleduct_env`), and auto-start scripts on the persistent volume allows full resume in **< 30 seconds**. |

### 1.2 Model & Inference Engine
- **Model:** `openai/gpt-oss-20b` (13.05 GB weights cached in `/workspace/models`)
- **Serving Engine:** SGLang on Port `18000` (referencing [docs/SGLANG_GPTOSS.md](file:///Users/abhishekchavan/Documents/VAST/TeleDuct/docs/SGLANG_GPTOSS.md))

---

## 2. High-Level Phase Roadmap

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: FOUNDATION (Detailed Below)                                              │
│ - Member 1: Local Ingestion Hub (Docker Compose: Postgres + OTel + Grafana)       │
│ - Member 2: Remote GPU Bootstrap + SGLang Serving + Layer A Daemons               │
│ - Joint Handshake: Verified remote GPU telemetry streaming to local visualizer    │
└────────────────────────────────────────┬──────────────────────────────────────────┘
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: LIVE LOGIT INTERCEPTION & REAL-TIME FRAGILITY STREAMING                  │
│ - Member 1: Grafana SRE Cockpit dashboards for Sampling Margin & Fragility Alerts │
│ - Member 2: Wire ChainOfIntentLogitProcessor to live SGLang token loop            │
└────────────────────────────────────────┬──────────────────────────────────────────┘
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: MOE ROUTING TELEMETRY & GATE DISTRIBUTION CAPTURE                        │
│ - Member 1: MoE expert routing heatmaps, load skew, and routing entropy panels    │
│ - Member 2: PyTorch MoE router gate hooks (patch_moe_router_gates) on live model  │
└────────────────────────────────────────┬──────────────────────────────────────────┘
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 4: MULTI-STAGE GO DAEMON OPTIMIZATION & 50-RUN BENCHMARK                    │
│ - Member 1: Automated SQL failure attribution and trajectory consistency queries  │
│ - Member 2: Multi-stage Go daemon build (if needed) & GPU batch stress execution  │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

# 3. PHASE 1 DEEP-DIVE: Complete Implementation & Work Split

Phase 1 establishes the end-to-end data pipeline: SGLang running on the rented GPU emits hardware and engine telemetry to Member 1's local Docker Compose stack via secure tunnels.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TEAM MEMBER 1 (Laptop)                            │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                         Docker Compose Hub                          │   │
│   │   ┌──────────────┐   ┌─────────────────┐   ┌────────────────────┐   │   │
│   │   │  PostgreSQL  │   │  OTel Collector │   │     Prometheus     │   │   │
│   │   │  (Port 5432) │   │   (Port 4318)   │   │    (Port 9090)     │   │   │
│   │   └──────▲───────┘   └────────▲────────┘   └─────────┬──────────┘   │   │
│   │          │                    │                      │              │   │
│   │          │                    │               ┌──────▼──────┐       │   │
│   │          │                    │               │   Grafana   │       │   │
│   │          │                    │               │ (Port 3000) │       │   │
│   │          │                    │               └─────────────┘       │   │
│   └──────────┼────────────────────┼─────────────────────────────────────┘   │
│              │                    │                                         │
│         ┌────┴────────────────────┴────┐                                    │
│         │    ngrok / Secure Tunnels    │                                    │
│         │   (TCP 5432 & HTTP 4318)     │                                    │
│         └──────────────▲───────────────┘                                    │
└────────────────────────┼────────────────────────────────────────────────────┘
                         │ (Encrypted Telemetry Stream)
┌────────────────────────┼────────────────────────────────────────────────────┐
│                        │         TEAM MEMBER 2 (Rented GPU)                 │
│  /workspace            │                                                    │
│  ┌─────────────────────┴────────────────────────────────────────────────┐   │
│  │  Layer A Daemons (teleduct_env)                                      │   │
│  │  - gpu_poller.py       (NVML Hardware: VRAM, SM%, Temp, Watts)       │   │
│  │  - sglang_scraper.py   (Engine: TTFT, Throughput, Queue Depth)       │   │
│  │  - emitter.py          (DualWrite: OTel HTTP + PostgreSQL TCP)       │   │
│  └─────────────────────▲────────────────────────────────────────────────┘   │
│                        │                                                    │
│  ┌─────────────────────┴────────────────────────────────────────────────┐   │
│  │  SGLang Inference Server (Port 18000)                                │   │
│  │  - Model: openai/gpt-oss-20b (13.05 GB cached in /workspace/models)  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3.1 Track A: Team Member 1 (Local Ingestion Hub & Visualizers)

### Target Directory Layout
```
infra/local_hub/
├── docker-compose.yml
├── otel-collector-config.yaml
├── prometheus.yml
├── init.sql
└── start_local_hub.sh
```

### Complete Code Files for Track A

#### File: `infra/local_hub/docker-compose.yml`
```yaml
services:
  postgres:
    image: postgres:16-alpine
    container_name: teleduct-postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: teleduct_admin
      POSTGRES_PASSWORD: teleduct_secure_pass_2026
      POSTGRES_DB: teleduct_telemetry
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U teleduct_admin -d teleduct_telemetry"]
      interval: 5s
      timeout: 5s
      retries: 5

  otel-collector:
    image: otel/opentelemetry-collector-contrib:0.100.0
    container_name: teleduct-otel-collector
    restart: unless-stopped
    command: ["--config=/etc/otelcol-contrib/config.yaml"]
    volumes:
      - ./otel-collector-config.yaml:/etc/otelcol-contrib/config.yaml:ro
    ports:
      - "4318:4318"   # OTLP/HTTP receiver
      - "4317:4317"   # OTLP/gRPC receiver
      - "8889:8889"   # Prometheus metrics exporter
      - "13133:13133" # Health check
    depends_on:
      postgres:
        condition: service_healthy

  prometheus:
    image: prom/prometheus:v2.51.0
    container_name: teleduct-prometheus
    restart: unless-stopped
    command:
      - --config.file=/etc/prometheus/prometheus.yml
      - --storage.tsdb.path=/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - promdata:/prometheus
    ports:
      - "9090:9090"
    depends_on:
      - otel-collector

  grafana:
    image: grafana/grafana:10.4.0
    container_name: teleduct-grafana
    restart: unless-stopped
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    ports:
      - "3000:3000"
    volumes:
      - grafanadata:/var/lib/grafana
    depends_on:
      - prometheus
      - postgres

volumes:
  pgdata:
  promdata:
  grafanadata:
```

#### File: `infra/local_hub/otel-collector-config.yaml`
```yaml
receivers:
  otlp:
    protocols:
      http:
        endpoint: 0.0.0.0:4318
      grpc:
        endpoint: 0.0.0.0:4317

processors:
  batch:
    timeout: 1s
    send_batch_size: 256

exporters:
  prometheus:
    endpoint: 0.0.0.0:8889
    namespace: teleduct
  logging:
    verbosity: normal

extensions:
  health_check:
    endpoint: 0.0.0.0:13133

service:
  extensions: [health_check]
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [logging]
    metrics:
      receivers: [otlp]
      processors: [batch]
      exporters: [prometheus, logging]
```

#### File: `infra/local_hub/prometheus.yml`
```yaml
global:
  scrape_interval: 2s
  evaluation_interval: 2s

scrape_configs:
  - job_name: 'otel-collector'
    static_configs:
      - targets: ['otel-collector:8889']
```

#### File: `infra/local_hub/init.sql`
```sql
CREATE TABLE IF NOT EXISTS sessions (
    session_id       UUID PRIMARY KEY,
    parent_goal      TEXT,
    agent_type       TEXT,
    started_at       TIMESTAMPTZ DEFAULT now(),
    ended_at         TIMESTAMPTZ,
    final_decision   TEXT,
    total_steps      INT
);

CREATE TABLE IF NOT EXISTS intent_events (
    intent_id            UUID PRIMARY KEY,
    session_id           UUID REFERENCES sessions(session_id),
    step_id              TEXT NOT NULL,
    step_index           INT,
    node_name            TEXT,
    parent_goal          TEXT,
    current_objective    TEXT,
    available_tools      JSONB,
    selected_action      TEXT,
    selection_rationale  TEXT,
    guardrail_checks     JSONB,
    goal_alignment_check TEXT,
    created_at           TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS logit_telemetry (
    id               BIGSERIAL PRIMARY KEY,
    session_id       UUID,
    step_id          TEXT,
    request_id       TEXT,
    timestamp_ns     BIGINT,
    top_k_token_ids  JSONB,
    top_k_probs      JSONB,
    sampling_margin  DOUBLE PRECISION,
    fragility_flag   BOOLEAN,
    confidence_label TEXT
);

CREATE TABLE IF NOT EXISTS moe_telemetry (
    id                 BIGSERIAL PRIMARY KEY,
    session_id         UUID,
    step_id            TEXT,
    timestamp_ns       BIGINT,
    layer_idx          INT,
    num_experts_total  INT,
    experts_per_token  INT,
    avg_routing_margin DOUBLE PRECISION,
    routing_fragile    BOOLEAN,
    expert_load        JSONB
);

CREATE TABLE IF NOT EXISTS hardware_telemetry (
    id                   BIGSERIAL PRIMARY KEY,
    timestamp_ns         BIGINT,
    gpu_vram_used_gb     DOUBLE PRECISION,
    gpu_vram_pressure    DOUBLE PRECISION,
    gpu_sm_utilization   DOUBLE PRECISION,
    gpu_memory_bandwidth DOUBLE PRECISION,
    gpu_temperature_c    DOUBLE PRECISION,
    gpu_power_watts      DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS sglang_metrics (
    id               BIGSERIAL PRIMARY KEY,
    timestamp_ns     BIGINT,
    ttft_ms          DOUBLE PRECISION,
    tokens_per_sec   DOUBLE PRECISION,
    queue_depth      INT,
    kv_cache_usage   DOUBLE PRECISION,
    running_requests INT,
    latency_p95_ms   DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS failure_reports (
    id               BIGSERIAL PRIMARY KEY,
    session_id       UUID,
    failed_at_step   INT,
    category         TEXT,
    evidence         JSONB,
    recommended_fix  TEXT,
    created_at       TIMESTAMPTZ DEFAULT now()
);

-- Indexes for query performance
CREATE INDEX IF NOT EXISTS idx_intent_session ON intent_events(session_id);
CREATE INDEX IF NOT EXISTS idx_logit_session  ON logit_telemetry(session_id, step_id);
CREATE INDEX IF NOT EXISTS idx_moe_session    ON moe_telemetry(session_id, step_id);
CREATE INDEX IF NOT EXISTS idx_logit_fragile  ON logit_telemetry(fragility_flag) WHERE fragility_flag = true;
CREATE INDEX IF NOT EXISTS idx_hw_timestamp   ON hardware_telemetry(timestamp_ns);
```

#### File: `infra/local_hub/start_local_hub.sh`
```bash
#!/usr/bin/env bash
set -e

echo "=================================================="
echo "🚀 Starting TeleDuct Local Ingestion Hub..."
echo "=================================================="
cd "$(dirname "$0")"
docker compose up -d

echo "⏳ Waiting for PostgreSQL health..."
until docker exec teleduct-postgres pg_isready -U teleduct_admin -d teleduct_telemetry > /dev/null 2>&1; do
    sleep 1
done

echo "✅ Local Ingestion Hub is LIVE!"
echo "   - PostgreSQL: localhost:5432 (DB: teleduct_telemetry, User: teleduct_admin)"
echo "   - OTel HTTP:  http://localhost:4318"
echo "   - Prometheus: http://localhost:9090"
echo "   - Grafana:    http://localhost:3000 (admin / admin)"
echo ""
echo "👉 Start your ngrok tunnels in separate terminal windows:"
echo "   Terminal 1: ngrok http 4318"
echo "   Terminal 2: ngrok tcp 5432"
```

### Terminal Commands for Team Member 1
```bash
# 1. Start the Docker Compose Stack
cd infra/local_hub
chmod +x start_local_hub.sh
./start_local_hub.sh

# 2. Expose OTel Collector via ngrok (Terminal 1)
ngrok http 4318
# -> Note public HTTPS URL: https://xxxx-xxxx.ngrok-free.app

# 3. Expose PostgreSQL via ngrok (Terminal 2)
ngrok tcp 5432
# -> Note public TCP address & port: tcp://0.tcp.ngrok.io:12345

# 4. Formulate environment variables and share with Member 2:
# export OTEL_ENDPOINT="https://xxxx-xxxx.ngrok-free.app"
# export PG_DSN="postgresql://teleduct_admin:teleduct_secure_pass_2026@0.tcp.ngrok.io:12345/teleduct_telemetry"
```

---

## 3.2 Track B: Team Member 2 (Remote GPU Automation & SGLang Serving)

### Target Directory Layout
```
Layer_A/
├── scripts/
│   ├── bootstrap_gpu.sh
│   ├── start_layer_a.sh
│   └── stop_layer_a.sh
├── run_daemons.py
└── test_remote_to_local_pipeline.py
```

### Complete Code Files for Track B

#### File: `Layer_A/scripts/bootstrap_gpu.sh`
```bash
#!/usr/bin/env bash
# One-time bootstrap on a fresh persistent volume (/workspace)
set -e

echo "=================================================="
echo "⚡ TeleDuct GPU Environment Bootstrap"
echo "=================================================="

mkdir -p /workspace/models
mkdir -p /workspace/scripts
mkdir -p /workspace/logs

if [ ! -d "/workspace/teleduct_env" ]; then
    echo "📦 Creating persistent virtualenv at /workspace/teleduct_env..."
    python3 -m venv /workspace/teleduct_env
fi

source /workspace/teleduct_env/bin/activate
pip install --upgrade pip

echo "📦 Installing Layer A Telemetry Dependencies..."
pip install \
    pynvml==11.5.0 \
    psycopg2-binary==2.9.9 \
    requests==2.31.0 \
    opentelemetry-api==1.24.0 \
    opentelemetry-sdk==1.24.0 \
    opentelemetry-exporter-otlp-proto-http==1.24.0

echo "✅ Environment bootstrapped successfully on /workspace!"
```

#### File: `Layer_A/scripts/start_layer_a.sh`
```bash
#!/usr/bin/env bash
# Daily resume script — run anytime the instance starts or resumes
set -e

echo "=================================================="
echo "🚀 TeleDuct Layer A Auto-Start Sequence"
echo "=================================================="

# 1. Verify GPU presence
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ ERROR: nvidia-smi not found. GPU not attached."
    exit 1
fi
echo "✅ GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# 2. Activate Persistent Virtualenv
source /workspace/teleduct_env/bin/activate

# 3. Check / Start SGLang on Port 18000
if lsof -i :18000 > /dev/null 2>&1; then
    echo "✅ SGLang server is already running on port 18000."
else
    echo "⏳ Starting SGLang server with openai/gpt-oss-20b..."
    nohup sglang serve \
        --model-path openai/gpt-oss-20b \
        --dtype bfloat16 \
        --trust-remote-code \
        --attention-backend triton \
        --mem-fraction-static 0.85 \
        --download-dir /workspace/models \
        --host 127.0.0.1 \
        --port 18000 \
        --tensor-parallel-size 1 \
        > /workspace/logs/sglang.log 2>&1 &

    echo "⏳ Waiting for SGLang /health (loading weights from cache)..."
    until curl -s http://127.0.0.1:18000/health > /dev/null 2>&1; do
        sleep 3
        echo -n "."
    done
    echo ""
    echo "✅ SGLang server is HEALTHY on port 18000!"
fi

echo "=================================================="
echo "🎯 SGLang Ready! Logs located at /workspace/logs/sglang.log"
echo "=================================================="
```

#### File: `Layer_A/scripts/stop_layer_a.sh`
```bash
#!/usr/bin/env bash
echo "🛑 Stopping SGLang and Layer A Daemons..."
pkill -f "sglang serve" || true
pkill -f "run_daemons.py" || true
echo "✅ All processes stopped."
```

#### File: `Layer_A/run_daemons.py`
```python
#!/usr/bin/env python3
"""
Unified runner for Layer A background daemons:
- GPUPoller (NVML GPU hardware metrics)
- SGLangMetricsScraper (Engine throughput, TTFT, queue depth)
"""
import os
import sys
import time
import signal
from emitter import DualWriteEmitter
from gpu_poller import GPUPoller
from sglang_scraper import SGLangMetricsScraper

def main():
    otel_endpoint = os.environ.get("OTEL_ENDPOINT", "http://localhost:4318")
    pg_dsn = os.environ.get("PG_DSN", None)
    sglang_url = os.environ.get("SGLANG_URL", "http://localhost:18000")
    poll_interval = float(os.environ.get("POLL_INTERVAL", "2.0"))

    print("==================================================")
    print("🛰️  Starting TeleDuct Layer A Telemetry Daemons")
    print(f"   - OTel Endpoint : {otel_endpoint}")
    print(f"   - PG DSN        : {'Configured' if pg_dsn else 'None (OTel only)'}")
    print(f"   - SGLang URL    : {sglang_url}")
    print(f"   - Poll Interval : {poll_interval}s")
    print("==================================================")

    emitter = DualWriteEmitter(otel_endpoint=otel_endpoint, pg_dsn=pg_dsn)
    
    poller = GPUPoller(emitter, interval=poll_interval)
    poller.start()

    scraper = SGLangMetricsScraper(emitter, sglang_url=sglang_url, interval=poll_interval + 1.0)
    scraper.start()

    print("✅ Daemons active and streaming. Press Ctrl+C to stop.")

    def handle_exit(sig, frame):
        print("\n🛑 Stopping daemons...")
        poller.stop()
        scraper.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
```

#### File: `Layer_A/test_remote_to_local_pipeline.py`
```python
#!/usr/bin/env python3
"""
End-to-end smoke test executed on the GPU instance:
1. Verifies SGLang health and live generation
2. Emits hardware and logit test events via DualWriteEmitter
3. Validates tunnel delivery to Member 1's local hub
"""
import os
import sys
import time
import requests
from emitter import DualWriteEmitter

def run_test():
    otel_endpoint = os.environ.get("OTEL_ENDPOINT")
    pg_dsn = os.environ.get("PG_DSN")
    sglang_url = os.environ.get("SGLANG_URL", "http://127.0.0.1:18000")

    if not otel_endpoint:
        print("❌ ERROR: OTEL_ENDPOINT environment variable must be set.")
        sys.exit(1)

    print(f"🔍 1. Checking SGLang Health at {sglang_url}/health...")
    try:
        r = requests.get(f"{sglang_url}/health", timeout=5)
        print(f"   ✅ SGLang /health responded with status {r.status_code}")
    except Exception as e:
        print(f"   ❌ SGLang unreachable: {e}")
        sys.exit(1)

    print("🔍 2. Testing Live Generation on SGLang (gpt-oss-20b)...")
    prompt = "What is the capital of France?"
    gen_payload = {
        "text": prompt,
        "sampling_params": {"max_new_tokens": 10, "temperature": 0.0}
    }
    t0 = time.time()
    resp = requests.post(f"{sglang_url}/generate", json=gen_payload, timeout=10)
    elapsed = (time.time() - t0) * 1000.0
    text = resp.json().get("text", "").strip()
    print(f"   ✅ Generated in {elapsed:.1f}ms: \"{text}\"")

    print("🔍 3. Testing DualWriteEmitter Telemetry Dispatch...")
    emitter = DualWriteEmitter(otel_endpoint=otel_endpoint, pg_dsn=pg_dsn)
    
    # Emit test hardware event
    emitter.emit_hardware_event({
        "event_type": "gpu_hardware_state",
        "timestamp_ns": time.time_ns(),
        "gpu_vram_used_gb": 13.05,
        "gpu_vram_pressure": 0.54,
        "gpu_sm_utilization": 75.0,
        "gpu_memory_bandwidth": 38.0,
        "gpu_temperature_c": 58.0,
        "gpu_power_watts": 260.0
    })

    # Emit test logit event
    emitter.emit_logit_event({
        "event_type": "logit_capture",
        "timestamp_ns": time.time_ns(),
        "request_id": "handshake-req-001",
        "session_id": "00000000-0000-0000-0000-000000000001",
        "step_id": "step_handshake",
        "node_name": "handshake_node",
        "top_k_token_ids": [101, 102, 103],
        "top_k_probs": [0.88, 0.08, 0.04],
        "sampling_margin": 0.80,
        "fragility_flag": False,
        "confidence_label": "HIGH"
    })

    print("   ✅ Telemetry packets emitted through tunnel.")
    print("==================================================")
    print("🎉 PHASE 1 REMOTE SMOKE TEST COMPLETE!")
    print("==================================================")

if __name__ == "__main__":
    run_test()
```

### Terminal Commands for Team Member 2
```bash
# 1. Connect to GPU instance terminal and enter /workspace
cd /workspace

# 2. Clone repo (or pull latest changes)
mkdir -p /workspace/repo && cd /workspace/repo
if [ ! -d "TeleDuct" ]; then
    git clone https://github.com/CHIRAL-CENTER/TeleDuct-POC.git TeleDuct
fi
cd TeleDuct

# 3. Bootstrap Virtualenv on Persistent Volume (One-time)
chmod +x Layer_A/scripts/*.sh
./Layer_A/scripts/bootstrap_gpu.sh

# 4. Export Tunnel URLs (Received from Member 1)
export OTEL_ENDPOINT="https://xxxx-xxxx.ngrok-free.app"
export PG_DSN="postgresql://teleduct_admin:teleduct_secure_pass_2026@0.tcp.ngrok.io:12345/teleduct_telemetry"

# 5. Start SGLang
./Layer_A/scripts/start_layer_a.sh

# 6. Run Pipeline Smoke Test
python3 Layer_A/test_remote_to_local_pipeline.py

# 7. Start Background Daemons
python3 Layer_A/run_daemons.py
```

---

# 4. PHASE 1 MEMBER-WISE COMPLETION CHECKLIST & VERIFICATION GUIDE

Use this checklist to verify whether Phase 1 has been successfully achieved.

## 4.1 Team Member 1 (Local Ingestion Hub) Checklist

| Item | Description | Expected Output / Verification Command | Status |
|---|---|---|---|
| **1. Docker Containers** | All 4 containers running and healthy | `docker compose ps`<br>Expected: `teleduct-postgres`, `teleduct-otel-collector`, `teleduct-prometheus`, `teleduct-grafana` all `Up (healthy)` or `Up`. | [ ] |
| **2. Database Schema** | Tables and indexes initialized | `docker exec teleduct-postgres psql -U teleduct_admin -d teleduct_telemetry -c "\dt"`<br>Expected: `sessions`, `intent_events`, `logit_telemetry`, `moe_telemetry`, `hardware_telemetry`, `sglang_metrics`, `failure_reports`. | [ ] |
| **3. OTel Collector Receiver** | OTLP/HTTP listening on 4318 | `curl -i http://localhost:13133/`<br>Expected: HTTP `200 OK` (Healthcheck extension). | [ ] |
| **4. Prometheus Target** | Prometheus scraping OTel metrics | Open `http://localhost:9090/targets`<br>Expected: `otel-collector:8889` status is `UP`. | [ ] |
| **5. Grafana UI** | Grafana dashboard accessible | Open `http://localhost:3000`<br>Expected: Login with `admin / admin` succeeds. | [ ] |
| **6. Tunnels Active** | ngrok exposing port 4318 and 5432 | Check ngrok terminal UI<br>Expected: `4318 (HTTP)` and `5432 (TCP)` public URLs active. | [ ] |

---

## 4.2 Team Member 2 (Remote GPU & Serving) Checklist

| Item | Description | Expected Output / Verification Command | Status |
|---|---|---|---|
| **1. GPU Hardware State** | GPU recognized with 24GB VRAM | `nvidia-smi`<br>Expected: RTX 3090/4090, ~13GB VRAM in use when model loaded. | [ ] |
| **2. Persistent Volume Cache** | Model weights cached on `/workspace` | `ls -la /workspace/models/models--openai--gpt-oss-20b/`<br>Expected: Snapshots present; SGLang logs report `Found local HF snapshot ...; skipping download`. | [ ] |
| **3. SGLang Server Health** | Engine listening on port 18000 | `curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:18000/health`<br>Expected: `200`. | [ ] |
| **4. Live Token Generation** | SGLang produces tokens under 1.0s | `curl -s http://127.0.0.1:18000/generate -H "Content-Type: application/json" -d '{"text":"What is 2+2?","sampling_params":{"max_new_tokens":10,"temperature":0}}'`<br>Expected: JSON with `"text"` field. | [ ] |
| **5. Background Daemons** | Hardware and metric pollers running | `ps aux | grep run_daemons.py`<br>Expected: Python process active, streaming telemetry every 2–3 seconds. | [ ] |
| **6. Fast Resume Proof** | Instance pause & resume test | Stop instance, reconnect, run `./Layer_A/scripts/start_layer_a.sh`<br>Expected: Fully online in **< 30 seconds**. | [ ] |

---

## 4.3 Joint End-to-End Handshake Verification

After Member 1 and Member 2 complete their individual checklists, execute the joint handshake:

1. **Member 2 runs the pipeline test on the GPU:**
   ```bash
   python3 Layer_A/test_remote_to_local_pipeline.py
   ```
2. **Member 1 verifies ingestion in PostgreSQL:**
   ```bash
   docker exec -it teleduct-postgres psql -U teleduct_admin -d teleduct_telemetry -c "SELECT id, gpu_vram_used_gb, gpu_sm_utilization, gpu_power_watts FROM hardware_telemetry ORDER BY id DESC LIMIT 5;"
   ```
   **Expected Result:** Newly inserted row with `gpu_vram_used_gb = 13.05` and `gpu_sm_utilization = 75.0`.
3. **Member 1 verifies logit telemetry insertion:**
   ```bash
   docker exec -it teleduct-postgres psql -U teleduct_admin -d teleduct_telemetry -c "SELECT request_id, sampling_margin, fragility_flag, confidence_label FROM logit_telemetry ORDER BY id DESC LIMIT 5;"
   ```
   **Expected Result:** Row with `request_id = 'handshake-req-001'`, `sampling_margin = 0.8`, `fragility_flag = f`.
4. **Member 1 checks Grafana:**
   - In Grafana Explore (`http://localhost:3000/explore`), query Prometheus for `teleduct_gpu_vram_used_gb`.
   - **Expected Result:** Live time-series metric graph appears.

---

# 5. SUBSEQUENT PHASES OVERVIEW

### Phase 2: Live Logit Interception & Real-time Fragility Streaming
- **Objective:** Attach `ChainOfIntentLogitProcessor` to the SGLang generation loop so every token decoding step calculates $P(\text{top}_1) - P(\text{top}_2)$ and marks steps with margin $< 0.05$ as fragile.
- **Team Member 1 Responsibilities:**
  - Build Grafana SRE Cockpit dashboards (Sampling Margin distribution histogram, live Fragility Alert trigger).
  - Write SQL views linking step indices to token confidence levels.
- **Team Member 2 Responsibilities:**
  - Wire the custom logit processor into SGLang launch flags (`--enable-custom-logit-processor`).
  - Generate multi-turn agent prompt batches to validate real-time logit interception without latency degradation.

### Phase 3: MoE Routing Telemetry & Gate Distribution Capture
- **Objective:** Hook sparse MoE router gates via `patch_moe_router_gates` to capture expert assignment weights, routing entropy, and expert load imbalances.
- **Team Member 1 Responsibilities:**
  - Build expert load heatmaps in Grafana.
  - Query PostgreSQL `moe_telemetry` to detect routing bottlenecks.
- **Team Member 2 Responsibilities:**
  - Attach PyTorch forward hooks to live MoE layers during inference.
  - Stream `avg_routing_margin` and `expert_load` distributions.

### Phase 4: Multi-Stage Go Daemon Optimization & Stress Benchmarking
- **Objective:** Validate that telemetry collection causes zero jitter. Execute 50-run consecutive compliance evaluations.
- **Team Member 1 Responsibilities:**
  - Run failure attribution analysis scripts.
  - Calculate Trajectory Consistency Scores.
- **Team Member 2 Responsibilities:**
  - If Python daemons consume > 1% CPU, compile and deploy the single-binary Go collector (`teleduct-gpu-agent`).
  - Execute the 50-run benchmark harness on the GPU and confirm zero packet drops across network fluctuations.
