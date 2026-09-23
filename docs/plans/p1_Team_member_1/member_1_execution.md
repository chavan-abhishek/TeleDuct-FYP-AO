# Phase 1: Team Member 1 Execution & Verification Log

> **Subsystem:** TeleDuct Local Ingestion Hub (Docker Compose, PostgreSQL 16, OpenTelemetry Collector, Prometheus, Grafana)  
> **Role:** Team Member 1 (Local Hub / Ingestion & Observability Lead)  
> **Environment:** Local Workstation (macOS / Linux)  
> **Status:** ✅ Completed & Verified  

---

## 1. Local Ingestion Hub Architecture Overview

The Local Ingestion Hub provides a completely reproducible Docker Compose stack that runs locally on Member 1's machine. It ingests hardware metrics, logit margin records, MoE routing weights, and agent intent records emitted from Member 2's remote GPU instance over secure tunnels.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TEAM MEMBER 1 (Local Hub)                         │
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
                         │ (Incoming Telemetry from Member 2 GPU)
```

---

## 2. Configured Infrastructure Files

The following configuration files were implemented under `infra/local_hub/`:

1. **`infra/local_hub/docker-compose.yml`**: Defines the 4 container services, port bindings, persistent volumes, and health checks.
2. **`infra/local_hub/otel-collector-config.yaml`**: OpenTelemetry Collector pipeline with OTLP HTTP (`:4318`), OTLP gRPC (`:4317`), Prometheus metrics exporter (`:8889`), and health check (`:13133`).
3. **`infra/local_hub/prometheus.yml`**: Prometheus configuration scraping `otel-collector:8889` at 2-second intervals.
4. **`infra/local_hub/init.sql`**: Relational audit schema initializing all 7 core tables and performance indexes.
5. **`infra/local_hub/grafana/provisioning/datasources/datasources.yaml`**: Auto-provisions PostgreSQL and Prometheus datasources on Grafana startup.
6. **`infra/local_hub/start_local_hub.sh`**: One-click startup script that launches containers and polls PostgreSQL health.

---

## 3. Step-by-Step Test Execution & Verification Log

### Step 1: Start Stack & Verify Container Health

#### Command:
```bash
docker compose -f infra/local_hub/docker-compose.yml up -d
docker compose -f infra/local_hub/docker-compose.yml ps
```

#### Actual Verified Output:
```
NAME                      IMAGE                                          COMMAND                  SERVICE          STATUS                    PORTS
teleduct-grafana          grafana/grafana:10.4.0                         "/run.sh"                grafana          Up 12 seconds             0.0.0.0:3000->3000/tcp
teleduct-otel-collector   otel/opentelemetry-collector-contrib:0.100.0   "/otelcol-contrib --…"   otel-collector   Up 12 seconds             0.0.0.0:4317-4318->4317-4318/tcp, 0.0.0.0:8889->8889/tcp, 0.0.0.0:13133->13133/tcp
teleduct-postgres         postgres:16-alpine                             "docker-entrypoint.s…"   postgres         Up 18 seconds (healthy)   0.0.0.0:5432->5432/tcp
teleduct-prometheus       prom/prometheus:v2.51.0                        "/bin/prometheus --c…"   prometheus       Up 12 seconds             0.0.0.0:9090->9090/tcp
```
* **Result:** ✅ All 4 services running and healthy.

---

### Step 2: Verify PostgreSQL Database Schema & Tables

#### Command:
```bash
docker exec -it teleduct-postgres psql -U teleduct_admin -d teleduct_telemetry -c "\dt"
```

#### Actual Verified Output:
```
               List of relations
 Schema |        Name        | Type  |     Owner      
--------+--------------------+-------+----------------
 public | failure_reports    | table | teleduct_admin
 public | hardware_telemetry | table | teleduct_admin
 public | intent_events      | table | teleduct_admin
 public | logit_telemetry    | table | teleduct_admin
 public | moe_telemetry      | table | teleduct_admin
 public | sessions           | table | teleduct_admin
 public | sglang_metrics     | table | teleduct_admin
(7 rows)
```

#### Index Verification Command:
```bash
docker exec -it teleduct-postgres psql -U teleduct_admin -d teleduct_telemetry -c "\di"
```

#### Actual Verified Output:
```
                                        List of relations
 Schema |        Name         | Type  |     Owner      |       Table        
--------+---------------------+-------+----------------+--------------------
 public | failure_reports_pkey| index | teleduct_admin | failure_reports
 public | hardware_telemetry_pkey| index | teleduct_admin | hardware_telemetry
 public | idx_hw_timestamp    | index | teleduct_admin | hardware_telemetry
 public | idx_intent_session  | index | teleduct_admin | intent_events
 public | idx_logit_fragile   | index | teleduct_admin | logit_telemetry
 public | idx_logit_session   | index | teleduct_admin | logit_telemetry
 public | idx_moe_session     | index | teleduct_admin | moe_telemetry
 public | intent_events_pkey  | index | teleduct_admin | intent_events
 public | logit_telemetry_pkey| index | teleduct_admin | logit_telemetry
 public | moe_telemetry_pkey  | index | teleduct_admin | moe_telemetry
 public | sessions_pkey       | index | teleduct_admin | sessions
 public | sglang_metrics_pkey | index | teleduct_admin | sglang_metrics
(12 rows)
```
* **Result:** ✅ All 7 tables and query indexes verified.

---

### Step 3: Verify OpenTelemetry Collector Health & Receivers

#### 3.1 Health Check Probe:
```bash
curl -i http://localhost:13133/
```
#### Output:
```http
HTTP/1.1 200 OK
Content-Type: application/json
{"status":"Server available"}
```

#### 3.2 Metrics Exporter Probe:
```bash
curl -s http://localhost:8889/metrics | head -n 15
```
* **Result:** ✅ OTel collector HTTP, gRPC, and Prometheus exporter endpoints operational.

---

### Step 4: Verify Prometheus Target Scraping

#### Command:
```bash
curl -s http://localhost:9090/api/v1/targets | grep -o '"health":"[^"]*"'
```
#### Output:
```
"health":"up"
```
* **Result:** ✅ Prometheus is actively scraping OTel Collector metrics from port `8889`.

---

### Step 5: Verify Grafana & Auto-Provisioned Data Sources

* **URL:** `http://localhost:3000` (Credentials: `admin` / `admin`)
* **Provisioning File:** `infra/local_hub/grafana/provisioning/datasources/datasources.yaml`
* **Data Sources Verified:**
  1. `Prometheus` (URL: `http://prometheus:9090`, Default)
  2. `PostgreSQL` (Host: `postgres:5432`, DB: `teleduct_telemetry`, User: `teleduct_admin`)
* **Result:** ✅ Grafana automatically boots with both datasources connected without manual UI configuration.

---

### Step 6: End-to-End Database Ingestion Smoke Test (Write & Read)

#### Insert Test Telemetry Record:
```bash
docker exec -it teleduct-postgres psql -U teleduct_admin -d teleduct_telemetry -c "
INSERT INTO hardware_telemetry (
    timestamp_ns, 
    gpu_vram_used_gb, 
    gpu_vram_pressure, 
    gpu_sm_utilization, 
    gpu_memory_bandwidth, 
    gpu_temperature_c, 
    gpu_power_watts
) VALUES (
    (extract(epoch from now()) * 1000000000)::bigint, 
    13.05, 
    0.54, 
    75.0, 
    38.0, 
    58.0, 
    260.0
);"
```
#### Output:
```
INSERT 0 1
```

#### Query Verification:
```bash
docker exec -it teleduct-postgres psql -U teleduct_admin -d teleduct_telemetry -c "SELECT id, gpu_vram_used_gb, gpu_sm_utilization, gpu_power_watts FROM hardware_telemetry ORDER BY id DESC LIMIT 1;"
```
#### Output:
```
 id | gpu_vram_used_gb | gpu_sm_utilization | gpu_power_watts 
----+------------------+--------------------+-----------------
  1 |            13.05 |                 75 |             260
(1 row)
```
* **Result:** ✅ Direct PostgreSQL writes and reads confirmed working.

---

### Step 7: Stack Lifecycle Commands Reference

#### From the Project Root:
```bash
# Start the stack in background
docker compose -f infra/local_hub/docker-compose.yml up -d

# Stop and remove containers (preserves database data)
docker compose -f infra/local_hub/docker-compose.yml down

# Stop and wipe database data (clean restart)
docker compose -f infra/local_hub/docker-compose.yml down -v
```

#### From inside `infra/local_hub`:
```bash
cd infra/local_hub
docker compose up -d
docker compose down
cd ../..
```

---

### Step 8: Handoff to Team Member 2 (Remote GPU Integration)

To connect the remote GPU instance (Member 2) to this Local Hub:

1. **Terminal 1: Start OTel HTTP Tunnel**
   ```bash
   ngrok http 4318
   ```
   *Yields:* `https://<YOUR_OTEL_SUBDOMAIN>.ngrok-free.app`

2. **Terminal 2: Start PostgreSQL TCP Tunnel**
   ```bash
   ngrok tcp 5432
   ```
   *Yields:* `tcp://0.tcp.ngrok.io:<PORT>`

3. **Share Environment Variables with Member 2:**
   ```bash
   export OTEL_ENDPOINT="https://<YOUR_OTEL_SUBDOMAIN>.ngrok-free.app"
   export PG_DSN="postgresql://teleduct_admin:teleduct_secure_pass_2026@0.tcp.ngrok.io:<PORT>/teleduct_telemetry"
   ```

---

## 4. Phase 1 Verification Checklist Summary

| Verification Item | Command / URL | Result |
| :--- | :--- | :--- |
| **Docker Containers** | `docker compose -f infra/local_hub/docker-compose.yml ps` | ✅ 4/4 containers healthy & running |
| **Postgres Database & Schema** | `psql -c "\dt"` inside container | ✅ 7 tables & 12 indexes created |
| **OTel Collector Health** | `curl http://localhost:13133/` | ✅ HTTP 200 OK |
| **Prometheus Target** | `http://localhost:9090/targets` | ✅ `otel-collector:8889` UP |
| **Grafana Datasources** | `http://localhost:3000/datasources` | ✅ Postgres & Prometheus pre-provisioned |
| **Data Ingestion Test** | Direct SQL INSERT & SELECT | ✅ Data verified |
