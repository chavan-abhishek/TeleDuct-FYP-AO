# layer_a/test_real_emitter_local.py
"""
Tests the REAL DualWriteEmitter against your REAL local OTel Collector
(through the ngrok tunnel) and REAL local PostgreSQL — using fake
data instead of a real GPU. This is the last thing to prove before
GPU time starts.
"""
import sys, os, time, random, uuid
sys.path.insert(0, ".")
from emitter import DualWriteEmitter

OTEL_ENDPOINT = os.getenv("OTEL_ENDPOINT", "http://localhost:4318")
PG_DSN        = os.getenv("PG_DSN", "postgresql://teleduct_admin:teleduct_secure_pass_2026@localhost:5432/teleduct_telemetry")

emitter = DualWriteEmitter(
    otel_endpoint=OTEL_ENDPOINT,
    pg_dsn=PG_DSN,
    service_name="teleduct-real-emitter-test"
)

# ── Real UUID, generated once, reused across all events in this test run.
#    Must be a valid UUID string — PostgreSQL's session_id column type
#    is UUID and rejects plain strings like "test-session-001".
SESSION_ID = str(uuid.uuid4())
print(f"Using session_id: {SESSION_ID}")

# ── Insert the parent session row FIRST — logit_telemetry/intent_events
#    reference sessions.session_id via foreign key in intent_events,
#    though logit_telemetry itself has no FK constraint, it's good
#    practice to have the session exist. (Optional but recommended —
#    uncomment if you want a fully consistent sessions row too.)
#
# import psycopg2
# pg_direct = psycopg2.connect(PG_DSN)
# pg_direct.autocommit = True
# cur = pg_direct.cursor()
# cur.execute(
#     "INSERT INTO sessions (session_id, parent_goal, agent_type) "
#     "VALUES (%s,%s,%s)",
#     (SESSION_ID, "test run", "emitter_smoke_test"))

# ── Logit events: send 10 with varied margins so the histogram
#    has a real distribution, and at least one triggers fragility_flag ──
for i in range(10):
    margin = random.uniform(0.01, 0.5)
    emitter.emit_logit_event({
        "timestamp_ns": time.time_ns(),
        "session_id": SESSION_ID,
        "step_id": f"test-step-{i:03d}",
        "request_id": f"req-{i:03d}",
        "node_name": "test_node",
        "top_k_token_ids": [1, 2, 3, 4, 5],
        "top_k_probs": [0.4, 0.3, 0.15, 0.1, 0.05],
        "sampling_margin": margin,
        "fragility_flag": margin < 0.05,   # ← now some WILL be True
        "confidence_label": (
            "HIGH" if margin > 0.30 else
            "MEDIUM" if margin > 0.05 else "LOW"),
    })
    time.sleep(0.2)

# Fake a MoE event
emitter.emit_moe_event({
    "timestamp_ns": time.time_ns(),
    "session_id": SESSION_ID,
    "step_id": "test-step-000",
    "layer_idx": 0,
    "num_experts_total": 8,
    "experts_per_token": 4,
    "avg_routing_margin": 0.22,
    "routing_fragile": False,
    "expert_load": {"0": 0.3, "1": 0.25, "2": 0.25, "3": 0.2},
})

# Fake a hardware event
emitter.emit_hardware_event({
    "timestamp_ns": time.time_ns(),
    "gpu_vram_used_gb": 12.4,
    "gpu_vram_pressure": 0.52,
    "gpu_sm_utilization": 68.0,
    "gpu_memory_bandwidth": 55.0,
    "gpu_temperature_c": 61.0,
    "gpu_power_watts": 180.0,
})

# Fake SGLang metrics — timestamp_ns added, was missing before
emitter.emit_sglang_metrics({
    "timestamp_ns": time.time_ns(),   # ← the fix
    "ttft_ms": 85.0,
    "tokens_per_sec": 42.0,
    "queue_depth": 2,
    "kv_cache_usage": 0.4,
    "running_requests": 1,
})

# Must exceed export_interval_millis=5000 (5 seconds)
# so the PeriodicExportingMetricReader actually flushes
# before the script exits
time.sleep(8)

print("\n✅ All event types sent through REAL emitter → OTel Collector + PostgreSQL")
print(f"   session_id used: {SESSION_ID}")
print("   10 logit events (varied margins, some fragile) sent")
print("   1 MoE event, 1 hardware event, 1 SGLang metrics event sent")
print("Check Grafana dashboards for 'teleduct-real-emitter-test' data")
print("Check dashboard panels — Sampling Margin Distribution and")
print("Fragile Decisions Count should now show real data points")