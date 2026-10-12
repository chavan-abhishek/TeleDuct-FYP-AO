#!/usr/bin/env python3
"""
Layer_A/scripts/benchmark_moe_stress.py
Continuous 20-run stress test evaluating MoE router gate telemetry throughput.
Produces layer-distinct routing margins and realistic expert specializations.
"""
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from emitter import DualWriteEmitter

def main():
    otel_endpoint = os.environ.get("OTEL_ENDPOINT", "http://localhost:4318")
    default_port = 5433 if sys.platform == "darwin" else 5432
    default_pg_dsn = f"postgresql://teleduct_admin:teleduct_secure_pass_2026@localhost:{default_port}/teleduct_telemetry"
    pg_dsn = os.environ.get("PG_DSN", default_pg_dsn)

    print("==================================================")
    print("⚡ Starting 20-Run MoE Stress Test Harness...")
    print(f"   - Target OTel : {otel_endpoint}")
    print(f"   - Target PG   : {pg_dsn.split('@')[-1]}")
    print("==================================================")

    emitter = DualWriteEmitter(otel_endpoint=otel_endpoint, pg_dsn=pg_dsn)

    t_start = time.time()
    for run in range(1, 21):
        session_id = str(uuid.uuid4())
        step_id = f"stress-moe-run-{run:03d}__step_0"

        # Realistic layer-distinct characteristics across the 4 transformer MoE layers:
        # Layer 0 (Shallow token representations): Tight margin, higher entropy
        # Layer 1 (Grammar/Syntax): Moderate margin
        # Layer 2 (Domain Semantics): Higher specialization
        # Layer 3 (Policy/Decision): Decisive gating with dedicated expert hot-spots
        layer_margins = {
            0: 0.04 + ((run * 7) % 15) * 0.010,   # 0.040 to 0.180 (frequent fragility)
            1: 0.12 + ((run * 11) % 20) * 0.012,  # 0.120 to 0.348 (nominal)
            2: 0.22 + ((run * 13) % 25) * 0.014,  # 0.220 to 0.556 (high confidence)
            3: 0.35 + ((run * 17) % 20) * 0.015,  # 0.350 to 0.635 (decisive)
        }

        # Hot-spot expert varies by layer to simulate real expert domain specialization:
        layer_hot_experts = {
            0: (run % 2),           # Layer 0 alternates Expert 0 & 1
            1: 2 + (run % 2),       # Layer 1 alternates Expert 2 & 3
            2: 4 + (run % 2),       # Layer 2 alternates Expert 4 & 5
            3: 6 + (run % 2),       # Layer 3 alternates Expert 6 & 7
        }

        for layer in range(4):
            margin = round(layer_margins[layer], 4)
            fragile = margin < 0.10
            hot_e = layer_hot_experts[layer]

            # Realistic load distribution: dominant expert takes 35-45% load on that layer
            load = {}
            for e in range(8):
                if e == hot_e:
                    load[str(e)] = 0.40
                elif e == (hot_e + 1) % 8:
                    load[str(e)] = 0.20
                else:
                    load[str(e)] = round(0.40 / 6, 3)

            emitter.emit_moe_event({
                "event_type": "moe_routing",
                "timestamp_ns": time.time_ns(),
                "layer_idx": layer,
                "num_experts_total": 8,
                "experts_per_token": 2,
                "avg_routing_margin": margin,
                "routing_fragile": fragile,
                "expert_load": load,
                "session_id": session_id,
                "step_id": step_id,
                "node_name": "moe_stress_worker"
            })

        print(f"   [Run {run:02d}/20] L0: {layer_margins[0]:.3f} | L1: {layer_margins[1]:.3f} | L2: {layer_margins[2]:.3f} | L3: {layer_margins[3]:.3f}")
        time.sleep(0.05)

    total_time = time.time() - t_start
    print(f"\n🎉 20-Run Stress Test Complete in {total_time:.2f}s (80 Total MoE Records Emitted).")

if __name__ == "__main__":
    main()
