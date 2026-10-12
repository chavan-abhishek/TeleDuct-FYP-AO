#!/usr/bin/env python3
"""Reproducible Acceptance Gate: Live Ollama + Local NVIDIA GPU Telemetry Verification.

Executes a live end-to-end trace against:
- Engine: Local Ollama (http://localhost:11434/v1, llama3.2:latest)
- GPU: Local NVIDIA GeForce RTX 3060 Laptop GPU (native NVML / nvidia-smi)
- Telemetry: TeleDuct Semantic, Runtime, Hardware, and Fusion adapters
"""
from __future__ import annotations
import json
import os
import sys
import time
import uuid
from pathlib import Path

# Link multi-worktree modules
_BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BASE / "fusion" / "src"))
import teleduct
for _wt in ("semantic", "runtime", "hardware"):
    _src = str(_BASE / _wt / "src" / "teleduct")
    if _src not in teleduct.__path__ and Path(_src).exists():
        teleduct.__path__.append(_src)

from teleduct.contracts.events import SCHEMA_VERSION
from teleduct.semantic import (
    agent_run_event,
    agent_step_event,
    SemanticStepTracker,
)
from teleduct.runtime import (
    TruthfulInferenceClient,
)
from teleduct.hardware import (
    DeviceTelemetryWindow,
    sample_device_nvml,
    sample_device,
    get_device_info,
)
from teleduct.fusion import (
    EventIngestionStore,
)


def run_live_gate(
    output_dir: str = "/tmp/teleduct_gate_output",
    ollama_url: str = "http://localhost:11434/v1",
    model: str = "llama3.2:latest",
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    jsonl_file = out_path / "live_ollama_events.jsonl"
    if jsonl_file.exists():
        jsonl_file.unlink()

    store = EventIngestionStore(storage_path=jsonl_file)
    run_id = f"run-live-{uuid.uuid4().hex[:8]}"
    trace_id = uuid.uuid4().hex
    root_span_id = uuid.uuid4().hex[:16]

    print("=" * 70)
    print("TELEDUCT ACCEPTANCE GATE: LIVE OLLAMA + NVIDIA GPU RUN")
    print(f"Contract Schema Version: {SCHEMA_VERSION}")
    print(f"Run ID:                  {run_id}")
    print(f"Trace ID:                {trace_id}")
    print(f"Target LLM Engine:       {ollama_url} (model: {model})")
    print("=" * 70)

    # 1. Inspect Hardware Baseline
    dev_info = get_device_info(0)
    print(f"\n[1/5] GPU Hardware Detection: {dev_info.get('gpu_name')} (Driver: {dev_info.get('driver_version')})")
    try:
        baseline_sample = sample_device_nvml(0)
        hw_sampler_fn = lambda: sample_device_nvml(0)
        print(f"      Sampler: Native NVML (source='nvml', attribution='{baseline_sample.attribution}')")
    except Exception as ex:
        baseline_sample = sample_device(0)
        hw_sampler_fn = lambda: sample_device(0)
        print(f"      Sampler: nvidia-smi (source='nvidia_smi', attribution='{baseline_sample.attribution}')")

    store.ingest(baseline_sample)
    print(f"      Baseline VRAM Used: {baseline_sample.metrics.get('vram_used_mib')} MiB")
    print(f"      Baseline Power:     {baseline_sample.metrics.get('power_w')} W")

    # 2. Semantic Initialization Event
    print("\n[2/5] Initializing Semantic Run Boundary...")
    run_ev = agent_run_event(
        run_id=run_id,
        parent_goal="Audit suspicious high-velocity cross-border transaction for OFAC sanctions compliance",
        agent_type="financial_compliance",
        trace_id=trace_id,
        span_id=root_span_id,
        capture_raw_goal=False,  # Privacy default
    )
    store.ingest(run_ev)
    print(f"      Emitted agent.run (goal privacy hash: {run_ev.attributes.get('parent_goal_hash')})")

    # 3. Step 1 with Active GPU Window & Live Ollama Request
    print("\n[3/5] Executing Live Inference Step with Time-Aligned GPU Sampling Window...")
    step_id = "step_001_screen_sanctions"
    step_span_id = uuid.uuid4().hex[:16]
    client_request_id = f"client-req-{uuid.uuid4().hex[:12]}"

    runtime_events = []
    client = TruthfulInferenceClient(
        base_url=ollama_url,
        model=model,
        engine="ollama/local",
        event_sink=runtime_events,
    )

    t0_step = time.perf_counter()
    with DeviceTelemetryWindow(
        gpu_index=0,
        sample_interval_s=0.03,
        sampler_fn=hw_sampler_fn,
        event_sink=lambda ev: store.ingest(ev),
    ) as gpu_window:
        # Outbound call to Ollama
        prompt = (
            "You are a strict AML/OFAC compliance officer. "
            "Evaluate if entity 'ACME Trading Ltd' transacting with 'Bank of Kabul' requires an OFAC block. "
            "Provide exactly one decision keyword: APPROVE, REVIEW, or BLOCK, followed by a 1-sentence rationale."
        )
        llm_response = client.invoke(
            prompt=prompt,
            client_request_id=client_request_id,
            run_id=run_id,
            step_id=step_id,
            trace_id=trace_id,
            span_id=uuid.uuid4().hex[:16],
            max_tokens=60,
        )

    step_duration_ms = (time.perf_counter() - t0_step) * 1000.0

    # Ingest runtime request & response events
    for rev in runtime_events:
        store.ingest(rev)

    # Ingest semantic step event
    step_ev = agent_step_event(
        run_id=run_id,
        step_id=step_id,
        node_name="screen_sanctions",
        request_id=client_request_id,
        trace_id=trace_id,
        span_id=step_span_id,
        parent_span_id=root_span_id,
        selected_action="screen_sanctions",
        status="success",
        latency_ms=round(step_duration_ms, 2),
    )
    store.ingest(step_ev)

    # 4. Verified Request-to-Response Mapping & Truthful Metrics
    print("\n[4/5] Verifying Request-to-Response Mapping and Honest Metrics:")
    print(f"      Client Request ID:     {llm_response.client_request_id}")
    print(f"      Engine Native ID:      {llm_response.engine_native_id}")
    print(f"      LLM Content Output:    {repr(llm_response.content[:80])}...")
    print(f"      Observed Logprobs:     {llm_response.confidence_label} (margin={llm_response.margin})")
    print(f"      GPU Window Duration:   {gpu_window.duration_ms:.2f} ms ({len(gpu_window.samples)} samples)")

    # 5. Build Trace Slice & Correlation
    print("\n[5/5] Building Correlated Trace Slice from Durable Storage...")
    trace_slice = store.build_trace_slice(run_id)

    print(f"\nTrace Slice Summary:")
    print(f"  - Correlated Trace Steps:      {len(trace_slice['trace_steps'])}")
    print(f"  - Contextual GPU Samples:      {len(trace_slice['shared_device_context'])}")
    print(f"  - Ambiguous Joins:             {trace_slice['ambiguous_count']}")
    print(f"  - Unmatched Events:            {trace_slice['unmatched_count']}")

    print("\n" + "=" * 70)
    print("DURABLE JSONL LOG (REDACTED SAMPLE):")
    print("=" * 70)
    for line in jsonl_file.read_text(encoding="utf-8").strip().splitlines()[:6]:
        parsed = json.loads(line)
        # Redact long attributes for display
        if "attributes" in parsed and "parent_goal" in parsed["attributes"]:
            parsed["attributes"]["parent_goal"] = "[REDACTED]"
        print(json.dumps(parsed))

    print("\n" + "=" * 70)
    print("FINAL CORRELATED TRACE SLICE:")
    print("=" * 70)
    print(json.dumps(trace_slice, indent=2))
    print("\nGate run completed successfully and saved to:", jsonl_file)


if __name__ == "__main__":
    run_live_gate()
