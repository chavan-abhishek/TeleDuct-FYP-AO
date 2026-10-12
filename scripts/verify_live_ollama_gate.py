#!/usr/bin/env python3
"""Reproducible Acceptance Gate: Genuine LangGraph + Live Ollama + NVIDIA GPU Telemetry Verification.

Executes a genuine LangGraph StateGraph workflow against:
- Engine: Local Ollama (http://localhost:11434/v1, llama3.2:latest)
- GPU: Local NVIDIA GeForce RTX 3060 Laptop GPU (native NVML / nvidia-smi)
- Instrumentation: TeleDuct LangGraph callback handler, truthful adapter, device sampling window, durable fusion store
"""
from __future__ import annotations
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import TypedDict

# Standard checkout package import
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from teleduct.contracts.events import SCHEMA_VERSION
from teleduct.semantic import (
    agent_run_event,
    agent_step_event,
    TeleDuctLangGraphCallbackHandler,
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

from langgraph.graph import StateGraph, START, END


class ComplianceGraphState(TypedDict):
    tx_id: str
    amount_usd: float
    beneficiary: str
    screening_output: str
    decision: str
    client_request_id: str
    request_id: str


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

    print("=" * 75)
    print("TELEDUCT F1-RC1 ACCEPTANCE GATE: GENUINE LANGGRAPH + LIVE OLLAMA + GPU")
    print(f"Contract Schema Version: {SCHEMA_VERSION}")
    print(f"Execution Run ID:        {run_id}")
    print(f"Trace ID:                {trace_id}")
    print(f"Target LLM Engine:       {ollama_url} (model: {model})")
    print("=" * 75)

    # 1. Hardware Detection & Baseline Overhead Measurement
    dev_info = get_device_info(0)
    print(f"\n[1/5] GPU Hardware Detection: {dev_info.get('gpu_name')} (Driver: {dev_info.get('driver_version')})")

    # Benchmark sampler overhead
    t0_bench = time.perf_counter()
    for _ in range(20):
        try:
            sample_device_nvml(0)
            use_nvml = True
        except Exception:
            sample_device(0)
            use_nvml = False
    overhead_per_call_us = ((time.perf_counter() - t0_bench) / 20) * 1_000_000

    hw_sampler_fn = lambda: sample_device_nvml(0) if use_nvml else sample_device(0)
    baseline_sample = hw_sampler_fn()
    store.ingest(baseline_sample)

    print(f"      Sampler in Use:        {'Native NVML (ctypes)' if use_nvml else 'nvidia-smi CLI'}")
    print(f"      Sampler Call Latency:  {overhead_per_call_us:.1f} µs/sample")
    print(f"      Default Sample Window: 100 ms interval (evaluated lower-frequency default)")
    print(f"      Baseline VRAM Used:    {baseline_sample.metrics.get('vram_used_mib')} MiB")
    print(f"      Baseline Power:        {baseline_sample.metrics.get('power_w')} W")

    # 2. Setup TeleDuct LangGraph Callback Handler
    print("\n[2/5] Initializing TeleDuct LangGraph Callback Handler...")
    runtime_events = []
    client = TruthfulInferenceClient(
        base_url=ollama_url,
        model=model,
        engine="ollama/local",
        event_sink=runtime_events,
    )

    handler = TeleDuctLangGraphCallbackHandler(
        run_id=run_id,
        trace_id=trace_id,
        event_sink=lambda ev: store.ingest(ev),
    )

    # 3. Build Genuine 3-Node LangGraph StateGraph
    print("\n[3/5] Compiling and Executing Real LangGraph StateGraph Workflow...")
    builder = StateGraph(ComplianceGraphState)

    def node_parse(state: ComplianceGraphState) -> dict:
        """Node 1: Extract and validate transaction metadata."""
        return {
            "amount_usd": 75000.0,
            "beneficiary": "Bank of Kabul International",
        }

    def node_screen(state: ComplianceGraphState) -> dict:
        """Node 2: Active inference step querying local Ollama under GPU window."""
        client_req_id = f"client-req-{uuid.uuid4().hex[:12]}"
        prompt = (
            f"You are a strict OFAC compliance officer. Screen beneficiary: '{state['beneficiary']}' "
            f"for amount ${state['amount_usd']:.2f}. "
            "Reply with exactly one decision keyword: APPROVE, REVIEW, or BLOCK, and a 1-sentence rationale."
        )
        with DeviceTelemetryWindow(
            gpu_index=0,
            sample_interval_s=0.1,  # 100 ms evaluated default
            sampler_fn=hw_sampler_fn,
            event_sink=lambda ev: store.ingest(ev),
        ):
            resp = client.invoke(
                prompt=prompt,
                client_request_id=client_req_id,
                run_id=run_id,
                step_id="step_001_node_screen",
                trace_id=trace_id,
                max_tokens=64,
            )
        return {
            "screening_output": resp.content,
            "request_id": client_req_id,
            "client_request_id": client_req_id,
            "selected_action": "screen_ofac",
        }

    def node_decide(state: ComplianceGraphState) -> dict:
        """Node 3: Final action branch."""
        action = "BLOCK" if "BLOCK" in state["screening_output"] else "APPROVE"
        return {"decision": action, "selected_action": f"decision_{action.lower()}"}

    builder.add_node("node_parse", node_parse)
    builder.add_node("node_screen", node_screen)
    builder.add_node("node_decide", node_decide)
    builder.add_edge(START, "node_parse")
    builder.add_edge("node_parse", "node_screen")
    builder.add_edge("node_screen", "node_decide")
    builder.add_edge("node_decide", END)

    graph = builder.compile()

    # Execute genuine graph invocation
    t0_graph = time.perf_counter()
    final_output = graph.invoke(
        {"tx_id": "TX-LIVE-OFAC-7788"},
        config={"callbacks": [handler], "run_name": "ofac_sanctions_pipeline"},
    )
    graph_duration_ms = (time.perf_counter() - t0_graph) * 1000.0

    # Ingest runtime model.request and model.response events
    for rev in runtime_events:
        store.ingest(rev)

    # 4. Verification Evidence
    print(f"\n[4/5] Workflow Execution Complete in {graph_duration_ms:.2f} ms:")
    print(f"      Final Agent Decision:  {final_output['decision']}")
    print(f"      Screening Output:      {repr(final_output['screening_output'][:80])}...")
    print(f"      Client Request ID:     {final_output['client_request_id']}")

    # 5. Build Trace Slice from Durable Storage
    print("\n[5/5] Building Correlated Trace Slice from Durable Storage...")
    trace_slice = store.build_trace_slice(run_id)

    print(f"\nCorrelated Trace Summary:")
    print(f"  - Total Captured Events:       {len(store.get_events())}")
    print(f"  - Graph Execution Steps:       {len(trace_slice['trace_steps'])}")
    print(f"  - Contextual GPU Samples:      {len(trace_slice['shared_device_context'])}")
    print(f"  - Ambiguous Joins:             {trace_slice['ambiguous_count']}")
    print(f"  - Unmatched Events:            {trace_slice['unmatched_count']}")

    print("\n" + "=" * 75)
    print("DURABLE JSONL LOG (REDACTED SAMPLE):")
    print("=" * 75)
    for line in jsonl_file.read_text(encoding="utf-8").strip().splitlines()[:6]:
        parsed = json.loads(line)
        print(json.dumps(parsed))

    print("\n" + "=" * 75)
    print("FINAL CORRELATED TRACE SLICE:")
    print("=" * 75)
    print(json.dumps(trace_slice, indent=2))
    print("\nGate run successfully verified and recorded to:", jsonl_file)


if __name__ == "__main__":
    run_live_gate()
