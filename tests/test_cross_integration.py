"""Cross-component integration test consuming semantic, runtime, hardware and fusion adapters."""
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import MagicMock

# Dynamically link sister worktree paths if running in standalone worktree mode
import teleduct
_BASE_WT = Path(__file__).resolve().parents[2]
for _wt in ("semantic", "runtime", "hardware"):
    _p = str(_BASE_WT / _wt / "src" / "teleduct")
    if _p not in teleduct.__path__ and Path(_p).exists():
        teleduct.__path__.append(_p)

from teleduct.contracts import Event, create_event
from teleduct.semantic import (
    agent_run_event,
    agent_step_event,
    SemanticStepTracker,
)
from teleduct.runtime import (
    TruthfulInferenceClient,
    parse_openai_completion_response,
)
from teleduct.hardware import (
    DeviceTelemetryWindow,
)
from teleduct.fusion import (
    EventIngestionStore,
    correlate_requests,
)


class CrossComponentIntegrationTests(unittest.TestCase):
    def test_full_pipeline_cross_adapter_integration(self):
        """End-to-end integration pipeline:
        1. Semantic step tracker creates run and step boundaries.
        2. Runtime adapter handles outbound LLM request and parses response.
        3. Hardware window records time-aligned contextual GPU metrics.
        4. Fusion store ingests and persists all events to durable JSONL.
        5. Trace slice correlation produces a unified trace with 0 ambiguous joins.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "live_trace.jsonl"
            store = EventIngestionStore(storage_path=jsonl_path)

            run_id = "run-integration-test-01"
            trace_id = "trace-e2e-aabbcc"

            # 1. Semantic: agent run initialization (privacy-preserving)
            run_ev = agent_run_event(
                run_id=run_id,
                parent_goal="Validate compliance with anti-fraud rules",
                trace_id=trace_id,
                span_id="span-root-001",
            )
            store.ingest(run_ev)

            # 2. Mock LLM session for runtime adapter
            mock_session = MagicMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "id": "chatcmpl-engine-999",
                "model": "llama3.2:latest",
                "usage": {"prompt_tokens": 25, "completion_tokens": 15, "total_tokens": 40},
                "choices": [{"message": {"content": "Decision: APPROVED"}}]
            }
            mock_session.post.return_value = mock_resp

            runtime_events = []
            client = TruthfulInferenceClient(
                base_url="http://mock-llm:11434/v1",
                model="llama3.2:latest",
                engine="ollama",
                event_sink=runtime_events,
                session=mock_session,
            )

            # 3. Hardware window + Semantic step + Runtime execution
            mock_gpu_sample = lambda: create_event(
                event_type="gpu.device_sample",
                source="nvidia_smi",
                attribution="shared_device",
                metrics={"gpu_util_pct": 35.0, "power_w": 45.2, "vram_used_mib": 3200.0},
            )

            client_req_id = "req-client-e2e-001"
            with DeviceTelemetryWindow(
                gpu_index=0,
                sample_interval_s=0.02,
                sampler_fn=mock_gpu_sample,
                event_sink=lambda ev: store.ingest(ev),
            ) as window:
                # Semantic step
                step_ev = agent_step_event(
                    run_id=run_id,
                    step_id="step_000_evaluate_risk",
                    node_name="evaluate_risk",
                    request_id=client_req_id,
                    trace_id=trace_id,
                    span_id="span-step-001",
                    parent_span_id="span-root-001",
                    status="success",
                )
                store.ingest(step_ev)

                # Runtime call
                llm_res = client.invoke(
                    prompt="Evaluate risk score for transaction",
                    client_request_id=client_req_id,
                    run_id=run_id,
                    step_id="step_000_evaluate_risk",
                    span_id="span-llm-001",
                    trace_id=trace_id,
                )

            # Ingest runtime events
            for rev in runtime_events:
                store.ingest(rev)

            # 4. Validate durable JSONL persistence
            self.assertTrue(jsonl_path.exists())
            raw_lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertGreater(len(raw_lines), 3)

            # 5. Build correlated trace slice
            trace_slice = store.build_trace_slice(run_id)
            self.assertEqual(trace_slice["run_id"], run_id)
            self.assertEqual(trace_slice["ambiguous_count"], 0)
            self.assertEqual(trace_slice["unmatched_count"], 0)

            # Verify request mapping in trace step
            matched_step = next(s for s in trace_slice["trace_steps"] if s["request_id"] == client_req_id)
            self.assertEqual(matched_step["attribution"], "direct")
            self.assertEqual(len(matched_step["semantic_events"]), 1)
            # Both model.request and model.response should be linked
            self.assertEqual(len(matched_step["runtime_events"]), 2)

            # Verify hardware context separation
            self.assertGreater(len(trace_slice["shared_device_context"]), 0)
            for hw_json in trace_slice["shared_device_context"]:
                hw_ev = json.loads(hw_json)
                self.assertEqual(hw_ev["attribution"], "shared_device")
                self.assertNotEqual(hw_ev["attribution"], "direct")


if __name__ == "__main__":
    unittest.main()
