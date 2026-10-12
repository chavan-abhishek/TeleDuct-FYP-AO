"""Cross-component integration tests consuming semantic, runtime, hardware and fusion adapters."""
import json
import tempfile
from pathlib import Path
from typing import TypedDict
import unittest
from unittest.mock import MagicMock

from teleduct.contracts import Event, create_event
from teleduct.semantic import (
    agent_run_event,
    agent_step_event,
    SemanticStepTracker,
    TeleDuctLangGraphCallbackHandler,
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

try:
    from langgraph.graph import StateGraph, START, END
    HAS_LANGGRAPH = True
except ImportError:
    StateGraph = START = END = None  # type: ignore[assignment]
    HAS_LANGGRAPH = False


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
            self.assertEqual(len(matched_step["runtime_events"]), 2)

            # Verify hardware context separation
            self.assertGreater(len(trace_slice["shared_device_context"]), 0)
            for hw_json in trace_slice["shared_device_context"]:
                hw_ev = json.loads(hw_json)
                self.assertEqual(hw_ev["attribution"], "shared_device")
                self.assertNotEqual(hw_ev["attribution"], "direct")

    @unittest.skipUnless(HAS_LANGGRAPH, "langgraph required for genuine StateGraph test")
    def test_real_langgraph_callback_and_model_boundary(self):
        """Execute a genuine LangGraph StateGraph where TeleDuct callback observes actual execution:
        - 3 real nodes (parse -> screen -> decide)
        - Callback intercepts on_chain_start and on_chain_end
        - Middle node executes client.invoke()
        - Proven parent/child span hierarchy matching the graph structure
        """
        class AgentState(TypedDict):
            tx_id: str
            parsed_data: dict
            screening_result: str
            decision: str
            request_id: str

        # Mock LLM for the screening node
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": "chatcmpl-lg-real-001",
            "model": "llama3.2:latest",
            "choices": [{"message": {"content": "DECISION: BLOCK - sanctioned entity"}}]
        }
        mock_session.post.return_value = mock_resp

        store = EventIngestionStore()
        run_id = "run-lg-real-flow-1"
        trace_id = "trace-lg-real-42"

        runtime_events = []
        client = TruthfulInferenceClient(
            base_url="http://mock-llm:11434/v1",
            model="llama3.2:latest",
            event_sink=runtime_events,
            session=mock_session,
        )

        handler = TeleDuctLangGraphCallbackHandler(
            run_id=run_id,
            trace_id=trace_id,
            event_sink=lambda ev: store.ingest(ev),
        )

        builder = StateGraph(AgentState)

        def node_parse(state: AgentState) -> dict:
            return {"parsed_data": {"amount": 50000, "dest": "Kabul"}}

        def node_screen(state: AgentState) -> dict:
            req_id = "client-req-lg-node-02"
            # Active node span from handler
            resp = client.invoke(
                prompt=f"Screen destination: {state['parsed_data']['dest']}",
                client_request_id=req_id,
                run_id=run_id,
                step_id="node_screen",
                trace_id=trace_id,
            )
            return {
                "screening_result": resp.content,
                "request_id": req_id,
                "selected_action": "screen_compliance",
            }

        def node_decide(state: AgentState) -> dict:
            action = "BLOCK" if "BLOCK" in state["screening_result"] else "APPROVE"
            return {"decision": action, "selected_action": action}

        builder.add_node("node_parse", node_parse)
        builder.add_node("node_screen", node_screen)
        builder.add_node("node_decide", node_decide)
        builder.add_edge(START, "node_parse")
        builder.add_edge("node_parse", "node_screen")
        builder.add_edge("node_screen", "node_decide")
        builder.add_edge("node_decide", END)

        graph = builder.compile()

        # Execute genuine graph invocation
        final_state = graph.invoke(
            {"tx_id": "TX-9988"},
            config={"callbacks": [handler], "run_name": "aml_compliance_graph"},
        )

        self.assertEqual(final_state["decision"], "BLOCK")

        # Ingest runtime events from the model call
        for rev in runtime_events:
            store.ingest(rev)

        # Inspect captured events
        captured = store.get_events()
        event_types = [e.event_type for e in captured]

        self.assertIn("agent.run", event_types)
        step_events = [e for e in captured if e.event_type == "agent.step"]
        self.assertEqual(len(step_events), 3)

        # Verify parent span propagation from root agent.run to all 3 node steps
        run_ev = next(e for e in captured if e.event_type == "agent.run")
        self.assertEqual(run_ev.run_id, run_id)
        self.assertEqual(run_ev.trace_id, trace_id)
        self.assertIsNotNone(run_ev.span_id)

        for step in step_events:
            self.assertEqual(step.parent_span_id, run_ev.span_id)
            self.assertEqual(step.run_id, run_id)
            self.assertEqual(step.trace_id, trace_id)

        # Correlate and verify trace slice
        trace_slice = store.build_trace_slice(run_id)
        self.assertEqual(trace_slice["run_id"], run_id)
        self.assertEqual(trace_slice["ambiguous_count"], 0)
        self.assertEqual(trace_slice["unmatched_count"], 0)


if __name__ == "__main__":
    unittest.main()
