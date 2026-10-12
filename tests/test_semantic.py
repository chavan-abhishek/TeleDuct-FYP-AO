import unittest
from typing import TypedDict
from teleduct.semantic.normalize import (
    agent_run_event,
    agent_step_event,
    SemanticStepTracker,
    TeleDuctLangGraphCallbackHandler,
)
try:
    from langgraph.graph import StateGraph, START, END
    HAS_LANGGRAPH = True
except ImportError:
    StateGraph = START = END = None  # type: ignore[assignment]
    HAS_LANGGRAPH = False


class SemanticTests(unittest.TestCase):
    def test_normalization_and_join_ids(self):
        e = agent_step_event(
            run_id="r",
            step_id="s",
            request_id="req",
            node_name="plan",
            input_tokens=10,
            output_tokens=3,
        )
        self.assertEqual((e.run_id, e.step_id, e.request_id), ("r", "s", "req"))
        self.assertEqual(e.metrics["input_tokens"], 10)
        self.assertNotIn("prompt", e.attributes)

    def test_agent_run_event_privacy_default(self):
        # Default: raw parent_goal is NOT stored; parent_goal_hash is stored
        e = agent_run_event(run_id="run-101", parent_goal="verify transaction")
        self.assertEqual(e.event_type, "agent.run")
        self.assertEqual(e.run_id, "run-101")
        self.assertNotIn("parent_goal", e.attributes)
        self.assertIn("parent_goal_hash", e.attributes)
        self.assertEqual(len(e.attributes["parent_goal_hash"]), 16)

        # Explicit opt-in: capture_raw_goal=True stores the raw string
        e_optin = agent_run_event(
            run_id="run-101",
            parent_goal="verify transaction",
            capture_raw_goal=True,
        )
        self.assertEqual(e_optin.attributes["parent_goal"], "verify transaction")

    def test_semantic_step_tracker_sequence(self):
        sink = []
        tracker = SemanticStepTracker(run_id="run-toy-01", trace_id="trace-abc", event_sink=sink)
        # Step 1: parse
        res1, ev1 = tracker.trace_step(
            "parse_input",
            lambda s: {"tx": s["raw"], "selected_action": "parsed"},
            {"raw": "tx-12345"},
        )
        self.assertEqual(ev1.step_id, "step_000_parse_input")
        self.assertEqual(ev1.trace_id, "trace-abc")
        self.assertEqual(ev1.attributes["status"], "success")

        # Step 2: evaluate
        res2, ev2 = tracker.trace_step(
            "evaluate_policy",
            lambda s: {"status": "FLAG", "request_id": "req-model-99", "selected_action": "flag"},
            res1,
        )
        self.assertEqual(ev2.step_id, "step_001_evaluate_policy")
        self.assertEqual(ev2.request_id, "req-model-99")
        self.assertEqual(ev2.trace_id, "trace-abc")

        # Step 3: action
        res3, ev3 = tracker.trace_step(
            "notify_compliance",
            lambda s: {"notified": True, "selected_action": "notified"},
            res2,
            tool_name="email_compliance",
        )
        self.assertEqual(ev3.step_id, "step_002_notify_compliance")
        self.assertEqual(ev3.attributes["tool_name"], "email_compliance")
        self.assertEqual(len(tracker.events), 3)
        self.assertEqual(len(sink), 3)

    def test_step_tracker_error_path_emission(self):
        sink = []
        tracker = SemanticStepTracker(run_id="run-err-01", event_sink=sink)

        def bad_step(state):
            raise RuntimeError("API timeout")

        with self.assertRaises(RuntimeError):
            tracker.trace_step("flaky_node", bad_step, {})

        # Verify error event was emitted and persisted
        self.assertEqual(len(tracker.events), 1)
        self.assertEqual(len(sink), 1)
        err_ev = tracker.events[0]
        self.assertEqual(err_ev.attributes["status"], "error")
        self.assertEqual(err_ev.attributes["error_type"], "RuntimeError")
        self.assertIn("API timeout", err_ev.attributes["error_message"])
        self.assertIsNotNone(tracker.last_error_event)
        self.assertEqual(tracker.last_error_event.event_id, err_ev.event_id)

    def test_reject_bad_counts(self):
        with self.assertRaises(ValueError):
            agent_step_event(run_id="r", step_id="s", node_name="n", input_tokens=-1)

    def test_reject_missing_required(self):
        with self.assertRaises(ValueError):
            agent_step_event(run_id="", step_id="s", node_name="n")
        with self.assertRaises(ValueError):
            agent_run_event(run_id="")

    @unittest.skipUnless(HAS_LANGGRAPH, "langgraph required for graph execution tests")
    def test_langgraph_instrumentation_success(self):
        class State(TypedDict):
            query: str
            plan: str
            result: str

        builder = StateGraph(State)

        def parse_node(state: State) -> dict:
            return {"plan": f"plan for {state['query']}"}

        def execute_node(state: State) -> dict:
            return {"result": f"executed {state['plan']}", "selected_action": "executed"}

        builder.add_node("parse", parse_node)
        builder.add_node("execute", execute_node)
        builder.add_edge(START, "parse")
        builder.add_edge("parse", "execute")
        builder.add_edge("execute", END)

        graph = builder.compile()

        sink = []
        handler = TeleDuctLangGraphCallbackHandler(run_id="run-lg-01", event_sink=sink)
        output = graph.invoke(
            {"query": "check AML rules"},
            config={"callbacks": [handler], "run_name": "aml_workflow"},
        )

        self.assertIn("result", output)
        # Should have captured agent.run + 2 agent.step events
        types = [e.event_type for e in handler.events]
        self.assertIn("agent.run", types)
        step_events = [e for e in handler.events if e.event_type == "agent.step"]
        self.assertEqual(len(step_events), 2)

        # Parent/child span propagation validation
        run_event = next(e for e in handler.events if e.event_type == "agent.run")
        self.assertIsNotNone(run_event.span_id)

        for step in step_events:
            self.assertEqual(step.parent_span_id, run_event.span_id)
            self.assertEqual(step.attributes["status"], "success")

    @unittest.skipUnless(HAS_LANGGRAPH, "langgraph required for graph execution tests")
    def test_langgraph_instrumentation_error_propagation(self):
        class State(TypedDict):
            value: int

        builder = StateGraph(State)

        def good_node(state: State) -> dict:
            return {"value": state["value"] + 1}

        def failing_node(state: State) -> dict:
            raise ValueError("Intentional step failure")

        builder.add_node("good", good_node)
        builder.add_node("failing", failing_node)
        builder.add_edge(START, "good")
        builder.add_edge("good", "failing")
        builder.add_edge("failing", END)

        graph = builder.compile()

        handler = TeleDuctLangGraphCallbackHandler(run_id="run-lg-fail")
        with self.assertRaises(ValueError):
            graph.invoke({"value": 10}, config={"callbacks": [handler]})

        # Verify error event was captured with span hierarchy
        run_event = next(e for e in handler.events if e.event_type == "agent.run")
        step_events = [e for e in handler.events if e.event_type == "agent.step"]
        self.assertEqual(len(step_events), 2)

        good_step = step_events[0]
        self.assertEqual(good_step.attributes["status"], "success")
        self.assertEqual(good_step.parent_span_id, run_event.span_id)

        fail_step = step_events[1]
        self.assertEqual(fail_step.attributes["status"], "error")
        self.assertEqual(fail_step.attributes["error_type"], "ValueError")
        self.assertIn("Intentional step failure", fail_step.attributes["error_message"])
        self.assertEqual(fail_step.parent_span_id, run_event.span_id)


if __name__ == "__main__":
    unittest.main()
