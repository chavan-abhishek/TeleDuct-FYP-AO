import unittest
from teleduct.semantic.normalize import (
    agent_run_event,
    agent_step_event,
    SemanticStepTracker,
)


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

    def test_agent_run_event(self):
        e = agent_run_event(run_id="run-101", parent_goal="verify transaction")
        self.assertEqual(e.event_type, "agent.run")
        self.assertEqual(e.run_id, "run-101")
        self.assertEqual(e.attributes["parent_goal"], "verify transaction")

    def test_semantic_step_tracker_sequence(self):
        tracker = SemanticStepTracker(run_id="run-toy-01", trace_id="trace-abc")
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

    def test_step_tracker_error_path(self):
        tracker = SemanticStepTracker(run_id="run-err-01")

        def bad_step(state):
            raise RuntimeError("API timeout")

        with self.assertRaises(RuntimeError):
            tracker.trace_step("flaky_node", bad_step, {})

    def test_reject_bad_counts(self):
        with self.assertRaises(ValueError):
            agent_step_event(run_id="r", step_id="s", node_name="n", input_tokens=-1)

    def test_reject_missing_required(self):
        with self.assertRaises(ValueError):
            agent_step_event(run_id="", step_id="s", node_name="n")
        with self.assertRaises(ValueError):
            agent_run_event(run_id="r", parent_goal="")


if __name__ == "__main__":
    unittest.main()
