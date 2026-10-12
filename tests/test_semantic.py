import unittest
from teleduct.semantic.normalize import agent_step_event


class SemanticTests(unittest.TestCase):
    def test_normalization_and_join_ids(self):
        e = agent_step_event(run_id="r", step_id="s", request_id="req", node_name="plan",
                             input_tokens=10, output_tokens=3)
        self.assertEqual((e.run_id, e.step_id, e.request_id), ("r", "s", "req"))
        self.assertEqual(e.metrics["input_tokens"], 10)
        self.assertNotIn("prompt", e.attributes)

    def test_reject_bad_counts(self):
        with self.assertRaises(ValueError):
            agent_step_event(run_id="r", step_id="s", node_name="n", input_tokens=-1)
