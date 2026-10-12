import unittest
from teleduct.runtime.normalize import inference_event


class RuntimeTests(unittest.TestCase):
    def test_request_record(self):
        e = inference_event(request_id="req-2", engine="sglang", ttft_ms=43.5, input_tokens=45)
        self.assertEqual(e.request_id, "req-2")
        self.assertEqual(e.metrics["ttft_ms"], 43.5)
        self.assertNotIn("tpot_ms", e.metrics)

    def test_no_fabricated_missing(self):
        e = inference_event(request_id="req-2", engine="sglang")
        self.assertEqual(e.metrics, {})
