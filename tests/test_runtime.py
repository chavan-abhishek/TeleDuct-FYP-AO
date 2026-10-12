import unittest
from teleduct.runtime.normalize import (
    inference_event,
    inference_request_event,
    parse_openai_completion_response,
)


class RuntimeTests(unittest.TestCase):
    def test_request_record(self):
        e = inference_event(request_id="req-2", engine="sglang", ttft_ms=43.5, input_tokens=45)
        self.assertEqual(e.request_id, "req-2")
        self.assertEqual(e.metrics["ttft_ms"], 43.5)
        self.assertNotIn("tpot_ms", e.metrics)

    def test_no_fabricated_missing(self):
        e = inference_event(request_id="req-2", engine="sglang")
        self.assertEqual(e.metrics, {})

    def test_inference_request_event(self):
        e = inference_request_event(
            request_id="req-sglang-001",
            engine="sglang",
            model="openai/gpt-oss-20b",
            run_id="run-1",
            step_id="step-1",
            span_id="span-1",
        )
        self.assertEqual(e.event_type, "model.request")
        self.assertEqual(e.request_id, "req-sglang-001")
        self.assertEqual(e.attributes["model"], "openai/gpt-oss-20b")

    def test_parse_response_with_measured_logprobs(self):
        payload = {
            "id": "chatcmpl-test-001",
            "model": "openai/gpt-oss-20b",
            "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": "APPROVE"},
                "logprobs": {
                    "content": [{
                        "token": "APPROVE",
                        "top_logprobs": [
                            {"token": "APPROVE", "logprob": -0.1},
                            {"token": "FLAG", "logprob": -2.4},
                        ]
                    }]
                }
            }]
        }
        e = parse_openai_completion_response(payload, latency_ms=125.4, engine="sglang")
        self.assertEqual(e.event_type, "model.response")
        self.assertEqual(e.request_id, "chatcmpl-test-001")
        self.assertEqual(e.metrics["prompt_tokens"], 12)
        self.assertEqual(e.metrics["completion_tokens"], 5)
        self.assertEqual(e.metrics["latency_ms"], 125.4)
        self.assertNotIn("ttft_ms", e.metrics)  # Non-streaming must NOT report TTFT
        self.assertIn("sampling_margin", e.metrics)
        self.assertGreater(e.metrics["sampling_margin"], 0.5)
        self.assertEqual(e.attributes["logprobs_status"], "measured")
        self.assertFalse(e.attributes["fragility_flag"])

    def test_parse_response_without_logprobs_never_fabricates_margin(self):
        payload = {
            "id": "chatcmpl-test-002",
            "model": "llama3.2:latest",
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]
        }
        e = parse_openai_completion_response(payload, latency_ms=88.0, engine="ollama")
        self.assertEqual(e.request_id, "chatcmpl-test-002")
        self.assertEqual(e.attributes["logprobs_status"], "unavailable")
        self.assertNotIn("sampling_margin", e.metrics)
        self.assertNotIn("fragility_flag", e.attributes)

    def test_parse_response_missing_native_id_raises(self):
        payload = {"model": "test", "choices": []}
        with self.assertRaisesRegex(ValueError, "native 'id'"):
            parse_openai_completion_response(payload, latency_ms=10.0)

    def test_negative_latency_raises(self):
        payload = {"id": "req-1", "model": "test"}
        with self.assertRaisesRegex(ValueError, "non-negative"):
            parse_openai_completion_response(payload, latency_ms=-5.0)


if __name__ == "__main__":
    unittest.main()
