import concurrent.futures
import unittest
from unittest.mock import MagicMock
from teleduct.runtime.normalize import (
    inference_event,
    inference_request_event,
    parse_openai_completion_response,
)
from teleduct.runtime.adapter import (
    TruthfulInferenceClient,
    TruthfulCompletionResponse,
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

    def test_inference_request_event_client_id(self):
        e = inference_request_event(
            client_request_id="client-req-001",
            engine="sglang",
            model="openai/gpt-oss-20b",
            run_id="run-1",
            step_id="step-1",
            span_id="span-1",
        )
        self.assertEqual(e.event_type, "model.request")
        self.assertEqual(e.request_id, "client-req-001")
        self.assertEqual(e.attributes["client_request_id"], "client-req-001")
        self.assertEqual(e.attributes["model"], "openai/gpt-oss-20b")

    def test_parse_response_with_measured_logprobs(self):
        payload = {
            "id": "chatcmpl-native-123",
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
        e = parse_openai_completion_response(
            payload,
            latency_ms=125.4,
            client_request_id="client-req-123",
            engine="sglang",
        )
        self.assertEqual(e.event_type, "model.response")
        # Primary request_id is client_request_id for end-to-end trace correlation
        self.assertEqual(e.request_id, "client-req-123")
        self.assertEqual(e.attributes["client_request_id"], "client-req-123")
        self.assertEqual(e.attributes["engine_native_id"], "chatcmpl-native-123")
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
        e = parse_openai_completion_response(
            payload,
            latency_ms=88.0,
            client_request_id="client-req-88",
            engine="ollama",
        )
        self.assertEqual(e.request_id, "client-req-88")
        self.assertEqual(e.attributes["engine_native_id"], "chatcmpl-test-002")
        self.assertEqual(e.attributes["logprobs_status"], "unavailable")
        self.assertNotIn("sampling_margin", e.metrics)
        self.assertNotIn("fragility_flag", e.attributes)

    def test_parse_response_missing_ids_raises(self):
        payload = {"model": "test", "choices": []}
        with self.assertRaisesRegex(ValueError, "missing native 'id'"):
            parse_openai_completion_response(payload, latency_ms=10.0)

    def test_negative_latency_raises(self):
        payload = {"id": "req-1", "model": "test"}
        with self.assertRaisesRegex(ValueError, "non-negative"):
            parse_openai_completion_response(payload, latency_ms=-5.0)

    def test_adapter_invoke_and_mapping(self):
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": "engine-cmpl-42",
            "model": "llama3.2:latest",
            "choices": [{"message": {"content": "APPROVE: valid"}}]
        }
        mock_session.post.return_value = mock_resp

        events = []
        client = TruthfulInferenceClient(
            base_url="http://mock-llm:11434/v1",
            model="llama3.2:latest",
            event_sink=events,
            session=mock_session,
        )

        resp = client.invoke("Analyze risk", client_request_id="my-client-req-42")
        self.assertEqual(resp.content, "APPROVE: valid")
        self.assertEqual(resp.client_request_id, "my-client-req-42")
        self.assertEqual(resp.engine_native_id, "engine-cmpl-42")
        self.assertIsNone(resp.margin)
        self.assertEqual(resp.confidence_label, "UNAVAILABLE")

        # Two events emitted: model.request and model.response
        self.assertEqual(len(events), 2)
        req_ev, resp_ev = events[0], events[1]
        self.assertEqual(req_ev.event_type, "model.request")
        self.assertEqual(req_ev.request_id, "my-client-req-42")
        self.assertEqual(resp_ev.event_type, "model.response")
        self.assertEqual(resp_ev.request_id, "my-client-req-42")
        self.assertEqual(resp_ev.attributes["engine_native_id"], "engine-cmpl-42")

    def test_ten_concurrent_requests_maintain_correct_identity(self):
        def fake_post(url, json=None, headers=None, timeout=None):
            client_req = headers.get("X-Request-ID") if headers else "unknown"
            mock = MagicMock()
            mock.status_code = 200
            mock.json.return_value = {
                "id": f"engine-ack-for-{client_req}",
                "model": "llama3.2:latest",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "choices": [{"message": {"content": f"response to {client_req}"}}]
            }
            return mock

        mock_session = MagicMock()
        mock_session.post.side_effect = fake_post

        sink = []
        client = TruthfulInferenceClient(
            base_url="http://mock-llm:11434/v1",
            model="llama3.2:latest",
            event_sink=sink,
            session=mock_session,
        )

        def worker(idx: int):
            cid = f"concurrent-client-req-{idx:02d}"
            res = client.invoke(f"prompt {idx}", client_request_id=cid)
            return res

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, i) for i in range(10)]
            results = [f.result() for f in futures]

        self.assertEqual(len(results), 10)
        seen_client_ids = set()
        seen_engine_ids = set()

        for res in results:
            self.assertIn("concurrent-client-req-", res.client_request_id)
            self.assertEqual(res.engine_native_id, f"engine-ack-for-{res.client_request_id}")
            seen_client_ids.add(res.client_request_id)
            seen_engine_ids.add(res.engine_native_id)

        # All 10 client IDs and engine IDs must be distinct (zero cross-talk)
        self.assertEqual(len(seen_client_ids), 10)
        self.assertEqual(len(seen_engine_ids), 10)

        # Verify emitted events: exactly 10 requests and 10 responses, perfectly paired
        req_events = [e for e in sink if e.event_type == "model.request"]
        resp_events = [e for e in sink if e.event_type == "model.response"]
        self.assertEqual(len(req_events), 10)
        self.assertEqual(len(resp_events), 10)

        for req in req_events:
            matching_resp = next(r for r in resp_events if r.request_id == req.request_id)
            self.assertEqual(matching_resp.attributes["engine_native_id"], f"engine-ack-for-{req.request_id}")


if __name__ == "__main__":
    unittest.main()
