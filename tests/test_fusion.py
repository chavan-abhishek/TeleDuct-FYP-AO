import unittest
from teleduct.contracts import create_event
from teleduct.fusion.join import correlate_requests, EventIngestionStore


class FusionTests(unittest.TestCase):
    def test_direct_join_and_shared_device_separation(self):
        s = create_event(
            event_type="agent.step",
            source="semantic",
            attribution="direct",
            run_id="r",
            step_id="s",
            request_id="req",
        )
        r = create_event(
            event_type="model.response",
            source="runtime",
            attribution="direct",
            request_id="req",
        )
        g = create_event(
            event_type="gpu.device_sample",
            source="nvidia_smi",
            attribution="shared_device",
        )
        res = correlate_requests([s, r, g])
        self.assertEqual(len(res["matched"]), 1)
        self.assertEqual(len(res["shared_device"]), 1)
        self.assertEqual(len(res["unmatched"]), 0)

    def test_ambiguous_join_is_not_silent(self):
        a = create_event(
            event_type="agent.step",
            source="semantic",
            attribution="direct",
            run_id="r",
            step_id="s1",
            request_id="req",
        )
        b = create_event(
            event_type="agent.step",
            source="semantic",
            attribution="direct",
            run_id="r",
            step_id="s2",
            request_id="req",
        )
        c = create_event(
            event_type="model.response",
            source="runtime",
            attribution="direct",
            request_id="req",
        )
        res = correlate_requests([a, b, c])
        self.assertEqual(len(res["matched"]), 0)
        self.assertEqual(len(res["ambiguous"]), 1)

    def test_event_ingestion_store_idempotency(self):
        store = EventIngestionStore()
        event = create_event(
            event_type="agent.step",
            source="semantic",
            attribution="direct",
            run_id="r1",
            step_id="s1",
            request_id="req1",
            event_id="evt-fixed-uuid-001",
        )
        # First ingestion succeeds
        self.assertTrue(store.ingest(event))
        # Second ingestion of identical event_id is rejected idempotently
        self.assertFalse(store.ingest(event))
        self.assertEqual(len(store.get_events()), 1)

    def test_event_ingestion_store_batch_and_trace_slice(self):
        store = EventIngestionStore()
        s = create_event(
            event_type="agent.step",
            source="semantic",
            attribution="direct",
            run_id="run-slice-1",
            step_id="step-1",
            request_id="req-slice-1",
        )
        r = create_event(
            event_type="model.response",
            source="runtime",
            attribution="direct",
            run_id="run-slice-1",
            step_id="step-1",
            request_id="req-slice-1",
        )
        g = create_event(
            event_type="gpu.device_sample",
            source="nvidia_smi",
            attribution="shared_device",
        )
        added = store.ingest_batch([s, r, g, s])  # contains duplicate s
        self.assertEqual(added, 3)

        slice_data = store.build_trace_slice("run-slice-1")
        self.assertEqual(slice_data["run_id"], "run-slice-1")
        self.assertEqual(len(slice_data["trace_steps"]), 1)
        self.assertEqual(len(slice_data["shared_device_context"]), 1)
        self.assertEqual(slice_data["ambiguous_count"], 0)
        self.assertEqual(slice_data["unmatched_count"], 0)

    def test_concurrent_requests_share_gpu_without_direct_allocation(self):
        # Negative test: concurrent requests must NOT receive exclusive GPU attribution
        req_a_semantic = create_event(
            event_type="agent.step",
            source="semantic",
            attribution="direct",
            run_id="run-A",
            step_id="step-A",
            request_id="req-A",
        )
        req_b_semantic = create_event(
            event_type="agent.step",
            source="semantic",
            attribution="direct",
            run_id="run-B",
            step_id="step-B",
            request_id="req-B",
        )
        req_a_runtime = create_event(
            event_type="model.response",
            source="runtime",
            attribution="direct",
            request_id="req-A",
        )
        req_b_runtime = create_event(
            event_type="model.response",
            source="runtime",
            attribution="direct",
            request_id="req-B",
        )
        shared_gpu = create_event(
            event_type="gpu.device_sample",
            source="nvml",
            attribution="shared_device",
            metrics={"gpu_vram_used_mib": 4120.0},
        )
        res = correlate_requests([req_a_semantic, req_b_semantic, req_a_runtime, req_b_runtime, shared_gpu])
        self.assertEqual(len(res["matched"]), 2)
        # Shared device sample remains shared_device, never duplicated as direct to A or B
        self.assertEqual(len(res["shared_device"]), 1)
        self.assertEqual(res["shared_device"][0].attribution, "shared_device")

    def test_unmatched_events_isolated(self):
        orphan_semantic = create_event(
            event_type="agent.step",
            source="semantic",
            attribution="direct",
            run_id="r",
            step_id="s",
            request_id="req-no-backend",
        )
        orphan_runtime = create_event(
            event_type="model.response",
            source="runtime",
            attribution="direct",
            request_id="req-no-frontend",
        )
        res = correlate_requests([orphan_semantic, orphan_runtime])
        self.assertEqual(len(res["matched"]), 0)
        self.assertEqual(len(res["unmatched"]), 2)


if __name__ == "__main__":
    unittest.main()
