import json
import tempfile
from pathlib import Path
import unittest
from teleduct.contracts import Event, create_event
from teleduct.fusion.join import (
    correlate_requests,
    EventIngestionStore,
    ConflictingEventError,
)


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

    def test_event_ingestion_store_idempotency_and_conflict_rejection(self):
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
        # 1. First ingestion succeeds
        self.assertTrue(store.ingest(event))
        # 2. Second ingestion of identical event_id is rejected idempotently (returns False, no error)
        self.assertFalse(store.ingest(event))
        self.assertEqual(len(store.get_events()), 1)

        # 3. Third ingestion with same event_id but conflicting payload raises ConflictingEventError
        conflicting_event = create_event(
            event_type="agent.step",
            source="semantic",
            attribution="direct",
            run_id="r1",
            step_id="s_DIFFERENT_STEP",
            request_id="req_DIFFERENT",
            event_id="evt-fixed-uuid-001",
        )
        with self.assertRaises(ConflictingEventError):
            store.ingest(conflicting_event)

    def test_durable_jsonl_persistence_and_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_file = Path(tmpdir) / "events.jsonl"
            store1 = EventIngestionStore(storage_path=jsonl_file)

            ev1 = create_event(
                event_type="agent.run",
                source="semantic",
                attribution="direct",
                run_id="run-persist-1",
                step_id="init",
                event_id="evt-p-1",
            )
            ev2 = create_event(
                event_type="model.request",
                source="runtime",
                attribution="direct",
                request_id="req-p-1",
                event_id="evt-p-2",
            )
            store1.ingest(ev1)
            store1.ingest(ev2)

            self.assertTrue(jsonl_file.exists())
            lines = jsonl_file.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)

            # Reload into brand new store instance
            store2 = EventIngestionStore(storage_path=jsonl_file)
            loaded = store2.get_events()
            self.assertEqual(len(loaded), 2)
            loaded_ids = {e.event_id for e in loaded}
            self.assertEqual(loaded_ids, {"evt-p-1", "evt-p-2"})

    def test_observation_interval_restriction(self):
        store = EventIngestionStore()
        t_base = 1791800000_000_000_000  # ns
        # Event in Run A at T1
        e_run_a = Event(
            event_type="agent.step",
            source="semantic",
            attribution="direct",
            event_id="ev-run-a-1",
            run_id="run-A",
            step_id="step-1",
            request_id="req-A",
            timestamp_unix_ns=t_base,
        )
        # GPU sample inside Run A interval
        gpu_inside = Event(
            event_type="gpu.device_sample",
            source="nvidia_smi",
            attribution="shared_device",
            event_id="ev-gpu-inside",
            timestamp_unix_ns=t_base,
        )
        # GPU sample outside Run A interval (e.g. 5 hours later)
        gpu_outside = Event(
            event_type="gpu.device_sample",
            source="nvidia_smi",
            attribution="shared_device",
            event_id="ev-gpu-outside",
            timestamp_unix_ns=t_base + 5 * 3600 * 1_000_000_000,
        )

        store.ingest_batch([e_run_a, gpu_inside, gpu_outside])

        # Trace slice for run-A should strictly filter GPU samples to observation interval
        slice_a = store.build_trace_slice(
            "run-A",
            start_time_ns=t_base - 1000,
            end_time_ns=t_base + 1000,
        )
        self.assertEqual(len(slice_a["shared_device_context"]), 1)
        parsed_sample = json.loads(slice_a["shared_device_context"][0])
        self.assertEqual(parsed_sample["timestamp_unix_ns"], t_base)

    def test_concurrent_requests_share_gpu_without_direct_allocation(self):
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
