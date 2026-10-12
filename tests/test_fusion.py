import concurrent.futures
import json
import tempfile
from pathlib import Path
import unittest
from teleduct.contracts import Event, create_event
from teleduct.fusion.join import (
    correlate_requests,
    EventIngestionStore,
    ConflictingEventError,
    PersistenceError,
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
        self.assertTrue(store.ingest(event))
        self.assertFalse(store.ingest(event))
        self.assertEqual(len(store.get_events()), 1)

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

            store2 = EventIngestionStore(storage_path=jsonl_file)
            loaded = store2.get_events()
            self.assertEqual(len(loaded), 2)
            loaded_ids = {e.event_id for e in loaded}
            self.assertEqual(loaded_ids, {"evt-p-1", "evt-p-2"})

    def test_persistence_failure_fails_closed_without_in_memory_state(self):
        # Point to an impossible read-only path
        invalid_path = Path("/dev/null/forbidden_dir/events.jsonl")
        store = EventIngestionStore(storage_path=invalid_path)

        ev = create_event(
            event_type="agent.run",
            source="semantic",
            attribution="direct",
            run_id="run-fail-1",
            step_id="init",
            event_id="evt-fail-1",
        )
        # Ingestion must raise PersistenceError
        with self.assertRaises(PersistenceError):
            store.ingest(ev)

        # In-memory store must NOT retain the event after failed persistence
        self.assertEqual(len(store.get_events()), 0)
        self.assertNotIn("evt-fail-1", store._events)

    def test_restart_recovery_with_malformed_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_file = Path(tmpdir) / "corrupted.jsonl"
            valid_ev = create_event(
                event_type="agent.run",
                source="semantic",
                attribution="direct",
                run_id="run-valid",
                step_id="init",
                event_id="evt-valid",
            )
            # Write 1 valid line, 1 malformed JSON line, 1 invalid schema line, 1 valid line
            content = (
                valid_ev.to_json() + "\n"
                + "MALFORMED_JSON_STRING_HERE{{{\n"
                + '{"event_type": "invalid_event_missing_schema_version"}\n'
            )
            jsonl_file.write_text(content, encoding="utf-8")

            store = EventIngestionStore(storage_path=jsonl_file)
            events = store.get_events()
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].event_id, "evt-valid")
            self.assertEqual(store.corrupted_records_count, 2)

    def test_concurrent_thread_safe_ingestion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_file = Path(tmpdir) / "concurrent_events.jsonl"
            store = EventIngestionStore(storage_path=jsonl_file)

            def worker(idx: int):
                ev = create_event(
                    event_type="agent.step",
                    source="semantic",
                    attribution="direct",
                    run_id="run-concurrent",
                    step_id=f"step_{idx:03d}",
                    request_id=f"req_{idx:03d}",
                    event_id=f"evt-concurrent-{idx:03d}",
                )
                return store.ingest(ev)

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(worker, range(20)))

            self.assertEqual(sum(results), 20)
            self.assertEqual(len(store.get_events()), 20)

            # Reload into a fresh store to verify all 20 lines on disk
            reloaded = EventIngestionStore(storage_path=jsonl_file)
            self.assertEqual(len(reloaded.get_events()), 20)

    def test_duplicate_engine_native_ids_do_not_conflate_distinct_client_requests(self):
        # Scenario: Engine issues duplicate native ID 'chatcmpl-reused' for two distinct requests
        req_a_semantic = create_event(
            event_type="agent.step",
            source="semantic",
            attribution="direct",
            run_id="run-1",
            step_id="step-1",
            request_id="client-req-001",
        )
        req_b_semantic = create_event(
            event_type="agent.step",
            source="semantic",
            attribution="direct",
            run_id="run-1",
            step_id="step-2",
            request_id="client-req-002",
        )
        resp_a_runtime = create_event(
            event_type="model.response",
            source="runtime",
            attribution="direct",
            request_id="client-req-001",
            attributes={"client_request_id": "client-req-001", "engine_native_id": "chatcmpl-reused"},
        )
        resp_b_runtime = create_event(
            event_type="model.response",
            source="runtime",
            attribution="direct",
            request_id="client-req-002",
            attributes={"client_request_id": "client-req-002", "engine_native_id": "chatcmpl-reused"},
        )

        res = correlate_requests([req_a_semantic, req_b_semantic, resp_a_runtime, resp_b_runtime])
        self.assertEqual(len(res["matched"]), 2)
        matched_req_ids = {m["request_id"] for m in res["matched"]}
        self.assertEqual(matched_req_ids, {"client-req-001", "client-req-002"})
        self.assertEqual(len(res["ambiguous"]), 0)

    def test_unmatched_requests_and_responses_explicit_categorization(self):
        orphan_req = create_event(
            event_type="agent.step",
            source="semantic",
            attribution="direct",
            run_id="r",
            step_id="s",
            request_id="req-dropped-at-network",
        )
        orphan_resp = create_event(
            event_type="model.response",
            source="runtime",
            attribution="direct",
            request_id="req-phantom-response",
        )
        res = correlate_requests([orphan_req, orphan_resp])
        self.assertEqual(len(res["matched"]), 0)
        self.assertEqual(len(res["unmatched"]), 2)
        self.assertEqual(len(res["unmatched_requests"]), 1)
        self.assertEqual(len(res["unmatched_responses"]), 1)

    def test_observation_interval_restriction(self):
        store = EventIngestionStore()
        t_base = 1791800000_000_000_000  # ns
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
        gpu_inside = Event(
            event_type="gpu.device_sample",
            source="nvidia_smi",
            attribution="shared_device",
            event_id="ev-gpu-inside",
            timestamp_unix_ns=t_base,
        )
        gpu_outside = Event(
            event_type="gpu.device_sample",
            source="nvidia_smi",
            attribution="shared_device",
            event_id="ev-gpu-outside",
            timestamp_unix_ns=t_base + 5 * 3600 * 1_000_000_000,
        )

        store.ingest_batch([e_run_a, gpu_inside, gpu_outside])

        slice_a = store.build_trace_slice(
            "run-A",
            start_time_ns=t_base - 1000,
            end_time_ns=t_base + 1000,
        )
        self.assertEqual(len(slice_a["shared_device_context"]), 1)
        parsed_sample = json.loads(slice_a["shared_device_context"][0])
        self.assertEqual(parsed_sample["timestamp_unix_ns"], t_base)


if __name__ == "__main__":
    unittest.main()
