import unittest
import uuid
from teleduct.contracts import Event, create_event


class ContractTests(unittest.TestCase):
    def test_roundtrip(self):
        event = create_event(
            event_type="agent.step",
            source="semantic",
            attribution="direct",
            trace_id="0af7651916cd43dd8448eb211c80319c",
            span_id="b7ad6b7169203331",
            parent_span_id="00f067aa0ba902b7",
            run_id="run-1",
            step_id="step-1",
            metrics={"latency_ms": 7.2},
        )
        roundtripped = Event.from_json(event.to_json())
        self.assertEqual(roundtripped, event)
        self.assertEqual(roundtripped.event_id, event.event_id)
        self.assertEqual(roundtripped.span_id, "b7ad6b7169203331")
        self.assertEqual(roundtripped.parent_span_id, "00f067aa0ba902b7")

    def test_event_id_uniqueness(self):
        e1 = create_event(event_type="test", source="synthetic")
        e2 = create_event(event_type="test", source="synthetic")
        self.assertNotEqual(e1.event_id, e2.event_id)
        self.assertTrue(uuid.UUID(e1.event_id))
        self.assertTrue(uuid.UUID(e2.event_id))

    def test_custom_event_id_preservation(self):
        custom_id = "custom-uuid-12345"
        event = create_event(
            event_type="test",
            source="synthetic",
            event_id=custom_id,
        )
        self.assertEqual(event.event_id, custom_id)
        self.assertEqual(Event.from_json(event.to_json()).event_id, custom_id)

    def test_invalid_event_id(self):
        with self.assertRaisesRegex(ValueError, "event_id"):
            Event(
                event_type="test",
                source="synthetic",
                timestamp_unix_ns=1,
                event_id="",
            )

    def test_device_sample_not_direct(self):
        with self.assertRaisesRegex(ValueError, "device-wide"):
            create_event(event_type="gpu.sample", source="nvidia_smi", attribution="direct",
                         request_id="req-1")

    def test_direct_needs_join_key(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            create_event(event_type="model.response", source="runtime", attribution="direct")

    def test_direct_attribution_with_span_id(self):
        event = create_event(
            event_type="model.response",
            source="runtime",
            attribution="direct",
            span_id="span-123",
        )
        self.assertEqual(event.attribution, "direct")
        self.assertEqual(event.span_id, "span-123")

    def test_invalid_string_identifiers(self):
        with self.assertRaisesRegex(ValueError, "span_id"):
            create_event(event_type="test", source="synthetic", span_id="")
        with self.assertRaisesRegex(ValueError, "trace_id"):
            create_event(event_type="test", source="synthetic", trace_id="")

    def test_fail_closed_on_non_numeric_metrics(self):
        with self.assertRaisesRegex(ValueError, "numeric"):
            create_event(event_type="model.response", source="runtime", metrics={"tokens": "42"})

    def test_unknown_version(self):
        with self.assertRaisesRegex(ValueError, "schema version"):
            Event(event_type="test", source="synthetic", timestamp_unix_ns=1,
                  schema_version="0.2.0")


if __name__ == "__main__":
    unittest.main()
