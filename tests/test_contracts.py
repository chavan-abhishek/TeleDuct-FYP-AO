import unittest
from teleduct.contracts import Event, create_event


class ContractTests(unittest.TestCase):
    def test_roundtrip(self):
        event = create_event(event_type="agent.step", source="semantic", attribution="direct",
                             run_id="run-1", step_id="step-1", metrics={"latency_ms": 7.2})
        self.assertEqual(Event.from_json(event.to_json()), event)

    def test_device_sample_not_direct(self):
        with self.assertRaisesRegex(ValueError, "device-wide"):
            create_event(event_type="gpu.sample", source="nvidia_smi", attribution="direct",
                         request_id="req-1")

    def test_direct_needs_join_key(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            create_event(event_type="model.response", source="runtime", attribution="direct")

    def test_fail_closed_on_non_numeric_metrics(self):
        with self.assertRaisesRegex(ValueError, "numeric"):
            create_event(event_type="model.response", source="runtime", metrics={"tokens": "42"})

    def test_unknown_version(self):
        with self.assertRaisesRegex(ValueError, "schema version"):
            Event(event_type="test", source="synthetic", timestamp_unix_ns=1,
                  schema_version="0.2.0")


if __name__ == "__main__":
    unittest.main()
