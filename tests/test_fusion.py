import unittest
from teleduct.contracts import create_event
from teleduct.fusion.join import correlate_requests


class FusionTests(unittest.TestCase):
    def test_direct_join_and_shared_device_separation(self):
        s = create_event(event_type="agent.step", source="semantic", attribution="direct",
                         run_id="r", step_id="s", request_id="req")
        r = create_event(event_type="inference.response", source="runtime", attribution="direct",
                         request_id="req")
        g = create_event(event_type="gpu.device_sample", source="nvidia_smi",
                         attribution="shared_device")
        res = correlate_requests([s, r, g])
        self.assertEqual(len(res["matched"]), 1)
        self.assertEqual(len(res["shared_device"]), 1)
        self.assertEqual(len(res["unmatched"]), 0)

    def test_ambiguous_join_is_not_silent(self):
        a = create_event(event_type="agent.step", source="semantic", attribution="direct",
                         run_id="r", step_id="s1", request_id="req")
        b = create_event(event_type="agent.step", source="semantic", attribution="direct",
                         run_id="r", step_id="s2", request_id="req")
        c = create_event(event_type="inference.response", source="runtime", attribution="direct",
                         request_id="req")
        res = correlate_requests([a, b, c])
        self.assertEqual(len(res["matched"]), 0)
        self.assertEqual(len(res["ambiguous"]), 1)
