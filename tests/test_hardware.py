import unittest
from teleduct.hardware.nvidia_smi import parse_row, sample_device
from unittest.mock import patch
from types import SimpleNamespace


class HardwareTests(unittest.TestCase):
    def test_parse(self):
        data = parse_row("0, 44, 2086, 62.5, 68\n")
        self.assertEqual(data["gpu_index"], 0)
        self.assertEqual(data["metrics"]["vram_used_mib"], 2086.0)

    def test_missing_metric_not_zero(self):
        self.assertNotIn("power_w", parse_row("0, 44, 2086, N/A, 68")["metrics"])

    @patch("teleduct.hardware.nvidia_smi.subprocess.run")
    def test_sample_has_shared_scope(self, fake):
        fake.return_value = SimpleNamespace(stdout="0, 44, 2086, 62.5, 68\n")
        event = sample_device(0)
        self.assertEqual(event.attribution, "shared_device")
        self.assertIsNone(event.request_id)
        fake.assert_called_once()
