import unittest
from teleduct.hardware.nvidia_smi import parse_row, sample_device, get_device_info
from unittest.mock import patch
from types import SimpleNamespace
import subprocess


class HardwareTests(unittest.TestCase):
    def test_parse(self):
        data = parse_row("0, 44, 2086, 62.5, 68\n")
        self.assertEqual(data["gpu_index"], 0)
        self.assertEqual(data["metrics"]["vram_used_mib"], 2086.0)

    def test_missing_metric_not_zero(self):
        self.assertNotIn("power_w", parse_row("0, 44, 2086, N/A, 68")["metrics"])

    def test_malformed_rows_raise(self):
        with self.assertRaisesRegex(ValueError, "exactly one GPU row"):
            parse_row("0, 44\n1, 45\n")
        with self.assertRaisesRegex(ValueError, "unexpected nvidia-smi output shape"):
            parse_row("0, 44, 100\n")

    def test_invalid_gpu_index(self):
        with self.assertRaisesRegex(ValueError, "invalid GPU index"):
            sample_device(-1)
        with self.assertRaisesRegex(ValueError, "invalid GPU index"):
            get_device_info(-1)

    def test_unsupported_source_raises(self):
        with self.assertRaisesRegex(ValueError, "unsupported GPU telemetry source"):
            sample_device(0, source="unknown")

    @patch("teleduct.hardware.nvidia_smi.subprocess.run")
    def test_sample_has_shared_scope(self, fake):
        fake.return_value = SimpleNamespace(stdout="0, 44, 2086, 62.5, 68\n")
        event = sample_device(0)
        self.assertEqual(event.attribution, "shared_device")
        self.assertIsNone(event.request_id)
        fake.assert_called_once()

    @patch("teleduct.hardware.nvidia_smi.subprocess.run")
    def test_subprocess_failure_raises_runtime_error(self, fake):
        fake.side_effect = subprocess.CalledProcessError(1, ["nvidia-smi"], stderr="NVIDIA driver error")
        with self.assertRaisesRegex(RuntimeError, "failed to query nvidia-smi"):
            sample_device(0)


if __name__ == "__main__":
    unittest.main()
