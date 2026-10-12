import time
import unittest
from unittest.mock import patch, MagicMock
from teleduct.contracts import create_event
from teleduct.hardware.nvidia_smi import (
    parse_row,
    sample_device,
    get_device_info,
    DeviceTelemetryWindow,
)
from teleduct.hardware.nvml import (
    NVMLAdapter,
    sample_device_nvml,
)


class HardwareTests(unittest.TestCase):
    def test_parse_row_clean(self):
        text = "0, 45, 12000, 180.5, 62\n"
        out = parse_row(text)
        self.assertEqual(out["gpu_index"], 0)
        self.assertEqual(out["metrics"]["gpu_util_pct"], 45.0)
        self.assertEqual(out["metrics"]["vram_used_mib"], 12000.0)
        self.assertEqual(out["metrics"]["power_w"], 180.5)
        self.assertEqual(out["metrics"]["gpu_temp_c"], 62.0)

    def test_parse_row_with_unsupported_metrics(self):
        text = "0, 0, 2048, [Not Supported], 50\n"
        out = parse_row(text)
        self.assertEqual(out["gpu_index"], 0)
        self.assertEqual(out["metrics"]["gpu_util_pct"], 0.0)
        self.assertEqual(out["metrics"]["vram_used_mib"], 2048.0)
        self.assertNotIn("power_w", out["metrics"])
        self.assertEqual(out["metrics"]["gpu_temp_c"], 50.0)

    def test_sample_device_source_is_strictly_nvidia_smi(self):
        with patch("subprocess.run") as mock_run:
            mock_res = MagicMock()
            mock_res.stdout = "0, 10, 4000, 50.0, 55\n"
            mock_run.return_value = mock_res

            ev = sample_device(gpu_index=0)
            self.assertEqual(ev.source, "nvidia_smi")
            self.assertEqual(ev.attribution, "shared_device")
            self.assertEqual(ev.event_type, "gpu.device_sample")

    def test_nvml_adapter_source_is_strictly_nvml(self):
        # Test NVML either natively if available or mocked
        try:
            ev = sample_device_nvml(gpu_index=0)
            self.assertEqual(ev.source, "nvml")
            self.assertEqual(ev.attribution, "shared_device")
            self.assertEqual(ev.event_type, "gpu.device_sample")
            self.assertIn("vram_used_mib", ev.metrics)
        except Exception:
            # Fallback mock verification if NVML library isn't available
            with patch("ctypes.CDLL") as mock_cdll:
                mock_lib = MagicMock()
                mock_lib.nvmlInit_v2.return_value = 0
                mock_lib.nvmlDeviceGetHandleByIndex_v2.return_value = 0
                mock_lib.nvmlDeviceGetUtilizationRates.return_value = 0
                mock_lib.nvmlDeviceGetMemoryInfo.return_value = 0
                mock_lib.nvmlDeviceGetPowerUsage.return_value = 0
                mock_lib.nvmlDeviceGetTemperature.return_value = 0
                mock_cdll.return_value = mock_lib

                adapter = NVMLAdapter()
                adapter.initialize()
                ev = adapter.sample_device(0)
                self.assertEqual(ev.source, "nvml")
                self.assertEqual(ev.attribution, "shared_device")

    def test_device_telemetry_window_time_aligned_samples(self):
        # Simulate active inference lasting ~100ms with mock/live sampler fallback
        mock_sample = lambda: create_event(
            event_type="gpu.device_sample",
            source="nvidia_smi",
            attribution="shared_device",
            metrics={"gpu_util_pct": 20.0, "power_w": 35.0, "vram_used_mib": 1024.0},
        )
        sink = []
        with DeviceTelemetryWindow(
            gpu_index=0,
            sample_interval_s=0.03,
            sampler_fn=mock_sample,
            event_sink=sink,
        ) as window:
            time.sleep(0.10)

        # Must have collected at least 2 time-aligned samples during window
        self.assertGreaterEqual(len(window.samples), 2)
        self.assertEqual(len(sink), len(window.samples))

        for sample in window.samples:
            self.assertEqual(sample.attribution, "shared_device")
            self.assertEqual(sample.event_type, "gpu.device_sample")

        summary = window.get_window_summary_event()
        self.assertEqual(summary.attribution, "shared_device")
        self.assertIn("observation_window", summary.attributes)
        obs = summary.attributes["observation_window"]
        self.assertIsNotNone(obs["start"])
        self.assertIsNotNone(obs["end"])
        self.assertGreater(obs["duration_ms"], 50.0)

    def test_parse_row_invalid_shape_raises(self):
        with self.assertRaises(ValueError):
            parse_row("0, 10, 20\n")

    def test_sample_device_negative_index_raises(self):
        with self.assertRaises(ValueError):
            sample_device(gpu_index=-1)

    def test_get_device_info_failure_graceful(self):
        with patch("subprocess.run", side_effect=RuntimeError("no binary")):
            info = get_device_info(0)
            self.assertEqual(info["gpu_name"], "unavailable")


if __name__ == "__main__":
    unittest.main()
