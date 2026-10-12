"""TeleDuct hardware telemetry adapter."""
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

__all__ = [
    "parse_row",
    "sample_device",
    "get_device_info",
    "DeviceTelemetryWindow",
    "NVMLAdapter",
    "sample_device_nvml",
]
