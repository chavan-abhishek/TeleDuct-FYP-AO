"""Direct native NVML adapter for TeleDuct using libnvidia-ml.so.1."""
from __future__ import annotations
import ctypes
import os
import sys
from typing import Any
from teleduct.contracts import Event, create_event


class _NvmlMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class _NvmlUtilization(ctypes.Structure):
    _fields_ = [
        ("gpu", ctypes.c_uint),
        ("memory", ctypes.c_uint),
    ]


class NVMLAdapter:
    """Zero-dependency NVML client using ctypes."""

    def __init__(self, lib_path: str = "libnvidia-ml.so.1") -> None:
        self.lib_path = lib_path
        self._lib = None
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return
        try:
            self._lib = ctypes.CDLL(self.lib_path)
        except OSError as ex:
            raise RuntimeError(f"Failed to load NVML library '{self.lib_path}': {ex}") from ex

        res = self._lib.nvmlInit_v2()
        if res != 0:
            raise RuntimeError(f"nvmlInit_v2 returned error code {res}")
        self._initialized = True

    def shutdown(self) -> None:
        if self._initialized and self._lib:
            try:
                self._lib.nvmlShutdown()
            finally:
                self._initialized = False

    def sample_device(self, gpu_index: int = 0) -> Event:
        """Sample NVIDIA GPU directly using NVML (source='nvml')."""
        if not self._initialized:
            self.initialize()
        assert self._lib is not None

        handle = ctypes.c_void_p()
        res = self._lib.nvmlDeviceGetHandleByIndex_v2(gpu_index, ctypes.byref(handle))
        if res != 0:
            raise RuntimeError(f"Failed to get NVML handle for device {gpu_index} (code {res})")

        metrics: dict[str, float | int] = {}

        # Utilization
        util = _NvmlUtilization()
        if self._lib.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(util)) == 0:
            metrics["gpu_util_pct"] = float(util.gpu)

        # Memory
        mem = _NvmlMemory()
        if self._lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem)) == 0:
            metrics["vram_used_mib"] = round(float(mem.used) / (1024 * 1024), 1)

        # Power (milliwatts -> watts)
        power_mw = ctypes.c_uint()
        if self._lib.nvmlDeviceGetPowerUsage(handle, ctypes.byref(power_mw)) == 0:
            metrics["power_w"] = round(float(power_mw.value) / 1000.0, 2)

        # Temperature
        temp_c = ctypes.c_uint()
        if self._lib.nvmlDeviceGetTemperature(handle, 0, ctypes.byref(temp_c)) == 0:
            metrics["gpu_temp_c"] = float(temp_c.value)

        return create_event(
            event_type="gpu.device_sample",
            source="nvml",
            attribution="shared_device",
            attributes={"gpu_index": gpu_index},
            metrics=metrics,
        )


_global_nvml: NVMLAdapter | None = None


def sample_device_nvml(gpu_index: int = 0) -> Event:
    """Convenience function to sample GPU via NVML."""
    global _global_nvml
    if _global_nvml is None:
        _global_nvml = NVMLAdapter()
    return _global_nvml.sample_device(gpu_index)
