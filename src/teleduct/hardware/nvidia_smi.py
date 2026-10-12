"""Low-dependency NVIDIA device sampler for TeleDuct.

This is a *shared-device* snapshot, NEVER per-request usage.
Production collector may use NVML; nvidia-smi serves as universal fallback.
"""
from __future__ import annotations
import datetime
import subprocess
import threading
import time
from typing import Any, Callable, Sequence
from teleduct.contracts import Event, create_event

FIELDS = ("index", "utilization.gpu", "memory.used", "power.draw", "temperature.gpu")


def _metric(value: str) -> float | None:
    raw = value.strip()
    if not raw or raw.upper() in {"N/A", "[NOT SUPPORTED]", "NOT SUPPORTED", "[N/A]"}:
        return None
    return float(raw)


def parse_row(text: str) -> dict:
    rows = [r.strip() for r in text.splitlines() if r.strip()]
    if len(rows) != 1:
        raise ValueError("expected exactly one GPU row; explicitly select GPU")
    pieces = [p.strip() for p in rows[0].split(",")]
    if len(pieces) != len(FIELDS):
        raise ValueError("unexpected nvidia-smi output shape")
    device_index = int(pieces[0])
    metric_names = ("gpu_util_pct", "vram_used_mib", "power_w", "gpu_temp_c")
    metrics = {
        k: v
        for k, raw in zip(metric_names, pieces[1:])
        if (v := _metric(raw)) is not None
    }
    return {"gpu_index": device_index, "metrics": metrics}


def sample_device(gpu_index: int = 0) -> Event:
    """Sample NVIDIA hardware state at shared-device scope via nvidia-smi.
    
    Source is strictly 'nvidia_smi'.
    """
    if gpu_index < 0:
        raise ValueError("invalid GPU index")
    cmd = [
        "nvidia-smi",
        f"--query-gpu={','.join(FIELDS)}",
        "--format=csv,noheader,nounits",
        "-i",
        str(gpu_index),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=5
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as ex:
        raise RuntimeError(f"failed to query nvidia-smi on device {gpu_index}: {ex}") from ex

    data = parse_row(result.stdout)
    return create_event(
        event_type="gpu.device_sample",
        source="nvidia_smi",
        attribution="shared_device",
        attributes={"gpu_index": data["gpu_index"]},
        metrics=data["metrics"],
    )


def get_device_info(gpu_index: int = 0) -> dict[str, str]:
    """Retrieve static device information (name, driver version)."""
    if gpu_index < 0:
        raise ValueError("invalid GPU index")
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,driver_version",
        "--format=csv,noheader",
        "-i",
        str(gpu_index),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=5
        )
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        return {
            "gpu_name": parts[0] if len(parts) > 0 else "unknown",
            "driver_version": parts[1] if len(parts) > 1 else "unknown",
        }
    except Exception:
        return {"gpu_name": "unavailable", "driver_version": "unavailable"}


class DeviceTelemetryWindow:
    """Context manager providing time-aligned hardware samples during inference or step execution."""

    def __init__(
        self,
        gpu_index: int = 0,
        sample_interval_s: float = 0.1,
        sampler_fn: Callable[[], Event] | None = None,
        event_sink: Callable[[Event], None] | list[Event] | None = None,
    ) -> None:
        self.gpu_index = gpu_index
        self.sample_interval_s = max(0.01, sample_interval_s)
        self.sampler_fn = sampler_fn or (lambda: sample_device(self.gpu_index))
        self.event_sink = event_sink
        self.samples: list[Event] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.start_time_iso: str | None = None
        self.end_time_iso: str | None = None
        self.duration_ms: float = 0.0

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                ev = self.sampler_fn()
                self.samples.append(ev)
                if self.event_sink is not None:
                    if isinstance(self.event_sink, list):
                        self.event_sink.append(ev)
                    elif callable(self.event_sink):
                        self.event_sink(ev)
            except Exception:
                pass
            self._stop_event.wait(self.sample_interval_s)

    def __enter__(self) -> DeviceTelemetryWindow:
        self.start_time_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self._t0 = time.perf_counter()
        # Take initial sample immediately
        try:
            init_ev = self.sampler_fn()
            self.samples.append(init_ev)
            if self.event_sink is not None:
                if isinstance(self.event_sink, list):
                    self.event_sink.append(init_ev)
                elif callable(self.event_sink):
                    self.event_sink(init_ev)
        except Exception:
            pass
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self.duration_ms = (time.perf_counter() - self._t0) * 1000.0
        self.end_time_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    def get_window_summary_event(self) -> Event:
        """Construct an aggregate event summarizing hardware metrics over the observation window."""
        metrics: dict[str, float | int] = {
            "sample_count": len(self.samples),
            "window_duration_ms": round(self.duration_ms, 2),
        }
        if self.samples:
            util_vals = [s.metrics["gpu_util_pct"] for s in self.samples if "gpu_util_pct" in s.metrics]
            if util_vals:
                metrics["mean_gpu_util_pct"] = round(sum(util_vals) / len(util_vals), 2)
                metrics["peak_gpu_util_pct"] = round(max(util_vals), 2)

            power_vals = [s.metrics["power_w"] for s in self.samples if "power_w" in s.metrics]
            if power_vals:
                metrics["mean_power_w"] = round(sum(power_vals) / len(power_vals), 2)
                metrics["peak_power_w"] = round(max(power_vals), 2)

            vram_vals = [s.metrics["vram_used_mib"] for s in self.samples if "vram_used_mib" in s.metrics]
            if vram_vals:
                metrics["peak_vram_used_mib"] = round(max(vram_vals), 1)

        source = self.samples[0].source if self.samples else "nvidia_smi"
        return create_event(
            event_type="gpu.device_sample",
            source=source,
            attribution="shared_device",
            attributes={
                "gpu_index": self.gpu_index,
                "observation_window": {
                    "start": self.start_time_iso,
                    "end": self.end_time_iso,
                    "duration_ms": round(self.duration_ms, 2),
                },
            },
            metrics=metrics,
        )
