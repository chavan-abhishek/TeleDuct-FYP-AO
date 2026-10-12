"""Low-dependency NVIDIA device sampler for Stage 0.

This is a *shared-device* snapshot, NEVER per-request usage.
Production collector may use NVML; CUPTI is a separate future adapter.
"""
from __future__ import annotations
import subprocess
from teleduct.contracts import Event, create_event

FIELDS = ("index", "utilization.gpu", "memory.used", "power.draw", "temperature.gpu")


def _metric(value: str) -> float | None:
    raw = value.strip()
    if not raw or raw.upper() in {"N/A", "[NOT SUPPORTED]", "NOT SUPPORTED"}:
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
    metrics = {k: v for k, raw in zip(metric_names, pieces[1:])
               if (v := _metric(raw)) is not None}
    return {"gpu_index": device_index, "metrics": metrics}


def sample_device(gpu_index: int = 0) -> Event:
    if gpu_index < 0:
        raise ValueError("invalid GPU index")
    cmd = ["nvidia-smi", f"--query-gpu={','.join(FIELDS)}", "--format=csv,noheader,nounits", "-i", str(gpu_index)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5)
    data = parse_row(result.stdout)
    return create_event(event_type="gpu.device_sample", source="nvidia_smi",
                        attribution="shared_device", attributes={"gpu_index": data["gpu_index"]},
                        metrics=data["metrics"])
