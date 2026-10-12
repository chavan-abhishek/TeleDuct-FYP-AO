# Workstream C — NVIDIA execution plane

**Branch**: `feat/nvidia-hardware`  
**Owner**: NVIDIA/CUDA systems engineer.  
**Dependency**: foundation event contract.

## Goal
Collect truthful GPU device samples, then prototype opt-in CUPTI API→kernel activity correlation as a separate bounded spike.

## Already scaffolded
- `src/teleduct/hardware/nvidia_smi.py`: low-dependency `nvidia-smi` sampler (device-scoped).
- `tests/test_hardware.py`: parser and mocked subprocess tests.

## Build tasks
1. On Nobara: record `nvidia-smi`, driver, toolkit availability (`nvcc` optional), GPU architecture, CUPTI headers/lib support. Do not change OS driver in this branch.
2. Implement bounded low-rate (e.g. 1Hz) sampling with source timestamps and missing-value/error states; optional NVML replacement later.
3. Add opt-in CUPTI demo against one known CUDA process: runtime/driver API calls → CUDA kernel and memcpy activity by CUPTI correlation ID.
4. Benchmark cost of the collector and record dropped CUPTI buffer events. Preserve source-specific clock semantics.
5. Document *separately* how an inference request maps to CUDA launch context. With batching, mapping can be many-to-many; mark unknown until demonstrated.

## Acceptance
- Sampler works on tested NVIDIA host and cleanly reports `unavailable` without one.
- Samples remain `shared_device`; no direct per-request VRAM, joules or compute claims.
- CUPTI proof is optional but must be measured and disabled by default.

## Owner boundaries
Only hardware adapter, tests, hardware workstream docs, WORKSTREAM.md. No privileged kernel probes, high-frequency kernel interception, or driver changes in v0.1.
