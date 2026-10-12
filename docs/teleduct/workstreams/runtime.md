# Workstream B — Inference runtime instrumentation

**Branch**: `feat/inference-runtime`  
**Owner**: ML inference engineer (Codex / AGY).  
**Dependency**: foundation contract. No semantic branch imports.

## Goal
Extract per-request serving metrics from one tested SGLang (or vLLM) release and normalize reliably.

## Already scaffolded
- `src/teleduct/runtime/normalize.py` → accepts actually measured metrics only.
- `tests/test_runtime.py` → offline tests.

## Build tasks
1. Record an exact version compatibility matrix: Python, PyTorch, CUDA driver/toolkit, chosen runtime, GPU, model, precision.
2. On RTX 3060 Mobile 6GB, use a small supported model (verify footprint before selecting). No Qwen2.5-7B BF16 locally.
3. Start inference server as a separate process/container, not inside a tracing Python process. Disable deep profile by default.
4. Extract native request ID, input/output tokens, TTFT, TPOT and queue time **only when that runtime actually exposes the per-request value**. Aggregate metrics must be labeled aggregate.
5. Record KV cache state at actual available scope; do not infer `cudaMalloc` from cache misses.
6. Prove request IDs survive concurrency; measure missing IDs rather than joining by time alone.

## Acceptance
- Real local inference request → correctly normalized event with request ID.
- Missing unsupported fields absent, not zero or guesses.
- 10 concurrent requests preserve unique join keys; provide machine-readable redacted fixture and reproducible commands.
- Baseline latency and instrumented latency measured.

## Owner boundaries
Only runtime adapter, runtime tests, runtime workstream doc and WORKSTREAM.md. No global scheduler monkeypatch until data proves necessary.
