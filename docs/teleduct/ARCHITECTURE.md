# TeleDuct architecture v0.1

**Goal**: Trace an agent step to a model request and, when attribution evidence permits, to its inference and CUDA activities.

```
SEMANTIC PLANE: LangGraph, tool/model spans, tokens, recorded decision context
            | trace_id, run_id, step_id, request_id
CORRELATION BACKBONE: versioned events + explicit attribution provenance
            | joins + ambiguity tracking
EXECUTION PLANE: engine request lifecycle + NVML device samples + optional CUPTI
            |
STORAGE AND UI: initial JSONL fixture -> Postgres -> dashboard
```

Two observation layers, one shared correlation backbone. The UI is a consumer, not a separate observer.

## Signal classes
1. `direct`: a reliable explicit request or agent-step key was propagated.
2. `time_window_estimate`: temporal association only, with defined uncertainty.
3. `shared_device`: whole-device sample; cannot claim per-request energy or VRAM.
4. `unattributed`: signal exists, no defensible join.

## Non-goals for v0.1
MoE internals, eBPF, request-level energy accounting under concurrency, full compliance, root-cause guarantees, distributed clocks.

## External constraints
- Nobara Linux 44, 16GB system RAM, RTX 3060 Mobile 6GB. Use small models and opt-in profiling.
- Pinned SGLang/vLLM/CUDA versions must be proven in a compatibility matrix before production adapters.
- Under dynamic batching, one CUDA kernel may serve many requests. Do not assign it entirely to one request.
