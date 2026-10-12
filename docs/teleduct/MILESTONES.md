# Integration roadmap and hard acceptance gates

## F0: Contract first (foundation PR to `main`)
- Shared stdlib schema `src/teleduct/contracts/events.py` is available and tested.
- Contract version, event IDs, timestamp units, attribution provenance documented.
- Existing `Layer_A`, `layer_b`, `infra` are unchanged; offline tests pass.
- Human reviews contract and audit warnings before merging.

## F1: Semantic ↔ Inference
- One real LangGraph step corresponds to a real SGLang inference request.
- Explicit IDs (`run_id`, `step_id`, `request_id`; trace/span where available) are transported across process/host boundaries and observable at both sides.
- Absent logprobs remain absent; synthetic events have explicit `source=synthetic`.
- Save redacted event samples with IDs, timing and proof of join; avoid raw prompts by default.

## F2: Hardware ↔ Inference (first truthful join)
- NVML sample records are `shared_device`; sampling cadence and clock domain documented.
- Inference windows may overlap. A `time_window_estimate` is not `direct` per-request allocation.
- Negative test: concurrent requests must not receive the same GPU total as unique attributable GPU usage.

## F3: Deep CUDA profiling (optional gate)
- CUPTI opt-in, instrument small controlled batches; record profiler overhead.
- Show CUPTI API↔GPU kernel correlation and separately document unproven request↔kernel relationship.
- Do not force changes to server internals or claim sub-request attribution without ground truth.

## F4: Dashboard and reproducible evaluation
- Display source, scope, missingness and method for every metric.
- Compare baseline tracing with TeleDuct for diagnosis, latency, overhead, reproducibility.
- MoE and divergence studies only after model support, runtime availability and calibration data are validated.

## Merge procedure
Every feature branch derives from the integrated foundation through `psw`; PRs target `psw` and must include reproducible test logs, changed files, measured metrics with units, limitations and merge checklist. Never merge merely because synthetic fixtures look plausible. Never target or merge into `main` without explicit human authorization.
