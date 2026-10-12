# Research and validation gates

**Hypothesis H1: correlation** — quantify correctly matched, ambiguous, and unmatched inference requests against injected ground-truth IDs.
**H2: overhead** — compare instrumentation off, semantic only, runtime+semantic, and opt-in CUPTI on TTFT, TPOT, throughput, memory and CPU.
**H3: diagnostic gain** — compare app-only traces against cross-layer traces on labeled scenarios (queue pressure, KV pressure, oversized context, tool timeout).

Method: repeatable prompts and fixed engine/model versions, warmup, >=30 measured trials per condition after pilot sizing; report distribution, CI and confounders. Do not pretend overlapping device power provides exact per-request Joules.

Feasibility gates:
- F0: stdlib contracts and offline synthetic fixtures pass CI.
- F1: real agent span joins a real inference request by stable ID.
- F2: GPU sampler works and is labeled `shared_device`.
- F3: opt-in CUDA activity capture and demonstrable API↔kernel correlation.
- F4: **separate proof** of request↔CUDA attribution with concurrency/batching; if it fails, expose shared-batch/ambiguous rather than overclaiming.

Current research claims are *questions*, not reported results.
