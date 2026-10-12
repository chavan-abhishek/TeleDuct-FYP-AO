# Shared event contract — version 0.1.0

Canonical implementation: `src/teleduct/contracts/events.py`.

Required fields: `schema_version`, `event_type`, `source`, `timestamp_unix_ns` (UTC wall clock), `event_id` (immutable unique UUID string for idempotent ingestion), `attribution`.
Optional join identifiers: `trace_id`, `span_id`, `parent_span_id`, `run_id`, `step_id`, `request_id`.
`metrics` holds scalar numeric metrics with unit in key (e.g. `ttft_ms`, `vram_used_mib`).
`attributes` holds JSON-compatible nonsecret, bounded labels. Never log raw prompt text by default.

Rules:
- Propagate `request_id` across agent↔engine boundaries explicitly (when runtime supports it).
- Device snapshots are `shared_device` even if only one agent appears active.
- For CUPTI use explicit CUDA correlation records, then **separately** establish an engine request→CUDA mapping. Inference scheduling/batching can make the mapping many-to-many.
- Missing telemetry = missing (`null`/absent), not zero and never synthesized.
- Monotonic clocks are preferred for local duration measures. This schema uses wall-clock UTC event timestamps; document clock conversion and skew before performing cross-process interval joins.
- Operational/financial `cost` for self-hosted inference must be a modeled estimate with a disclosed rate and allocation model; token counts are observed, token price is not inherent.

Example event:
```json
{"attributes":{"engine":"sglang"},"attribution":"direct","event_id":"550e8400-e29b-41d4-a716-446655440000","event_type":"model.response","metrics":{"ttft_ms":40.0},"parent_span_id":"00f067aa0ba902b7","request_id":"req-1","run_id":"run-1","schema_version":"0.1.0","source":"runtime","span_id":"b7ad6b7169203331","step_id":"step-2","timestamp_unix_ns":1730000000000000000,"trace_id":"abc"}
```
