# Architectural decisions (ADR summary)

- ADR-001: Single versioned event contract in the foundation before parallel development.
- ADR-002: One experimental engine implementation first, adapter interfaces permit another later.
- ADR-003: Stage 0 hardware telemetry is shared-device; per-request GPU attribution is research-gated.
- ADR-004: CUPTI work is opt-in and separate from application correctness.
- ADR-005: JSONL fixtures initially; Postgres after normalized schema and volume are justified.
- ADR-006: Event sources never fabricate unavailable metrics, tool confidence or hidden reasoning.
- ADR-007: Dashboard follows proven traces, not the other way around.
