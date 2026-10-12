# Workstream D — Correlation and storage

**Branch**: `feat/correlation-storage`  
**Owner**: backend/data architect.  
**Dependency**: shared Event contract; fixture-driven until other branches land.

## Goal
Create loss-aware, ambiguity-aware joins between agent spans, inference requests and device samples, then an append-only storage path.

## Already scaffolded
- `src/teleduct/fusion/join.py`: conservative exact request-ID join.
- `tests/test_fusion.py`: unmatched/shared-device/ambiguous tests.

## Build tasks
1. Implement JSONL write/read first (stdlib; ensure append-only and error-safe writes); Postgres adapter only after schema settles.
2. Define the event identifier + dedupe/idempotency policy; sort by trustworthy timestamps without silently collapsing duplicate events.
3. Explicit 1:many inference requests/steps; flag conflicting request identities.
4. Keep shared device samples as contextual time-series, not exact cost allocation.
5. Add malformed/missing ID, out-of-order, concurrent and clock-skew fixtures.
6. Emit diagnostics: `matched`, `unmatched`, `ambiguous`, `shared_device` and coverage rates.

## Acceptance
- Roundtrip event persistence, privacy-safe records and duplicate handling demonstrated.
- Concurrency fixture: no false direct joins when one request ID maps to multiple semantic steps.
- No fabricated request-level GPU cost under device multiplexing.
- Storage migration documented before production Postgres integration.

## Owner boundaries
Only fusion/storage code, tests, docs, WORKSTREAM.md. Shared contract changes require integration review.
