# Workstream A — Semantic agent tracing

**Branch**: `feat/semantic-traces`  
**Owner**: Agent/app engineer (teammate + Codex/AGY).  
**Dependency**: foundation `src/teleduct/contracts/events.py`.

## Goal
Instrument a small LangGraph agent and its model/tool calls, preserving trace/run/step/request identities. Record actual tool choices and token counts only when available, without raw prompts by default.

## Already scaffolded
- `src/teleduct/semantic/normalize.py` → dependency-free event normalization.
- `tests/test_semantic.py` → unit tests without internet/GPU.

## Build tasks
1. Add LangGraph callback/wrapper around 3-node toy agent (parse → plan → tool). Tests with stubbed model/tool.
2. Generate and propagate stable `trace_id`, `run_id`, `step_id` and the engine-visible `request_id` on each outbound model call; verify the runtime really echoes/preserves the latter.
3. Emit start/end/error spans for node, tool and model operations using OpenTelemetry GenAI conventions where they fit.
4. Populate actual reported token usage, never invent token counts, model cost or tool candidate probabilities.
5. Add privacy default and opt-in capture controls; no raw inputs or tool args in fixtures.

## Acceptance
- For at least 3 consecutive nodes the event stream preserves parent-child context and original request IDs.
- Exception paths emit an error event and don't alter tool behavior when telemetry is disabled.
- Unit tests and a deterministic mocked end-to-end LangGraph test pass.
- PR includes redacted JSONL and latency overhead relative to disabled instrumentation.

## Owner boundaries
Only `src/teleduct/semantic/**`, `tests/test_semantic*`, `docs/teleduct/workstreams/semantic.md`, and `WORKSTREAM.md`. Contract change requires coordinated PR.
