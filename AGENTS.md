# Agent instructions — TeleDuct

Scope: TeleDuct final-year research MVP, not a monitoring-product clone.

## Invariants
- Two planes: semantic and execution; joined through trace/run/step/request identifiers.
- Do not claim an NVML / nvidia-smi device-wide sample is exact GPU consumption by one request.
- CUPTI CUDA API ↔ kernel IDs are not automatically agent request ↔ kernel IDs. Prove the missing bridge.
- Do not present model output as genuine hidden reasoning; label decision context/provenance.
- Do not invent MoE outputs for dense models.
- No secrets or complete prompts/tool arguments in telemetry by default.
- Optional, fail-closed instrumentation; behavior unchanged when disabled.
- Do not add heavyweight inference dependencies to shared contracts.
- Preserve existing repo behavior and files; branch changes limited to assigned workstream.

## Local quality gate (no GPU)
`PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py' -v`
`python3 -m compileall -q src`

## Agent operating procedure
Read `WORKSTREAM.md` on the current feature branch, then `docs/teleduct/CONTRACT.md`.
Implement one vertical capability at a time with tests, explicit unavailable states, and a PR.
Do not merge or force-push automatically. No PR without acceptance evidence.
- Coordinate changes to README.md and root pyproject.toml in review; do not overwrite existing files.
