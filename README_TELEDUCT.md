# TeleDuct v0.1 — Cross-layer agent observability (foundation)

Telemetry contract and offline tests only. This file intentionally does not replace an existing README.

## Quick check
`PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py' -v`

## Architecture
Read `docs/teleduct/ARCHITECTURE.md`, `CONTRACT.md`, `RESEARCH.md`, and `WORKFLOW.md`.

## Status
This foundation **does not yet instrument a live agent, inference server, or GPU**. Feature branches add adapters and measured validation. No claims of per-request CUDA causality or energy accounting.
