# Workstream ownership (concurrent CLI agents)

| Branch | Suggested owner | Source scope | First acceptance gate |
|---|---|---|---|
| `feat/semantic-traces` | Codex 1 | `layer_b/sdk/`, `layer_b/demo_agent/`, `src/teleduct/semantic/` | Actual span → normalized semantic event; tool calls and retries recorded |
| `feat/inference-runtime` | AGY | `Layer_A/logit_processor.py`, `Layer_A/sglang_scraper.py`, `src/teleduct/runtime/` | Real request ID + observed tokens/TTFT; missing logprobs stay missing |
| `feat/nvidia-hardware` | Codex 2 | `Layer_A/gpu_poller.py`, `src/teleduct/hardware/` | NVML events labeled `shared_device`; opt-in isolated CUPTI research |
| `feat/correlation-storage` | AGY or teammate | `layer_b/storage/`, `layer_b/sdk/repository.py`, `src/teleduct/fusion/` | Join one real agent step and request with explicit identity; unmatched flagged |
| `feat/dashboard-evaluation` | After F1/F2 | `infra/local_hub/grafana/`, `src/teleduct/evaluation/` | Missing values accurately shown; diagnostic benchmark with controls |

**Scheduling:** AGY works on the foundation first; do not implement all five branches in parallel before merging contracts. Then run four agents in separate Git worktrees. Do not allow two agents to share a checkout or edit cross-owned files without coordination.

**Truthfulness:** Distinguish static code, mocked tests, live CPU, live SGLang, real NVML, and real CUPTI. Use exact observed versions and logged evidence in PRs. Never commit credentials or raw customer prompts.
