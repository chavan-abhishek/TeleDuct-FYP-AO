# Parallel agent workflow

1. Merge the foundation PR from `chore/00-foundation` to the existing default branch.
2. Create and publish `integration/v0.1` **from the updated default branch**.
3. Create feature branches from exactly the same integration base. Each has its own `WORKSTREAM.md`, source adapter, docs, and tests.
4. Each owner works in a separate `git worktree` and opens an individual PR targeting `integration/v0.1`.
5. Merge order: semantic + runtime; hardware; fusion; then dashboard/evaluation after the contract settles.
6. Finally PR `integration/v0.1` into default branch, after all CI checks and a human review.

Branch roster:
- `feat/semantic-traces` — agent spans, tool calls, token usage and privacy.
- `feat/inference-runtime` — request lifecycle + TTFT/TPOT/KV metrics adapters.
- `feat/nvidia-hardware` — true device-level NVML/nvidia-smi, opt-in CUPTI spike.
- `feat/correlation-storage` — explicit ID joins, ambiguity, JSONL/Postgres plan.
- `feat/dashboard-evaluation` — only after prior branches land; dashboard and experiment comparisons.

Each PR must report: changed contract fields, representative recorded sample (redacted), test output, hardware matrix, known missing capabilities, and observable overhead where applicable.

No long-lived branches per person and no separate incompatible shared schemas per branch. Workstream branches are short-lived feature branches, not permanent forks.
