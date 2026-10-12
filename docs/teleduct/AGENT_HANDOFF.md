# Ready-to-paste task for Codex / AGY / CLI agents

Work only on your current feature branch, **never main**. Read `AGENTS.md`, `WORKSTREAM.md`,
`docs/teleduct/CONTRACT.md` and `docs/teleduct/ARCHITECTURE.md` before editing.

Task: deliver exactly the goals and acceptance criteria in `WORKSTREAM.md` using the smallest
working vertical implementation. Do not edit another workstream's files or widen the event
schema without an explicit contract-change proposal. First inspect current code, report an
implementation plan, then implement code with offline tests and, where applicable, opt-in
hardware-dependent integration tests. Preserve original repo behavior. Disable collection by
default and never log raw prompts, credentials or sensitive tool arguments. Never present shared
GPU samples as direct per-request GPU accounting. Missing data must be absent/unavailable.

Before requesting review, run:

```
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 -m compileall -q src
git diff --check
```

Deliver in your final report:
1. Changed files + rationale.
2. How to run an exact reproducible sample.
3. Test results (observed, not projected).
4. Sample redacted event with correct attribution scope.
5. Limitations, clock/correlation ambiguity and next milestone.
6. Suggested PR title/body targeting `psw`.

Do not push, force-push, merge, change GitHub branch protections or alter GPU drivers without
explicit human instruction.
