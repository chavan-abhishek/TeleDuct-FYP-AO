# Contributing

- Use the dedicated feature worktree and read `WORKSTREAM.md`.
- Feature branches target `psw`; PRs target `psw` for all feature development. Never target `main`.
- Do not add API keys, real patient/finance/customer prompts or private records.
- Run `PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py' -v`.
- PR includes reproducible steps, limitations and screenshots/traces only if nonsecret and measured.
- Sync via PR; avoid editing someone else's workstream interface without contract review.
