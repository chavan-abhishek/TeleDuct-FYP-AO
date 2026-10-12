# TeleDuct threat model — Stage 0

Data hazards: PII in prompts/tool args, credentials in environment or logs, model outputs with
sensitive information, system-level process metadata, unexpected tracing privilege needs.

Defaults:
- telemetry disabled unless enabled explicitly;
- raw prompt/tool args and model outputs not stored;
- local-only collector interface until authentication is designed;
- avoid root, privileged containers, eBPF and process memory reads in v0.1;
- size limits and retention policy before storage productionization;
- documentation/fixture content must contain only synthetic or redacted values.

Threat model and permissions must be revisited before enabling CUPTI collection on a multi-tenant
or remotely operated GPU host.
