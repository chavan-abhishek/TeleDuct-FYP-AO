"""Versioned shared event contract. Python stdlib only; hardware-independent.

Never confuse temporal overlap with causal/per-request attribution.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import json
import time
from typing import Any
import uuid

SCHEMA_VERSION = "0.1.0"
ATTRIBUTION_LEVELS = frozenset({"direct", "time_window_estimate", "shared_device", "unattributed"})
SOURCES = frozenset({"semantic", "runtime", "nvidia_smi", "nvml", "cupti", "synthetic"})


@dataclass(frozen=True)
class Event:
    event_type: str
    source: str
    timestamp_unix_ns: int
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    attribution: str = "unattributed"
    trace_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None
    run_id: str | None = None
    step_id: str | None = None
    request_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float | int] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema version: {self.schema_version}")
        if not self.event_id or not isinstance(self.event_id, str) or not self.event_id.strip():
            raise ValueError("event_id must be a non-empty string")
        if not self.event_type or not self.event_type.strip():
            raise ValueError("event_type cannot be blank")
        if self.source not in SOURCES:
            raise ValueError(f"unknown source: {self.source}")
        if self.timestamp_unix_ns <= 0:
            raise ValueError("timestamp_unix_ns must be positive UTC epoch nanoseconds")
        if self.attribution not in ATTRIBUTION_LEVELS:
            raise ValueError(f"unknown attribution: {self.attribution}")
        if self.source in {"nvml", "nvidia_smi"} and self.attribution == "direct":
            raise ValueError("device-wide GPU samples cannot claim per-request direct attribution")
        if self.attribution == "direct" and not (self.request_id or (self.run_id and self.step_id) or self.span_id):
            raise ValueError("direct attribution requires request_id, span_id, or run_id+step_id")
        for field_name in ("trace_id", "span_id", "parent_span_id", "run_id", "step_id", "request_id"):
            val = getattr(self, field_name)
            if val is not None and (not isinstance(val, str) or not val.strip()):
                raise ValueError(f"{field_name} must be a non-empty string when provided")
        if not isinstance(self.attributes, dict) or not isinstance(self.metrics, dict):
            raise ValueError("attributes and metrics must be dicts")
        for name, value in self.metrics.items():
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise ValueError(f"metric {name} must be numeric and not boolean")
        # Fail closed on accidental non-JSON data; runtime-specific payloads must be normalized.
        json.dumps(asdict(self), allow_nan=False)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, payload: str) -> "Event":
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("event must be an object")
        return cls(**data)


def create_event(
    *,
    event_type: str,
    source: str,
    attribution: str = "unattributed",
    event_id: str | None = None,
    **kwargs: Any,
) -> Event:
    eid = event_id if event_id is not None else str(uuid.uuid4())
    return Event(
        event_type=event_type,
        source=source,
        timestamp_unix_ns=time.time_ns(),
        event_id=eid,
        attribution=attribution,
        **kwargs,
    )
