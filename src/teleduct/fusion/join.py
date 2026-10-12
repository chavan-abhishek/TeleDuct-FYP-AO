"""Conservative explicit-ID join and idempotent ingestion for TeleDuct.

Ambiguous shared request IDs remain ambiguous, not arbitrarily attributed.
Device snapshots are preserved as contextual telemetry, NEVER per-request usage.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Any
from teleduct.contracts import Event


def correlate_requests(events: list[Event]) -> dict:
    semantic = defaultdict(list)
    inference = defaultdict(list)
    shared: list[Event] = []
    unmatched: list[Event] = []
    for event in events:
        if event.attribution == "shared_device":
            shared.append(event)
        elif event.source == "semantic" and event.request_id:
            semantic[event.request_id].append(event)
        elif event.source == "runtime" and event.request_id:
            inference[event.request_id].append(event)
        else:
            unmatched.append(event)

    matched = []
    ambiguous = []
    for request_id in sorted(set(semantic) | set(inference)):
        ss, ii = semantic.get(request_id, []), inference.get(request_id, [])
        distinct_steps = {(s.run_id, s.step_id) for s in ss}
        if len(distinct_steps) > 1:
            ambiguous.append({"request_id": request_id, "semantic": ss, "runtime": ii})
        elif ss and ii:
            matched.append({
                "request_id": request_id,
                "semantic": ss,
                "runtime": ii,
                "attribution": "direct",
            })
        else:
            unmatched.extend(ss + ii)
    return {
        "matched": matched,
        "ambiguous": ambiguous,
        "shared_device": shared,
        "unmatched": unmatched,
    }


class EventIngestionStore:
    """In-memory idempotent telemetry store and correlator."""

    def __init__(self) -> None:
        self._events: dict[str, Event] = {}  # keyed by event_id

    def ingest(self, event: Event) -> bool:
        """Idempotently ingest an event. Returns True if new, False if duplicate."""
        if event.event_id in self._events:
            return False
        self._events[event.event_id] = event
        return True

    def ingest_batch(self, events: list[Event]) -> int:
        """Ingest a list of events idempotently, returning count of new items."""
        return sum(1 for e in events if self.ingest(e))

    def get_events(self) -> list[Event]:
        return list(self._events.values())

    def correlate(self) -> dict[str, Any]:
        """Correlate all ingested events with idempotent semantics."""
        return correlate_requests(self.get_events())

    def build_trace_slice(self, run_id: str) -> dict[str, Any]:
        """Build a unified trace slice for a specific run."""
        run_events = [e for e in self.get_events() if e.run_id == run_id or e.attribution == "shared_device"]
        result = correlate_requests(run_events)
        
        # Link direct matches
        trace_steps = []
        for match in result["matched"]:
            trace_steps.append({
                "request_id": match["request_id"],
                "semantic_events": [e.to_json() for e in match["semantic"]],
                "runtime_events": [e.to_json() for e in match["runtime"]],
                "attribution": "direct",
            })

        return {
            "run_id": run_id,
            "trace_steps": trace_steps,
            "shared_device_context": [e.to_json() for e in result["shared_device"]],
            "ambiguous_count": len(result["ambiguous"]),
            "unmatched_count": len([u for u in result["unmatched"] if u.run_id == run_id]),
        }
