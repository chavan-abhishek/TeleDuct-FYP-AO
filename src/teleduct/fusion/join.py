"""Conservative explicit-ID join, durable JSONL persistence, and idempotent ingestion for TeleDuct.

Ambiguous shared request IDs remain ambiguous, not arbitrarily attributed.
Device snapshots are preserved as contextual telemetry strictly within documented observation intervals.
"""
from __future__ import annotations
import json
import os
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any
from teleduct.contracts import Event


class ConflictingEventError(Exception):
    """Raised when an ingested event shares an event_id with an existing event but has differing payload."""
    pass


class PersistenceError(Exception):
    """Raised when an event fails to persist durably to storage."""
    pass


def correlate_requests(events: list[Event]) -> dict[str, Any]:
    """Correlate semantic and runtime events by explicit request identifiers.
    
    Rules:
    - Resolves client_request_id and engine_native_id mapping.
    - Preserves distinct client_request_ids even if an engine emits duplicate native IDs.
    - Ambiguous duplicate request mappings across different steps are flagged as ambiguous.
    - Shared device samples are separated and never directly attributed to a single request.
    - Unmatched requests and responses are preserved with explicit categorization.
    """
    semantic = defaultdict(list)
    inference_by_client = defaultdict(list)
    inference_by_engine = defaultdict(list)
    shared: list[Event] = []
    unmatched_raw: list[Event] = []

    for event in events:
        if event.attribution == "shared_device":
            shared.append(event)
        elif event.source == "semantic" and event.request_id:
            semantic[event.request_id].append(event)
        elif event.source == "runtime":
            cid = event.attributes.get("client_request_id") or event.request_id
            nid = event.attributes.get("engine_native_id")
            if cid:
                inference_by_client[cid].append(event)
            elif nid:
                inference_by_engine[nid].append(event)
            else:
                unmatched_raw.append(event)
        else:
            unmatched_raw.append(event)

    matched = []
    ambiguous = []
    seen_runtime_event_ids = set()
    unmatched_semantic = []

    # Correlate primarily by client_request_id
    all_client_keys = sorted(set(semantic.keys()) | set(inference_by_client.keys()))

    for req_key in all_client_keys:
        ss = semantic.get(req_key, [])
        ii = [ev for ev in inference_by_client.get(req_key, []) if ev.event_id not in seen_runtime_event_ids]

        if not ss:
            continue

        distinct_steps = {(s.run_id, s.step_id) for s in ss}
        if len(distinct_steps) > 1:
            ambiguous.append({
                "request_id": req_key,
                "semantic": ss,
                "runtime": ii,
                "reason": "Reused request_id across multiple distinct steps",
            })
            for ev in ii:
                seen_runtime_event_ids.add(ev.event_id)
        elif ss and ii:
            matched.append({
                "request_id": req_key,
                "semantic": ss,
                "runtime": ii,
                "attribution": "direct",
            })
            for ev in ii:
                seen_runtime_event_ids.add(ev.event_id)
        else:
            unmatched_semantic.extend(ss)

    # Gather remaining runtime events that were never matched
    unmatched_runtime = [
        ev for ev_list in list(inference_by_client.values()) + list(inference_by_engine.values())
        for ev in ev_list if ev.event_id not in seen_runtime_event_ids
    ]

    dedup_unmatched: dict[str, Event] = {}
    for ev in unmatched_raw + unmatched_semantic + unmatched_runtime:
        dedup_unmatched[ev.event_id] = ev

    return {
        "matched": matched,
        "ambiguous": ambiguous,
        "shared_device": shared,
        "unmatched": list(dedup_unmatched.values()),
        "unmatched_requests": [e for e in unmatched_semantic if e.source == "semantic"],
        "unmatched_responses": [e for e in unmatched_runtime if e.event_type == "model.response"],
    }


class EventIngestionStore:
    """Telemetry store featuring thread-safe idempotent ingestion, conflict rejection, and durable JSONL persistence."""

    def __init__(self, storage_path: str | Path | None = None) -> None:
        self.storage_path = Path(storage_path) if storage_path else None
        self._events: dict[str, Event] = {}  # keyed by event_id
        self._lock = threading.Lock()
        self.corrupted_records_count = 0
        if self.storage_path and self.storage_path.exists():
            self._load_from_disk()

    def _load_from_disk(self) -> None:
        assert self.storage_path is not None
        with open(self.storage_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    event = Event.from_json(raw)
                    self._events[event.event_id] = event
                except Exception:
                    self.corrupted_records_count += 1

    def _persist_event(self, event: Event) -> None:
        if self.storage_path:
            try:
                self.storage_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.storage_path, "a", encoding="utf-8") as f:
                    f.write(event.to_json() + "\n")
                    f.flush()
                    os.fsync(f.fileno())
            except OSError as ex:
                raise PersistenceError(f"Failed to persist event {event.event_id} to {self.storage_path}: {ex}") from ex

    def ingest(self, event: Event) -> bool:
        """Idempotently ingest an event with thread-safety and fail-closed persistence.
        
        - If new: persists to JSONL first. Only commits to memory if persistence succeeds.
        - If duplicate with IDENTICAL payload: returns False (safe idempotent no-op).
        - If duplicate with CONFLICTING payload: raises ConflictingEventError.
        - If persistence fails: raises PersistenceError without modifying in-memory index.
        """
        with self._lock:
            if event.event_id in self._events:
                existing = self._events[event.event_id]
                if existing.to_json() == event.to_json():
                    return False  # Idempotent match
                raise ConflictingEventError(
                    f"Conflicting payload for existing event_id '{event.event_id}'"
                )

            # Persist to disk FIRST before mutating in-memory store
            self._persist_event(event)
            self._events[event.event_id] = event
            return True

    def ingest_batch(self, events: list[Event]) -> int:
        """Ingest a list of events idempotently, returning count of newly added events."""
        return sum(1 for e in events if self.ingest(e))

    def get_events(self) -> list[Event]:
        with self._lock:
            return list(self._events.values())

    def correlate(self) -> dict[str, Any]:
        """Correlate all ingested events with conservative join semantics."""
        return correlate_requests(self.get_events())

    def build_trace_slice(
        self,
        run_id: str,
        start_time_ns: int | None = None,
        end_time_ns: int | None = None,
    ) -> dict[str, Any]:
        """Build a unified trace slice for a specific run.
        
        Hardware contextual samples are strictly restricted to the documented observation interval
        of the target run using timestamp_unix_ns.
        """
        all_events = self.get_events()
        run_direct_events = [e for e in all_events if e.run_id == run_id]

        # Determine observation interval boundaries in nanoseconds
        timestamps = [e.timestamp_unix_ns for e in run_direct_events]
        t_start = start_time_ns or (min(timestamps) if timestamps else None)
        t_end = end_time_ns or (max(timestamps) if timestamps else None)

        # Filter shared_device samples to documented observation interval
        filtered_device_samples = []
        for e in all_events:
            if e.attribution == "shared_device":
                if t_start is not None and t_end is not None:
                    if t_start <= e.timestamp_unix_ns <= t_end:
                        filtered_device_samples.append(e)
                else:
                    filtered_device_samples.append(e)

        result = correlate_requests(run_direct_events)

        trace_steps = []
        for match in result["matched"]:
            trace_steps.append({
                "request_id": match["request_id"],
                "semantic_events": [e.to_json() for e in match["semantic"]],
                "runtime_events": [e.to_json() for e in match["runtime"]],
                "attribution": "direct",
            })

        # Preserve standalone semantic steps that had no outbound model call
        matched_req_ids = {m["request_id"] for m in result["matched"]}
        for ev in run_direct_events:
            if ev.event_type == "agent.step" and ev.request_id not in matched_req_ids:
                trace_steps.append({
                    "step_id": ev.step_id,
                    "request_id": ev.request_id,
                    "semantic_events": [ev.to_json()],
                    "runtime_events": [],
                    "attribution": "direct",
                })

        # Extract root run event if present
        root_run_events = [e for e in run_direct_events if e.event_type == "agent.run"]
        run_event_json = root_run_events[0].to_json() if root_run_events else None

        # Exclude root session initialization and pure local steps (no request_id) from unmatched count
        unmatched_for_run = [
            u for u in result["unmatched"]
            if u.run_id == run_id and u.event_type != "agent.run" and u.request_id is not None
        ]

        return {
            "run_id": run_id,
            "run_event": run_event_json,
            "observation_interval_ns": {
                "start": t_start,
                "end": t_end,
            },
            "trace_steps": trace_steps,
            "shared_device_context": [e.to_json() for e in filtered_device_samples],
            "ambiguous_count": len(result["ambiguous"]),
            "unmatched_count": len(unmatched_for_run),
        }
