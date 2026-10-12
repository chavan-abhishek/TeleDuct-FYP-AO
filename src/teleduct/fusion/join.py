"""Conservative explicit-ID join. No inference by time overlap.

Ambiguous shared request IDs remain ambiguous, not arbitrarily attributed.
"""
from __future__ import annotations
from collections import defaultdict
from teleduct.contracts import Event


def correlate_requests(events: list[Event]) -> dict:
    semantic = defaultdict(list)
    inference = defaultdict(list)
    shared = []
    unmatched = []
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
            matched.append({"request_id": request_id, "semantic": ss, "runtime": ii,
                            "attribution": "direct"})
        else:
            unmatched.extend(ss + ii)
    return {"matched": matched, "ambiguous": ambiguous, "shared_device": shared,
            "unmatched": unmatched}
