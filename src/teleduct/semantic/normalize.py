"""Normalize application step boundaries into TeleDuct semantic events."""
from __future__ import annotations
import time
import uuid
from typing import Any, Callable
from teleduct.contracts import Event, create_event


def agent_run_event(
    *,
    run_id: str,
    parent_goal: str,
    agent_type: str = "compliance",
    trace_id: str | None = None,
) -> Event:
    """Session/run boundary initialization event."""
    if not run_id or not parent_goal:
        raise ValueError("run_id and parent_goal required")
    return create_event(
        event_type="agent.run",
        source="semantic",
        attribution="direct",
        run_id=run_id,
        step_id="init",
        trace_id=trace_id,
        attributes={"parent_goal": parent_goal, "agent_type": agent_type},
    )


def agent_step_event(
    *,
    run_id: str,
    step_id: str,
    node_name: str,
    request_id: str | None = None,
    trace_id: str | None = None,
    span_id: str | None = None,
    parent_span_id: str | None = None,
    tool_name: str | None = None,
    selected_action: str | None = None,
    status: str = "success",
    error_message: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    latency_ms: float | None = None,
) -> Event:
    """Discrete reasoning/action node event."""
    if not run_id or not step_id or not node_name:
        raise ValueError("run_id, step_id and node_name required")
    metrics: dict[str, float | int] = {}
    for key, value in (
        ("input_tokens", input_tokens),
        ("output_tokens", output_tokens),
        ("latency_ms", latency_ms),
    ):
        if value is not None:
            if value < 0:
                raise ValueError("metrics must be nonnegative")
            metrics[key] = value

    attributes: dict[str, Any] = {"node_name": node_name, "status": status}
    if tool_name:
        attributes["tool_name"] = tool_name
    if selected_action:
        attributes["selected_action"] = selected_action
    if error_message:
        attributes["error_message"] = str(error_message)[:200]

    return create_event(
        event_type="agent.step",
        source="semantic",
        attribution="direct",
        run_id=run_id,
        step_id=step_id,
        request_id=request_id,
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        attributes=attributes,
        metrics=metrics,
    )


class SemanticStepTracker:
    """Lightweight step wrapper preserving hierarchical trace identifiers."""

    def __init__(self, run_id: str, trace_id: str | None = None) -> None:
        self.run_id = run_id
        self.trace_id = trace_id or uuid.uuid4().hex
        self.step_counter = 0

    def trace_step(
        self,
        node_name: str,
        fn: Callable[[dict[str, Any]], dict[str, Any]],
        state: dict[str, Any],
        *,
        tool_name: str | None = None,
        request_id: str | None = None,
    ) -> tuple[dict[str, Any], Event]:
        step_id = f"step_{self.step_counter:03d}_{node_name}"
        span_id = uuid.uuid4().hex[:16]
        self.step_counter += 1
        t0 = time.perf_counter()
        try:
            result = fn(state)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            event = agent_step_event(
                run_id=self.run_id,
                step_id=step_id,
                node_name=node_name,
                request_id=request_id or result.get("request_id"),
                trace_id=self.trace_id,
                span_id=span_id,
                tool_name=tool_name or result.get("tool_name"),
                selected_action=result.get("selected_action"),
                status="success",
                latency_ms=round(latency_ms, 2),
            )
            return result, event
        except Exception as ex:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            event = agent_step_event(
                run_id=self.run_id,
                step_id=step_id,
                node_name=node_name,
                request_id=request_id,
                trace_id=self.trace_id,
                span_id=span_id,
                tool_name=tool_name,
                status="error",
                error_message=str(ex),
                latency_ms=round(latency_ms, 2),
            )
            raise ex from None
