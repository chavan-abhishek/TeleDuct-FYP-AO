"""Normalize application step boundaries into TeleDuct semantic events."""
from __future__ import annotations
import hashlib
import time
import uuid
from typing import Any, Callable, Sequence
from teleduct.contracts import Event, create_event

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
except ImportError:
    class BaseCallbackHandler:  # type: ignore[no-redef]
        """Fallback callback handler base if langchain_core is not present."""
        pass


def agent_run_event(
    *,
    run_id: str,
    parent_goal: str | None = None,
    agent_type: str = "compliance",
    trace_id: str | None = None,
    span_id: str | None = None,
    capture_raw_goal: bool = False,
) -> Event:
    """Session/run boundary initialization event.
    
    By default, raw goal strings are NOT captured in plain text to maintain privacy.
    Instead, a truncated SHA-256 hash (parent_goal_hash) is recorded unless
    explicitly enabled via capture_raw_goal=True.
    """
    if not run_id:
        raise ValueError("run_id required")
    
    attributes: dict[str, Any] = {"agent_type": agent_type}
    if parent_goal is not None:
        if capture_raw_goal:
            attributes["parent_goal"] = parent_goal
        else:
            goal_hash = hashlib.sha256(parent_goal.encode("utf-8")).hexdigest()[:16]
            attributes["parent_goal_hash"] = goal_hash

    return create_event(
        event_type="agent.run",
        source="semantic",
        attribution="direct",
        run_id=run_id,
        step_id="init",
        trace_id=trace_id,
        span_id=span_id,
        attributes=attributes,
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
    error_type: str | None = None,
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
    if error_type:
        attributes["error_type"] = str(error_type)[:100]

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
    """Step wrapper preserving hierarchical trace identifiers and capturing error events."""

    def __init__(
        self,
        run_id: str,
        trace_id: str | None = None,
        event_sink: Callable[[Event], None] | list[Event] | None = None,
    ) -> None:
        self.run_id = run_id
        self.trace_id = trace_id or uuid.uuid4().hex
        self.step_counter = 0
        self.events: list[Event] = []
        self.event_sink = event_sink
        self.last_error_event: Event | None = None

    def _emit(self, event: Event) -> None:
        self.events.append(event)
        if self.event_sink is not None:
            if isinstance(self.event_sink, list):
                self.event_sink.append(event)
            elif callable(self.event_sink):
                self.event_sink(event)

    def trace_step(
        self,
        node_name: str,
        fn: Callable[[dict[str, Any]], dict[str, Any]],
        state: dict[str, Any],
        *,
        tool_name: str | None = None,
        request_id: str | None = None,
        parent_span_id: str | None = None,
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
                parent_span_id=parent_span_id,
                tool_name=tool_name or result.get("tool_name"),
                selected_action=result.get("selected_action"),
                status="success",
                latency_ms=round(latency_ms, 2),
            )
            self._emit(event)
            return result, event
        except Exception as ex:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            error_event = agent_step_event(
                run_id=self.run_id,
                step_id=step_id,
                node_name=node_name,
                request_id=request_id,
                trace_id=self.trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                tool_name=tool_name,
                status="error",
                error_type=type(ex).__name__,
                error_message=str(ex),
                latency_ms=round(latency_ms, 2),
            )
            self.last_error_event = error_event
            self._emit(error_event)
            raise ex


class TeleDuctLangGraphCallbackHandler(BaseCallbackHandler):
    """LangGraph / LangChain callback handler maintaining hierarchical spans and emitting events."""

    def __init__(
        self,
        run_id: str,
        trace_id: str | None = None,
        event_sink: Callable[[Event], None] | list[Event] | None = None,
    ) -> None:
        super().__init__()
        self.run_id = run_id
        self.trace_id = trace_id or uuid.uuid4().hex
        self.events: list[Event] = []
        self.event_sink = event_sink
        self.step_counter = 0
        self._span_map: dict[str, str] = {}  # langchain run_id -> span_id
        self._start_times: dict[str, float] = {}
        self._root_run_id: str | None = None
        self._root_span_id: str | None = None

    def _emit(self, event: Event) -> None:
        self.events.append(event)
        if self.event_sink is not None:
            if isinstance(self.event_sink, list):
                self.event_sink.append(event)
            elif callable(self.event_sink):
                self.event_sink(event)

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: Sequence[str] | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        lc_id = str(run_id)
        span_id = uuid.uuid4().hex[:16]
        self._span_map[lc_id] = span_id
        self._start_times[lc_id] = time.perf_counter()

        if parent_run_id is None and self._root_run_id is None:
            self._root_run_id = lc_id
            self._root_span_id = span_id
            # Emit root agent.run event
            run_ev = agent_run_event(
                run_id=self.run_id,
                trace_id=self.trace_id,
                span_id=span_id,
            )
            self._emit(run_ev)

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        lc_id = str(run_id)
        span_id = self._span_map.get(lc_id, uuid.uuid4().hex[:16])
        t0 = self._start_times.pop(lc_id, None)
        latency_ms = (time.perf_counter() - t0) * 1000.0 if t0 else 0.0

        # Don't emit step for the outer graph wrapper itself if it's the root
        if lc_id == self._root_run_id:
            return

        parent_span_id = self._span_map.get(str(parent_run_id)) if parent_run_id else self._root_span_id
        step_name = name or f"node_{self.step_counter:03d}"
        step_id = f"step_{self.step_counter:03d}_{step_name}"
        self.step_counter += 1

        selected_action = None
        req_id = None
        if isinstance(outputs, dict):
            selected_action = outputs.get("selected_action")
            req_id = outputs.get("request_id")

        ev = agent_step_event(
            run_id=self.run_id,
            step_id=step_id,
            node_name=step_name,
            request_id=req_id,
            trace_id=self.trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            selected_action=selected_action,
            status="success",
            latency_ms=round(latency_ms, 2),
        )
        self._emit(ev)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        lc_id = str(run_id)
        span_id = self._span_map.get(lc_id, uuid.uuid4().hex[:16])
        t0 = self._start_times.pop(lc_id, None)
        latency_ms = (time.perf_counter() - t0) * 1000.0 if t0 else 0.0

        if lc_id == self._root_run_id:
            return

        parent_span_id = self._span_map.get(str(parent_run_id)) if parent_run_id else self._root_span_id
        step_name = name or f"node_{self.step_counter:03d}"
        step_id = f"step_{self.step_counter:03d}_{step_name}"
        self.step_counter += 1

        ev = agent_step_event(
            run_id=self.run_id,
            step_id=step_id,
            node_name=step_name,
            trace_id=self.trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            status="error",
            error_type=type(error).__name__,
            error_message=str(error),
            latency_ms=round(latency_ms, 2),
        )
        self._emit(ev)

    def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        lc_id = str(run_id)
        self._span_map[lc_id] = uuid.uuid4().hex[:16]
        self._start_times[lc_id] = time.perf_counter()

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        lc_id = str(run_id)
        span_id = self._span_map.get(lc_id, uuid.uuid4().hex[:16])
        t0 = self._start_times.pop(lc_id, None)
        latency_ms = (time.perf_counter() - t0) * 1000.0 if t0 else 0.0
        parent_span_id = self._span_map.get(str(parent_run_id)) if parent_run_id else self._root_span_id

        tool_name = name or "unknown_tool"
        step_id = f"tool_{self.step_counter:03d}_{tool_name}"
        self.step_counter += 1

        ev = agent_step_event(
            run_id=self.run_id,
            step_id=step_id,
            node_name=tool_name,
            tool_name=tool_name,
            trace_id=self.trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            status="success",
            latency_ms=round(latency_ms, 2),
        )
        self._emit(ev)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        lc_id = str(run_id)
        span_id = self._span_map.get(lc_id, uuid.uuid4().hex[:16])
        t0 = self._start_times.pop(lc_id, None)
        latency_ms = (time.perf_counter() - t0) * 1000.0 if t0 else 0.0
        parent_span_id = self._span_map.get(str(parent_run_id)) if parent_run_id else self._root_span_id

        tool_name = name or "unknown_tool"
        step_id = f"tool_{self.step_counter:03d}_{tool_name}"
        self.step_counter += 1

        ev = agent_step_event(
            run_id=self.run_id,
            step_id=step_id,
            node_name=tool_name,
            tool_name=tool_name,
            trace_id=self.trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            status="error",
            error_type=type(error).__name__,
            error_message=str(error),
            latency_ms=round(latency_ms, 2),
        )
        self._emit(ev)
