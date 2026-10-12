"""Normalize *already observed* application step boundaries into TeleDuct events.

LangGraph callbacks/OTel propagation are an implementation task, not claimed here.
"""
from teleduct.contracts import Event, create_event


def agent_step_event(*, run_id: str, step_id: str, node_name: str,
                     request_id: str | None = None, trace_id: str | None = None,
                     tool_name: str | None = None, input_tokens: int | None = None,
                     output_tokens: int | None = None) -> Event:
    if not run_id or not step_id or not node_name:
        raise ValueError("run_id, step_id and node_name required")
    metrics = {}
    for key, value in (("input_tokens", input_tokens), ("output_tokens", output_tokens)):
        if value is not None:
            if value < 0:
                raise ValueError("token counts must be nonnegative")
            metrics[key] = value
    attributes = {"node_name": node_name}
    if tool_name:
        attributes["tool_name"] = tool_name
    return create_event(event_type="agent.step", source="semantic", attribution="direct",
                        run_id=run_id, step_id=step_id, request_id=request_id,
                        trace_id=trace_id, attributes=attributes, metrics=metrics)
