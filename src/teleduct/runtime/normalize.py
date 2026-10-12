"""Engine-agnostic lifecycle event normalization; not yet a live SGLang adapter."""
from teleduct.contracts import Event, create_event


def inference_event(*, request_id: str, engine: str, ttft_ms: float | None = None,
                    tpot_ms: float | None = None, queue_ms: float | None = None,
                    input_tokens: int | None = None, output_tokens: int | None = None) -> Event:
    if not request_id or not engine:
        raise ValueError("request_id and engine required")
    metrics = {}
    for key, value in (("ttft_ms", ttft_ms), ("tpot_ms", tpot_ms), ("queue_ms", queue_ms),
                       ("input_tokens", input_tokens), ("output_tokens", output_tokens)):
        if value is not None:
            if value < 0:
                raise ValueError("measurements cannot be negative")
            metrics[key] = value
    return create_event(event_type="inference.response", source="runtime", attribution="direct",
                        request_id=request_id, attributes={"engine": engine}, metrics=metrics)
