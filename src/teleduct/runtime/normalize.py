"""Engine-agnostic lifecycle event normalization for TeleDuct runtime."""
from __future__ import annotations
import math
from typing import Any
from teleduct.contracts import Event, create_event


def inference_event(
    *,
    request_id: str,
    engine: str,
    ttft_ms: float | None = None,
    tpot_ms: float | None = None,
    queue_ms: float | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    latency_ms: float | None = None,
) -> Event:
    """Existing adapter preserved for backward compatibility."""
    if not request_id or not engine:
        raise ValueError("request_id and engine required")
    metrics: dict[str, float | int] = {}
    for key, value in (
        ("ttft_ms", ttft_ms),
        ("tpot_ms", tpot_ms),
        ("queue_ms", queue_ms),
        ("input_tokens", input_tokens),
        ("output_tokens", output_tokens),
        ("latency_ms", latency_ms),
    ):
        if value is not None:
            if value < 0:
                raise ValueError("measurements cannot be negative")
            metrics[key] = value
    return create_event(
        event_type="model.response",
        source="runtime",
        attribution="direct",
        request_id=request_id,
        attributes={"engine": engine},
        metrics=metrics,
    )


def inference_request_event(
    *,
    request_id: str,
    engine: str,
    model: str,
    run_id: str | None = None,
    step_id: str | None = None,
    span_id: str | None = None,
    trace_id: str | None = None,
) -> Event:
    """Outbound model request event."""
    if not request_id or not engine or not model:
        raise ValueError("request_id, engine, and model required")
    return create_event(
        event_type="model.request",
        source="runtime",
        attribution="direct",
        request_id=request_id,
        run_id=run_id,
        step_id=step_id,
        span_id=span_id,
        trace_id=trace_id,
        attributes={"engine": engine, "model": model},
    )


def parse_openai_completion_response(
    response_data: dict[str, Any],
    *,
    latency_ms: float,
    engine: str = "sglang",
    run_id: str | None = None,
    step_id: str | None = None,
    span_id: str | None = None,
    trace_id: str | None = None,
) -> Event:
    """Truthfully parse an OpenAI / SGLang compatible chat completion response.

    Rules:
    - Never fabricates default margins (e.g. 0.50). If logprobs are absent,
      sampling_margin is omitted.
    - Never reports non-streaming total latency as TTFT.
    """
    request_id = response_data.get("id")
    if not request_id:
        raise ValueError("response payload missing native 'id'")
    if latency_ms < 0:
        raise ValueError("latency_ms must be non-negative")

    model = response_data.get("model", "unknown")
    metrics: dict[str, float | int] = {"latency_ms": round(latency_ms, 2)}
    attributes: dict[str, Any] = {"engine": engine, "model": model}

    usage = response_data.get("usage")
    if isinstance(usage, dict):
        if "prompt_tokens" in usage and usage["prompt_tokens"] is not None:
            metrics["prompt_tokens"] = usage["prompt_tokens"]
        if "completion_tokens" in usage and usage["completion_tokens"] is not None:
            metrics["completion_tokens"] = usage["completion_tokens"]
        if "total_tokens" in usage and usage["total_tokens"] is not None:
            metrics["total_tokens"] = usage["total_tokens"]

    # Extract observed logprobs if present without synthetic fallback
    choices = response_data.get("choices", [])
    logprobs_found = False
    if choices and isinstance(choices, list) and len(choices) > 0:
        choice = choices[0]
        if isinstance(choice, dict):
            finish_reason = choice.get("finish_reason")
            if finish_reason:
                attributes["finish_reason"] = finish_reason

            logprobs_info = choice.get("logprobs")
            if isinstance(logprobs_info, dict) and "content" in logprobs_info:
                content_tokens = logprobs_info.get("content")
                if isinstance(content_tokens, list) and len(content_tokens) > 0:
                    margins = []
                    for t in content_tokens:
                        top_lps = t.get("top_logprobs", []) if isinstance(t, dict) else []
                        if len(top_lps) >= 2:
                            p1 = math.exp(top_lps[0]["logprob"])
                            p2 = math.exp(top_lps[1]["logprob"])
                            margins.append(max(0.0, min(1.0, p1 - p2)))
                    if margins:
                        # Bottleneck (minimal) sampling margin across tokens
                        metrics["sampling_margin"] = round(min(margins), 4)
                        attributes["fragility_flag"] = metrics["sampling_margin"] < 0.05
                        attributes["logprobs_status"] = "measured"
                        logprobs_found = True

    if not logprobs_found:
        attributes["logprobs_status"] = "unavailable"

    return create_event(
        event_type="model.response",
        source="runtime",
        attribution="direct",
        request_id=request_id,
        run_id=run_id,
        step_id=step_id,
        span_id=span_id,
        trace_id=trace_id,
        attributes=attributes,
        metrics=metrics,
    )
