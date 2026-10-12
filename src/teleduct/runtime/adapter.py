"""Truthful inference adapter for TeleDuct runtime."""
from __future__ import annotations
import math
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable
import requests

from teleduct.contracts import Event
from teleduct.runtime.normalize import (
    inference_request_event,
    parse_openai_completion_response,
)


@dataclass(frozen=True)
class TruthfulCompletionResponse:
    content: str
    margin: float | None
    fragility_flag: bool | None
    confidence_label: str
    client_request_id: str
    engine_native_id: str | None
    raw_data: dict[str, Any]


class TruthfulInferenceClient:
    """Live LLM / engine client that captures honest telemetry without synthetic fallbacks."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "llama3.2:latest",
        engine: str = "ollama",
        event_sink: Callable[[Event], None] | list[Event] | None = None,
        timeout: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.engine = engine
        self.event_sink = event_sink
        self.timeout = timeout
        self.session = session or requests.Session()
        self.events: list[Event] = []

    def _emit(self, event: Event) -> None:
        self.events.append(event)
        if self.event_sink is not None:
            if isinstance(self.event_sink, list):
                self.event_sink.append(event)
            elif callable(self.event_sink):
                self.event_sink(event)

    def invoke(
        self,
        prompt: str,
        *,
        client_request_id: str | None = None,
        run_id: str | None = None,
        step_id: str | None = None,
        span_id: str | None = None,
        trace_id: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 128,
        logprobs: bool = True,
        top_logprobs: int = 5,
    ) -> TruthfulCompletionResponse:
        req_id = client_request_id or f"client-req-{uuid.uuid4().hex[:12]}"

        # 1. Emit outbound model.request event
        req_event = inference_request_event(
            client_request_id=req_id,
            engine=self.engine,
            model=self.model,
            run_id=run_id,
            step_id=step_id,
            span_id=span_id,
            trace_id=trace_id,
        )
        self._emit(req_event)

        # 2. Execute network request
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
        }
        headers = {"X-Request-ID": req_id}

        t0 = time.perf_counter()
        resp = self.session.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        data = resp.json()

        # 3. Truthfully parse response
        resp_event = parse_openai_completion_response(
            data,
            latency_ms=latency_ms,
            client_request_id=req_id,
            engine=self.engine,
            run_id=run_id,
            step_id=step_id,
            span_id=span_id,
            trace_id=trace_id,
        )
        self._emit(resp_event)

        # 4. Extract content and honest logprob metrics (no synthetic 0.50 fallback)
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content", "")

        sampling_margin = resp_event.metrics.get("sampling_margin")
        fragility_flag = resp_event.attributes.get("fragility_flag")
        engine_native_id = resp_event.attributes.get("engine_native_id")

        if sampling_margin is not None:
            confidence_label = (
                "HIGH" if sampling_margin > 0.30 else
                ("LOW" if fragility_flag else "MEDIUM")
            )
        else:
            confidence_label = "UNAVAILABLE"

        return TruthfulCompletionResponse(
            content=content,
            margin=sampling_margin,
            fragility_flag=fragility_flag,
            confidence_label=confidence_label,
            client_request_id=req_id,
            engine_native_id=engine_native_id,
            raw_data=data,
        )
