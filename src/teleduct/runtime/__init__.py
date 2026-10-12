"""TeleDuct inference runtime adapter."""
from teleduct.runtime.normalize import (
    inference_event,
    inference_request_event,
    parse_openai_completion_response,
)
from teleduct.runtime.adapter import (
    TruthfulInferenceClient,
    TruthfulCompletionResponse,
)

__all__ = [
    "inference_event",
    "inference_request_event",
    "parse_openai_completion_response",
    "TruthfulInferenceClient",
    "TruthfulCompletionResponse",
]
