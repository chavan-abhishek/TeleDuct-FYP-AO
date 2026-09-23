# test_tunnel_span.py — corrected
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
import os, time

OTEL_ENDPOINT = os.getenv("OTEL_ENDPOINT", "http://localhost:4318")

resource = Resource.create({"service.name": "tunnel-test-service"})
provider = TracerProvider(resource=resource)
provider.add_span_processor(BatchSpanProcessor(
    OTLPSpanExporter(
        endpoint=f"{OTEL_ENDPOINT}/v1/traces",   # ← explicit full path
        headers={"ngrok-skip-browser-warning": "true"}
    )
))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("tunnel-test")
with tracer.start_as_current_span("test_span_through_ngrok") as span:
    span.set_attribute("test.confirmed", True)
    print("Span sent through ngrok tunnel — check OTel Collector / Grafana now")

time.sleep(3)