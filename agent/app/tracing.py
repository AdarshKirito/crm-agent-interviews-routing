"""Send ADK traces to Arize Phoenix when PHOENIX_COLLECTOR_ENDPOINT is set.

Local:  uvx --from arize-phoenix phoenix serve   (UI at http://localhost:6006)
        PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
Docker: PHOENIX_COLLECTOR_ENDPOINT=http://host.docker.internal:6006

Plain OpenTelemetry SDK wiring rather than phoenix.otel.register(): its provider drops
its exporter on the first add_span_processor() call, which ADK's API server makes, and
a bare http:// endpoint makes it choose gRPC with TLS, which a local Phoenix rejects.
"""

import logging
import os

logger = logging.getLogger(__name__)
_done = False


def setup_tracing() -> bool:
    global _done
    if _done or not os.getenv("PHOENIX_COLLECTOR_ENDPOINT"):
        return _done
    from openinference.instrumentation.google_adk import GoogleADKInstrumentor
    from openinference.semconv.resource import ResourceAttributes
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    endpoint = os.environ["PHOENIX_COLLECTOR_ENDPOINT"].rstrip("/")
    if not endpoint.endswith("/v1/traces"):
        endpoint += "/v1/traces"
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):  # nothing installed yet: install ours
        project = os.getenv("PHOENIX_PROJECT_NAME", "crmroute")
        provider = TracerProvider(resource=Resource.create({ResourceAttributes.PROJECT_NAME: project}))
        trace.set_tracer_provider(provider)
    # Otherwise the global provider is already set (it cannot be replaced): export from it too.
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    GoogleADKInstrumentor().instrument(tracer_provider=provider)
    _done = True
    logger.info("Phoenix tracing enabled -> %s", endpoint)
    return True
