from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from core.settings import get_settings


def setup_tracing():
    settings = get_settings()
    otlp_endpoint = settings.otlp_endpoint
    resource = Resource.create(
        {
            "service.name": settings.service_name,
        }
    )

    provider = TracerProvider(resource=resource)

    # 1️⃣ Always enable console exporter (dev-friendly)
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    # 2️⃣ Optional OTLP exporter (collector, Tempo, etc.)
    if otlp_endpoint != "":
        try:
            # For proto.http exporter, provide full URL including /v1/traces
            endpoint = (
                otlp_endpoint
                if otlp_endpoint.endswith("/v1/traces")
                else f"{otlp_endpoint}/v1/traces"
            )
            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
            )
        except Exception as e:
            # Log warning but don't fail - OTLP endpoint might not be available
            import structlog

            logger = structlog.get_logger()
            logger.warning(
                "otlp_exporter_unavailable", endpoint=otlp_endpoint, error=str(e)
            )

    trace.set_tracer_provider(provider)
