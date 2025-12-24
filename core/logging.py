import logging

import structlog
from opentelemetry import trace
from structlog.contextvars import merge_contextvars

from core.settings import get_settings


def get_log_level() -> str:
    """Get log level from settings."""
    return get_settings().log_level.upper()


def add_trace_ids(_, __, event_dict):
    span = trace.get_current_span()
    if not span:
        return event_dict

    ctx = span.get_span_context()
    if not ctx.is_valid:
        return event_dict

    event_dict["trace_id"] = format(ctx.trace_id, "032x")
    event_dict["span_id"] = format(ctx.span_id, "016x")
    return event_dict


def setup_logging():
    """Configure structured logging with OpenTelemetry trace integration."""
    log_level = get_log_level()

    logging.basicConfig(
        level=log_level,
        format="%(message)s",
    )

    structlog.configure(
        processors=[
            merge_contextvars,
            add_trace_ids,
            structlog.processors.TimeStamper(fmt="iso"),
            # If log level is too low, abort pipeline and throw away log entry.
            structlog.stdlib.filter_by_level,
            # Add the name of the logger to event dict.
            structlog.stdlib.add_logger_name,
            # Add log level to event dict.
            structlog.stdlib.add_log_level,
            # Perform %-style formatting.
            structlog.stdlib.PositionalArgumentsFormatter(),
            # Add a timestamp in ISO 8601 format.
            structlog.processors.TimeStamper(fmt="iso"),
            # If the "stack_info" key in the event dict is true, remove it and
            # render the current stack trace in the "stack" key.
            structlog.processors.StackInfoRenderer(),
            # If the "exc_info" key in the event dict is either true or a
            # sys.exc_info() tuple, remove "exc_info" and render the exception
            # with traceback into the "exception" key.
            structlog.processors.format_exc_info,
            # If some value is in bytes, decode it to a Unicode str.
            structlog.processors.UnicodeDecoder(),
            # Add callsite parameters.
            structlog.processors.CallsiteParameterAdder(
                {
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                }
            ),
            # Render the final event dict as JSON.
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
