import time
import uuid

import structlog
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from structlog.contextvars import bind_contextvars, clear_contextvars

logger = structlog.get_logger()

CORRELATION_HEADER = "X-Correlation-ID"


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Middleware that adds correlation IDs to requests for distributed tracing."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()

        correlation_id = request.headers.get(CORRELATION_HEADER) or str(uuid.uuid4())

        # Bind context for this request
        bind_contextvars(
            correlation_id=correlation_id,
            path=request.url.path,
            method=request.method,
        )

        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start_time) * 1000

            logger.info(
                "request_completed",
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
            )

            response.headers[CORRELATION_HEADER] = correlation_id
            return response

        finally:
            clear_contextvars()
