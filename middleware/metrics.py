import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from core.metrics import ERROR_COUNT, REQUEST_COUNT, REQUEST_LATENCY


class PrometheusMetricsMiddleware(BaseHTTPMiddleware):
    """Middleware that collects Prometheus metrics for HTTP requests."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()
        method = request.method
        path = request.url.path

        status_code = "500"

        try:
            response = await call_next(request)
            status_code = str(response.status_code)
            return response

        finally:
            duration = time.perf_counter() - start_time

            REQUEST_COUNT.labels(
                method=method,
                path=path,
                status_code=status_code,
            ).inc()

            REQUEST_LATENCY.labels(
                method=method,
                path=path,
                status_code=status_code,
            ).observe(duration)

            if status_code.startswith("5"):
                ERROR_COUNT.labels(
                    method=method,
                    path=path,
                    status_code=status_code,
                ).inc()
