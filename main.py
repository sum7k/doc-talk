import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from fastapi import Depends, FastAPI, Response, status
from fastapi.responses import JSONResponse
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from auth.routers.auth import router as auth_router
from core.exceptions import (
    DomainException,
    ReadinessError,
    domain_exception_handler,
    generic_exception_handler,
)
from core.logging import setup_logging
from core.readiness import check_db_readiness
from core.security import RequireMember
from core.settings import Settings, get_settings
from core.tracing import setup_tracing
from middleware import CorrelationIdMiddleware, PrometheusMetricsMiddleware

READINESS_TIMEOUT = 1.0  # seconds

setup_logging()
setup_tracing()

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Startup
    # await init_db()

    # Start token cleanup service (runs every hour)
    from auth.services.token_cleanup import get_cleanup_service

    cleanup_service = get_cleanup_service(cleanup_interval_seconds=3600)
    cleanup_service.start()
    logger.info("application_startup_complete")

    yield

    # Shutdown
    await cleanup_service.stop()
    logger.info("application_shutdown_complete")


app = FastAPI(lifespan=lifespan)

# Register exception handlers
app.add_exception_handler(DomainException, domain_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# Add middleware
app.add_middleware(CorrelationIdMiddleware)
app.add_middleware(PrometheusMetricsMiddleware)

# Include routers
app.include_router(auth_router)

FastAPIInstrumentor.instrument_app(app)


@app.get("/health", tags=["infra"])
async def health():
    return {"status": "ok"}


@app.get("/ready", tags=["infra"])
async def ready():
    try:
        await asyncio.wait_for(
            asyncio.gather(
                check_db_readiness(),  # REQUIRED
                # check_redis(redis),        # OPTIONAL
            ),
            timeout=READINESS_TIMEOUT,
        )
        return {"status": "ready"}

    except (asyncio.TimeoutError, ReadinessError):
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready"},
        )


@app.get("/info")
def info(user: RequireMember, settings: Settings = Depends(get_settings)):
    logger.info("handling info request", user_id=user.id)
    return {
        "db": settings.db_url,
    }


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
