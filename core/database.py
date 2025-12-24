from contextlib import asynccontextmanager
from typing import Annotated, AsyncGenerator

from fastapi.params import Depends
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from core.instrumentation import setup_db_metrics
from core.settings import get_settings


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


# Get database configuration
settings = get_settings()
db_config = settings.get_database_config()

# Create async engine with configuration
engine = create_async_engine(
    db_config.url,
    echo=db_config.echo,
    pool_size=db_config.pool_size,
    max_overflow=db_config.max_overflow,
)

SQLAlchemyInstrumentor().instrument(engine=engine.sync_engine)
setup_db_metrics(engine.sync_engine)


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)


async_session = async_sessionmaker(
    bind=engine, class_=AsyncSession, autocommit=False, autoflush=False
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for FastAPI - yields a database session."""
    async with async_session() as session:
        try:
            yield session
            await session.commit()  # Commit on successful completion
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def get_db_session_context() -> AsyncGenerator[AsyncSession, None]:
    """Context manager for notebooks/scripts - use with 'async with'."""
    async for session in get_db_session():
        yield session
        break


DBSessionDep = Annotated[AsyncSession, Depends(get_db_session)]
