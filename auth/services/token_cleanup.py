"""Background task for cleaning up expired tokens.

This module provides a scheduled task that removes expired tokens
from the database to prevent database bloat.
"""

import asyncio
from datetime import datetime, timezone

import structlog
from sqlalchemy import delete
from sqlalchemy.engine import CursorResult

from auth.models.db import Token
from core.database import async_session

logger = structlog.get_logger()


class TokenCleanupService:
    """Service for cleaning up expired tokens from the database."""

    def __init__(self, cleanup_interval_seconds: int = 3600):
        """
        Initialize the token cleanup service.

        Args:
            cleanup_interval_seconds: How often to run cleanup (default: 1 hour)
        """
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self._task = None
        self._running = False

    async def cleanup_expired_tokens(self) -> int:
        """
        Delete all expired tokens from the database.

        Returns:
            Number of tokens deleted
        """
        async with async_session() as session:
            try:
                now = datetime.now(timezone.utc)

                # Delete expired tokens
                stmt = delete(Token).where(Token.expires_at < now)
                result: CursorResult = await session.execute(stmt)  # type: ignore[assignment]
                await session.commit()

                deleted_count: int = result.rowcount or 0

                if deleted_count > 0:
                    logger.info(
                        "token_cleanup_completed",
                        deleted_count=deleted_count,
                        cleanup_time=now.isoformat(),
                    )

                return deleted_count

            except Exception as e:
                logger.error(
                    "token_cleanup_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                await session.rollback()
                raise

    async def cleanup_inactive_tokens(self, days_to_keep: int = 30) -> int:
        """
        Delete inactive tokens older than specified days.

        Args:
            days_to_keep: Keep inactive tokens for this many days

        Returns:
            Number of tokens deleted
        """
        async with async_session() as session:
            try:
                from datetime import timedelta

                cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)

                stmt = delete(Token).where(
                    Token.is_active == False,  # noqa: E712
                    Token.updated_at < cutoff_date,  # noqa: E712
                )
                result: CursorResult = await session.execute(stmt)  # type: ignore[assignment]
                await session.commit()

                deleted_count: int = result.rowcount or 0

                if deleted_count > 0:
                    logger.info(
                        "inactive_token_cleanup_completed",
                        deleted_count=deleted_count,
                        cutoff_date=cutoff_date.isoformat(),
                    )

                return deleted_count

            except Exception as e:
                logger.error(
                    "inactive_token_cleanup_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                await session.rollback()
                raise

    async def run_periodic_cleanup(self):
        """Run cleanup task periodically."""
        self._running = True
        logger.info(
            "token_cleanup_service_started",
            interval_seconds=self.cleanup_interval_seconds,
        )

        while self._running:
            try:
                # Cleanup expired tokens
                await self.cleanup_expired_tokens()

                # Cleanup old inactive tokens (keep last 30 days)
                await self.cleanup_inactive_tokens(days_to_keep=30)

            except Exception as e:
                logger.exception(
                    "periodic_cleanup_error",
                    error=str(e),
                )

            # Wait for next interval
            await asyncio.sleep(self.cleanup_interval_seconds)

    def start(self):
        """Start the background cleanup task."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.run_periodic_cleanup())
            logger.info("token_cleanup_task_created")
        else:
            logger.warning("token_cleanup_task_already_running")

    async def stop(self):
        """Stop the background cleanup task."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            logger.info("token_cleanup_task_stopped")


# Global instance
_cleanup_service = None


def get_cleanup_service(cleanup_interval_seconds: int = 3600) -> TokenCleanupService:
    """Get or create the token cleanup service singleton."""
    global _cleanup_service
    if _cleanup_service is None:
        _cleanup_service = TokenCleanupService(cleanup_interval_seconds)
    return _cleanup_service
