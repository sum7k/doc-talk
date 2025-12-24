"""Shared test fixtures and utilities for unit tests.

This module provides reusable fixtures, mocks, and utilities
for testing the application.
"""

from typing import Any, List, Optional
from unittest.mock import AsyncMock, Mock

import pytest

from auth.repositories.auth import TokenRepository, UserRepository
from auth.services.auth import PasswordService, TokenService
from core.settings import JWTConfig


class AsyncCtxMgr:
    """Simple async context manager used for `async with session.begin():`"""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class ExecuteResult:
    """
    Lightweight mimic of SQLAlchemy result for tests.
    - .scalars().all() -> list
    - .scalar_one() -> item or raise NoResultFound
    - .scalar_one_or_none() -> item or None
    """

    def __init__(
        self, scalars_list: Optional[List[Any]] = None, scalar_single: Any = None
    ):
        self._scalars_list = list(scalars_list or [])
        self._scalar_single = scalar_single

        # scalars() returns object with .all() (sync) in SQLAlchemy; we mimic that
        class _Scalars:
            def __init__(self, lst):
                self._lst = lst

            def all(self):
                return self._lst

        self._scalars = _Scalars(self._scalars_list)

    def scalars(self):
        return self._scalars

    def scalar_one(self):
        # prefer explicit scalar_one override, else use first element, else raise
        if self._scalar_single is not None:
            return self._scalar_single
        if self._scalars_list:
            return self._scalars_list[0]
        # raise same exception SQLAlchemy raises for no result
        from sqlalchemy.exc import NoResultFound

        raise NoResultFound("No row found for query")

    def scalar_one_or_none(self):
        """Return one result or None."""
        if self._scalar_single is not None:
            return self._scalar_single
        if self._scalars_list:
            return self._scalars_list[0]
        return None


def make_session_mock() -> Mock:
    """
    Return a Mock that behaves like sqlalchemy.ext.asyncio.AsyncSession for tests.
    - session.execute is AsyncMock returning ExecuteResult
    - session.begin() returns an async context manager object
    - commit/rollback/flush/refresh/delete are AsyncMock (awaitable)
    - add is regular Mock (add is sync in SQLAlchemy)
    """
    session = Mock(name="AsyncSessionMock")

    # Async methods
    session.execute = AsyncMock(name="execute")
    session.flush = AsyncMock(name="flush")
    session.refresh = AsyncMock(name="refresh")
    session.delete = AsyncMock(name="delete")
    session.commit = AsyncMock(name="commit")
    session.rollback = AsyncMock(name="rollback")

    # add is sync in SQLAlchemy; keep as Mock
    session.add = Mock(name="add")

    # store a default execute result that tests can replace
    session._last_execute_result = ExecuteResult([])

    async def _execute_side_effect(*args, **kwargs):
        # always return the stored ExecuteResult (synchronous object)
        return session._last_execute_result

    session.execute.side_effect = _execute_side_effect

    # begin() should return an async context manager
    session.begin = Mock(name="begin", return_value=AsyncCtxMgr())

    return session


@pytest.fixture
def session_mock():
    """Provides a mocked AsyncSession for testing."""
    return make_session_mock()


@pytest.fixture
def mock_password_service():
    """Provides a mocked PasswordService."""
    return AsyncMock(spec=PasswordService)


@pytest.fixture
def mock_token_service():
    """Provides a mocked TokenService."""
    return AsyncMock(spec=TokenService)


@pytest.fixture
def mock_user_repository(session_mock):
    """Provides a mocked UserRepository."""
    return Mock(spec=UserRepository)


@pytest.fixture
def mock_token_repository(session_mock):
    """Provides a mocked TokenRepository."""
    return Mock(spec=TokenRepository)


@pytest.fixture
def jwt_config():
    """Provides a test JWT configuration."""
    return JWTConfig(
        secret_key="test-secret-key-with-at-least-32-characters-for-security",
        algorithm="HS256",
        access_token_expire_minutes=60,
    )
