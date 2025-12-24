from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from unittest.mock import Mock

import pytest
from jose import jwt

from auth.repositories.auth import TokenRepository
from auth.services.auth import TokenService
from tests.unit.conftest import ExecuteResult
from tests.unit.factories import UserDTOFactory


def session_execute_returns(
    session_mock: Mock,
    *,
    scalars: Optional[Iterable[Any]] = None,
    scalar_one: Any = None,
):
    """
    Configure what session.execute(...) should return.
    scalars: iterable of objects to be returned by result.scalars().all()
    scalar_one: object to be returned by result.scalar_one()
    """
    scalars_list = list(scalars) if scalars is not None else []
    session_mock._last_execute_result = ExecuteResult(scalars_list, scalar_one)


@pytest.fixture
def token_service(session_mock, jwt_config):
    """Fixture to create TokenService with mocked dependencies."""
    token_repository = TokenRepository(session_mock)
    return TokenService(token_repository, jwt_config)


@pytest.mark.asyncio
async def test_create_access_token_with_default_expiry(token_service):
    """Test creating access token with default expiry time"""
    user_dto = UserDTOFactory.create()

    token = await token_service.create_access_token(user_dto)

    # Verify it returns a valid JWT that can be decoded
    assert isinstance(token, str)
    decoded = jwt.decode(
        token, token_service.secret_key, algorithms=[token_service.algorithm]
    )
    assert "exp" in decoded


@pytest.mark.asyncio
async def test_create_access_token_with_custom_expiry(token_service):
    """Test creating access token with custom expiry time"""
    user_dto = UserDTOFactory.create()
    custom_expiry = timedelta(hours=2)

    token = await token_service.create_access_token(user_dto, custom_expiry)

    # Verify custom expiry was used
    decoded = jwt.decode(
        token, token_service.secret_key, algorithms=[token_service.algorithm]
    )
    exp_time = datetime.fromtimestamp(decoded["exp"], timezone.utc)
    expected_exp = datetime.now(timezone.utc) + custom_expiry
    assert abs((exp_time - expected_exp).total_seconds()) < 5


@pytest.mark.asyncio
async def test_create_access_token_calls_repository_create(token_service, session_mock):
    """Test that create_access_token properly calls the repository create method"""
    user_dto = UserDTOFactory.create()

    await token_service.create_access_token(user_dto)

    # Verify session methods were called (indicating repository.create was called)
    session_mock.add.assert_called_once()
    session_mock.refresh.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_access_token_preserves_original_data(token_service):
    """Test that original data dictionary is not modified"""
    user_dto = UserDTOFactory.create()
    id = user_dto.id
    email = user_dto.email

    await token_service.create_access_token(user_dto)

    # Verify original UserDTO wasn't modified
    assert id == user_dto.id
    assert email == user_dto.email


@pytest.mark.asyncio
async def test_create_access_token_uses_correct_algorithm_and_issuer(token_service):
    """Test that JWT uses correct algorithm and issuer"""
    user_dto = UserDTOFactory.create()

    token = await token_service.create_access_token(user_dto)

    # Verify correct issuer and algorithm
    decoded = jwt.decode(
        token, token_service.secret_key, algorithms=[token_service.algorithm]
    )
    assert decoded["iss"] == "fastapi-skeleton"
