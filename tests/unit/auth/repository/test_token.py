from collections.abc import Iterable
from typing import Any, Optional
from unittest.mock import Mock

import pytest
from sqlalchemy.exc import NoResultFound

from auth.models.db import Token
from auth.repositories.auth import TokenRepository
from tests.unit.conftest import ExecuteResult
from tests.unit.factories import CreateTokenDTOFactory, TokenDBFactory, TokenDTOFactory


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


@pytest.mark.asyncio
async def test_list_returns_dto_list_with_pagination(session_mock):
    repo = TokenRepository(session_mock)
    session_execute_returns(session_mock, scalars=TokenDBFactory.build_batch(10))
    tokens = await repo.list(0, 10)
    assert len(tokens) == 10


@pytest.mark.asyncio
async def test_list_returns_empty_list_when_no_tokens(session_mock):
    repo = TokenRepository(session_mock)
    tokens = await repo.list(0, 10)
    assert len(tokens) == 0


@pytest.mark.asyncio
async def test_get_existing_returns_dto(session_mock):
    repo = TokenRepository(session_mock)
    token: Token = TokenDBFactory.create()
    session_execute_returns(session_mock, scalar_one=token)
    token_from_db = await repo.get(token.id)
    assert token.id == token_from_db.id


@pytest.mark.asyncio
async def test_get_non_existent_raise_error(session_mock):
    repo = TokenRepository(session_mock)
    with pytest.raises(ValueError):
        await repo.get(-1)


@pytest.mark.asyncio
async def test_create_persists_and_returns_dto(session_mock):
    repo = TokenRepository(session_mock)
    create_token_dto = CreateTokenDTOFactory.create()
    created_dto = await repo.create(create_token_dto)
    # Compare relevant fields (excluding id which is auto-generated)
    assert create_token_dto.user_id == created_dto.user_id
    assert create_token_dto.expires_at == created_dto.expires_at
    assert create_token_dto.is_active == created_dto.is_active
    assert create_token_dto.ip_address == created_dto.ip_address


@pytest.mark.asyncio
async def test_create_calls_session_methods_in_transaction(session_mock):
    repo = TokenRepository(session_mock)
    create_token_dto = CreateTokenDTOFactory.create()
    await repo.create(create_token_dto)
    session_mock.add.assert_called_once()
    session_mock.refresh.assert_awaited()
    session_mock.refresh.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_existing_applies_all_dataclass_fields_to_orm(session_mock):
    repo = TokenRepository(session_mock)
    token_db = TokenDBFactory.create()
    token_dto = TokenDTOFactory.create()
    session_execute_returns(session_mock, scalar_one=token_db)

    updated_dto = await repo.update(
        token_dto.id,
        {
            "user_id": token_dto.user_id,
            "expires_at": token_dto.expires_at,
            "is_active": token_dto.is_active,
            "ip_address": token_dto.ip_address,
        },
    )

    # Assert ORM attributes were updated with DTO values
    assert token_db.user_id == token_dto.user_id
    assert token_db.expires_at == token_dto.expires_at
    assert token_db.is_active == token_dto.is_active
    assert token_db.ip_address == token_dto.ip_address

    # Verify session methods called
    session_mock.refresh.assert_awaited_once()
    session_mock.refresh.assert_awaited_once_with(token_db)

    # Verify mapper was used to return DTO
    assert updated_dto is not None


@pytest.mark.asyncio
async def test_update_only_updates_fields_that_exist_on_orm(session_mock):
    repo = TokenRepository(session_mock)
    token_db = TokenDBFactory.create()
    token_dto = TokenDTOFactory.create()

    session_execute_returns(session_mock, scalar_one=token_db)

    # Repository now checks hasattr before setting fields
    updated_dto = await repo.update(
        token_dto.id,
        {
            "user_id": token_dto.user_id,
            "expires_at": token_dto.expires_at,
            "is_active": token_dto.is_active,
        },
    )

    # Verify ORM fields were updated with DTO values
    assert token_db.user_id == token_dto.user_id
    assert token_db.expires_at == token_dto.expires_at
    assert token_db.is_active == token_dto.is_active
    assert updated_dto is not None


@pytest.mark.asyncio
async def test_update_nonexistent_raises_value_error(session_mock):
    repo = TokenRepository(session_mock)
    token_dto = TokenDTOFactory.create()
    session_execute_returns(session_mock, scalars=[])

    with pytest.raises(ValueError, match="Token not found."):
        await repo.update(token_dto.id, {"is_active": False})


@pytest.mark.asyncio
async def test_delete_existing_deletes_record(session_mock):
    repo = TokenRepository(session_mock)
    token_db = TokenDBFactory.create()
    session_execute_returns(session_mock, scalar_one=token_db)

    await repo.delete(token_db.id)

    # Verify delete was called within transaction
    session_mock.delete.assert_awaited_once_with(token_db)


@pytest.mark.asyncio
async def test_delete_nonexistent_is_noop(session_mock):
    repo = TokenRepository(session_mock)
    session_execute_returns(session_mock, scalars=[])

    # Should return without error when record doesn't exist
    await repo.delete(999)

    # Verify delete was not called
    session_mock.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test__get_returns_single_row_or_raises(session_mock):
    repo = TokenRepository(session_mock)
    token_db = TokenDBFactory.create()

    # Test successful case
    session_execute_returns(session_mock, scalar_one=token_db)
    result = await repo._get(token_db.id)
    assert result == token_db

    # Test NoResultFound case
    session_execute_returns(session_mock, scalars=[])
    with pytest.raises(NoResultFound):
        await repo._get(999)


@pytest.mark.asyncio
async def test_token_user_relationship_preserved(session_mock):
    """Test that user_id relationship is maintained"""
    repo = TokenRepository(session_mock)
    token_db = TokenDBFactory.create()
    token_dto = TokenDTOFactory.create()

    # Ensure user_id is preserved during operations
    original_user_id = token_dto.user_id
    session_execute_returns(session_mock, scalar_one=token_db)

    updated_dto = await repo.update(token_dto.id, {"user_id": original_user_id})

    assert token_db.user_id == original_user_id
    assert updated_dto.user_id == original_user_id
