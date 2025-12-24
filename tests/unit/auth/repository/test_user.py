from collections.abc import Iterable
from typing import Any, Optional
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy.exc import NoResultFound

from auth.models.db import User
from auth.repositories.auth import UserRepository
from tests.unit.conftest import ExecuteResult
from tests.unit.factories import CreateUserDTOFactory, UserDBFactory, UserDTOFactory


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
    repo = UserRepository(session_mock)
    session_execute_returns(session_mock, scalars=UserDBFactory.build_batch(10))
    users = await repo.list(0, 10)
    assert len(users) == 10


@pytest.mark.asyncio
async def test_list_returns_empty_list_when_no_users(session_mock):
    repo = UserRepository(session_mock)
    users = await repo.list(0, 10)
    assert len(users) == 0


@pytest.mark.asyncio
async def test_get_existing_returns_dto(session_mock):
    repo = UserRepository(session_mock)
    user: User = UserDBFactory.create()
    session_execute_returns(session_mock, scalar_one=user)
    user_from_db = await repo.get(user.id)
    assert user.id == user_from_db.id


@pytest.mark.asyncio
async def test_get_non_existent_raise_error(session_mock):
    repo = UserRepository(session_mock)
    with pytest.raises(ValueError):
        await repo.get("non-existent-uuid")


@pytest.mark.asyncio
async def test_create_persists_and_returns_dto(session_mock):
    repo = UserRepository(session_mock)
    create_user_dto = CreateUserDTOFactory.create()
    created_dto = await repo.create(create_user_dto)
    # Compare relevant fields (excluding id which is auto-generated)
    assert create_user_dto.email == created_dto.email
    assert create_user_dto.is_active == created_dto.is_active
    assert create_user_dto.role == created_dto.role


@pytest.mark.asyncio
async def test_create_calls_session_methods_in_transaction(session_mock):
    repo = UserRepository(session_mock)
    create_user_dto = CreateUserDTOFactory.create()
    await repo.create(create_user_dto)
    session_mock.add.assert_called_once()
    session_mock.refresh.assert_awaited()
    session_mock.refresh.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_existing_applies_all_dataclass_fields_to_orm(session_mock):
    repo = UserRepository(session_mock)
    user_db = UserDBFactory.create()
    user_dto = UserDTOFactory.create()
    session_execute_returns(session_mock, scalar_one=user_db)

    updated_dto = await repo.update(
        user_dto.id,
        {
            "email": user_dto.email,
            "is_active": user_dto.is_active,
            "role": user_dto.role,
        },
    )

    # Assert ORM attributes were updated with DTO values
    assert user_db.email == user_dto.email
    assert user_db.is_active == user_dto.is_active
    assert user_db.role == user_dto.role

    # Verify session methods called
    session_mock.refresh.assert_awaited_once()

    # Verify mapper was used to return DTO
    assert updated_dto is not None


@pytest.mark.asyncio
async def test_update_only_updates_fields_that_exist_on_orm(session_mock):
    repo = UserRepository(session_mock)
    user_db = UserDBFactory.create()
    user_dto = UserDTOFactory.create()

    session_execute_returns(session_mock, scalar_one=user_db)

    # Repository now checks hasattr before setting fields
    updated_dto = await repo.update(
        user_dto.id,
        {
            "email": user_dto.email,
            "is_active": user_dto.is_active,
            "role": user_dto.role,
        },
    )

    # Verify ORM fields were updated with DTO values
    assert user_db.email == user_dto.email
    assert user_db.is_active == user_dto.is_active
    assert user_db.role == user_dto.role
    assert updated_dto is not None


@pytest.mark.asyncio
async def test_update_nonexistent_raises_value_error(session_mock):
    repo = UserRepository(session_mock)
    user_dto = UserDTOFactory.create()
    session_execute_returns(session_mock, scalars=[])

    with pytest.raises(ValueError, match="User not found."):
        await repo.update(user_dto.id, {"is_active": False})


@pytest.mark.asyncio
async def test_delete_existing_deletes_record(session_mock):
    repo = UserRepository(session_mock)
    user_db = UserDBFactory.create()
    session_execute_returns(session_mock, scalar_one=user_db)

    await repo.delete(user_db.id)

    # Verify delete was called within transaction
    session_mock.delete.assert_awaited_once_with(user_db)


@pytest.mark.asyncio
async def test_delete_nonexistent_is_noop(session_mock):
    repo = UserRepository(session_mock)
    session_execute_returns(session_mock, scalars=[])

    # Should return without error when record doesn't exist
    await repo.delete("non-existent-uuid")

    # Verify delete was not called
    session_mock.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test__get_returns_single_row_or_raises(session_mock):
    repo = UserRepository(session_mock)
    user_db = UserDBFactory.create()

    # Test successful case
    session_execute_returns(session_mock, scalar_one=user_db)
    result = await repo._get(user_db.id)
    assert result == user_db

    # Test NoResultFound case
    session_execute_returns(session_mock, scalars=[])
    with pytest.raises(NoResultFound):
        await repo._get("non-existent-uuid")


@pytest.mark.asyncio
async def test_get_by_email_returns_user_when_exists(session_mock):
    """Test get_by_email returns UserDTO when user exists"""
    repo = UserRepository(session_mock)
    user_db = UserDBFactory.create(email="exists@example.com")

    # Mock the result to have scalar_one_or_none method
    result_mock = Mock()
    result_mock.scalar_one_or_none.return_value = user_db
    session_mock.execute = AsyncMock(return_value=result_mock)

    result = await repo.get_by_email("exists@example.com")

    assert result is not None
    assert result.email == "exists@example.com"
    assert result.id == user_db.id


@pytest.mark.asyncio
async def test_get_by_email_returns_none_when_not_exists(session_mock):
    """Test get_by_email returns None when user doesn't exist"""
    repo = UserRepository(session_mock)

    # Mock the result to return None
    result_mock = Mock()
    result_mock.scalar_one_or_none.return_value = None
    session_mock.execute = AsyncMock(return_value=result_mock)

    result = await repo.get_by_email("notfound@example.com")

    assert result is None
