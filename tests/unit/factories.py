"""Test data factories using factory-boy.

These factories provide a convenient way to generate test data
for models and DTOs with sensible defaults and customizable attributes.
"""

import uuid
from datetime import datetime, timedelta, timezone

import factory
from faker import Faker

from auth.models.db import Token, User
from auth.models.domain import CreateTokenDTO, CreateUserDTO, TokenDTO, UserDTO
from auth.models.enums import UserRole

fake = Faker()


# ============================================================================
# DTO Factories
# ============================================================================


class CreateUserDTOFactory(factory.Factory):
    """Factory for creating CreateUserDTO instances for testing."""

    class Meta:
        model = CreateUserDTO

    email = factory.Faker("email")
    hashed_password = factory.LazyFunction(lambda: fake.password(length=60))
    is_active = True
    role = UserRole.MEMBER

    @classmethod
    def create_admin(cls, **kwargs):
        """Create an admin user create DTO."""
        return cls(role=UserRole.ADMIN, **kwargs)


class UserDTOFactory(factory.Factory):
    """Factory for creating UserDTO instances for testing."""

    class Meta:
        model = UserDTO

    id = factory.LazyFunction(uuid.uuid4)
    email = factory.Faker("email")
    hashed_password = factory.LazyFunction(lambda: fake.password(length=60))
    is_active = True
    role = UserRole.MEMBER
    created_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))
    updated_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))

    @classmethod
    def create_admin(cls, **kwargs):
        """Create an admin user DTO."""
        return cls(role=UserRole.ADMIN, **kwargs)

    @classmethod
    def create_owner(cls, **kwargs):
        """Create an owner user DTO."""
        return cls(role=UserRole.OWNER, **kwargs)


class CreateTokenDTOFactory(factory.Factory):
    """Factory for creating CreateTokenDTO instances for testing."""

    class Meta:
        model = CreateTokenDTO

    user_id = factory.LazyFunction(uuid.uuid4)
    expires_at = factory.LazyFunction(
        lambda: datetime.now(timezone.utc) + timedelta(hours=1)
    )
    is_active = True
    ip_address = factory.Faker("ipv4")


class TokenDTOFactory(factory.Factory):
    """Factory for creating TokenDTO instances for testing."""

    class Meta:
        model = TokenDTO

    id = factory.LazyFunction(uuid.uuid4)
    user_id = factory.LazyFunction(uuid.uuid4)
    expires_at = factory.LazyFunction(
        lambda: datetime.now(timezone.utc) + timedelta(hours=1)
    )
    is_active = True
    ip_address = factory.Faker("ipv4")
    created_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))
    updated_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))

    @classmethod
    def create_expired(cls, **kwargs):
        """Create an expired token DTO."""
        return cls(expires_at=datetime.now(timezone.utc) - timedelta(hours=1), **kwargs)

    @classmethod
    def create_inactive(cls, **kwargs):
        """Create an inactive token DTO."""
        return cls(is_active=False, **kwargs)


# ============================================================================
# Database Model Factories
# ============================================================================


class UserDBFactory(factory.Factory):
    """Factory for creating User database model instances for testing."""

    class Meta:
        model = User

    id = factory.LazyFunction(uuid.uuid4)
    email = factory.Faker("email")
    hashed_password = factory.LazyFunction(lambda: fake.password(length=60))
    is_active = True
    role = UserRole.MEMBER
    created_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))
    updated_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))

    @classmethod
    def create_admin(cls, **kwargs):
        """Create an admin user."""
        return cls(role=UserRole.ADMIN, **kwargs)


class TokenDBFactory(factory.Factory):
    """Factory for creating Token database model instances for testing."""

    class Meta:
        model = Token

    id = factory.LazyFunction(uuid.uuid4)
    user_id = factory.LazyFunction(uuid.uuid4)
    expires_at = factory.LazyFunction(
        lambda: datetime.now(timezone.utc) + timedelta(hours=1)
    )
    is_active = True
    ip_address = factory.Faker("ipv4")
    created_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))
    updated_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))
    user_id = factory.LazyFunction(lambda: uuid.uuid4())
    expires_at = factory.LazyFunction(lambda: fake.future_datetime(end_date="+30d"))
    is_active = True
    ip_address = factory.Faker("ipv4")
