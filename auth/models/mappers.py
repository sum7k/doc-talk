"""Domain-specific mapper instances for auth models.

This module creates specific mapper instances for User and Token entities
using the generic mapper from core infrastructure.
"""

from datetime import datetime, timezone
from typing import Any

from auth.models.db import Token, User
from auth.models.domain import CreateTokenDTO, CreateUserDTO, TokenDTO, UserDTO
from core.db.mappers import GenericMapper


def validate_auth_create_dto(create_dto: Any) -> None:
    """Validate Create DTO has required fields for auth domain.

    Args:
        create_dto: DTO to validate

    Raises:
        ValueError: If validation fails
    """

    # Token-specific validation
    if isinstance(create_dto, CreateTokenDTO):
        if not create_dto.user_id:
            raise ValueError("Token user_id is required")
        if not create_dto.expires_at:
            raise ValueError("Token expires_at is required")
        if create_dto.expires_at < datetime.now(timezone.utc):
            raise ValueError("Token expires_at must be in the future")

    # User-specific validation
    elif isinstance(create_dto, CreateUserDTO):
        if not create_dto.email:
            raise ValueError("User email is required")
        if not create_dto.hashed_password:
            raise ValueError("User hashed_password is required")


class AuthGenericMapper(GenericMapper):
    """Auth-specific mapper with domain validation logic."""

    def to_db_new(self, create_dto: Any):
        """Convert Create DTO to new ORM model with auth-specific validation.

        Args:
            create_dto: Create DTO (CreateUserDTO or CreateTokenDTO)

        Returns:
            New ORM model instance

        Raises:
            ValueError: If validation fails
        """
        # Validate before creating
        validate_auth_create_dto(create_dto)
        return super().to_db_new(create_dto)


# Create singleton instances of auth mappers with validation
UserMapper = AuthGenericMapper(UserDTO, User)
TokenMapper = AuthGenericMapper(TokenDTO, Token)
