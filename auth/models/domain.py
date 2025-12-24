"""Domain models (DTOs) for the authentication module.

These immutable data transfer objects represent domain entities at different
lifecycle stages, ensuring type safety and clear intent.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime

from auth.models.enums import UserRole

# ============================================================================
# User DTOs
# ============================================================================


@dataclass(frozen=True)
class CreateUserDTO:
    """DTO for creating a new user.

    Contains only the required fields for user creation.
    Password should already be hashed.
    """

    email: str
    hashed_password: str
    role: UserRole = UserRole.MEMBER
    is_active: bool = True


@dataclass(frozen=True)
class UserDTO:
    """DTO for an existing user.

    Represents a complete user entity retrieved from the database.
    All fields are required and immutable.
    """

    id: uuid.UUID
    email: str
    hashed_password: str
    role: UserRole
    is_active: bool
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class UpdateUserDTO:
    """DTO for updating a user.

    Contains only the fields that can be updated.
    All fields are optional - only provided fields will be updated.
    """

    email: str | None = None
    role: UserRole | None = None
    is_active: bool | None = None


# ============================================================================
# Token DTOs
# ============================================================================


@dataclass(frozen=True)
class CreateTokenDTO:
    """DTO for creating a new token.

    Contains only the required fields for token creation.
    """

    user_id: uuid.UUID
    expires_at: datetime
    is_active: bool = True
    ip_address: str | None = None


@dataclass(frozen=True)
class TokenDTO:
    """DTO for an existing token.

    Represents a complete token entity retrieved from the database.
    All fields are required and immutable.
    """

    id: uuid.UUID
    user_id: uuid.UUID
    expires_at: datetime
    is_active: bool
    ip_address: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class UpdateTokenDTO:
    """DTO for updating a token.

    Contains only the fields that can be updated.
    """

    is_active: bool | None = None
