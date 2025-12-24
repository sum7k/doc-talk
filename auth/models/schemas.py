import re
from datetime import datetime

from pydantic import UUID4, BaseModel, ConfigDict, field_validator

from auth.models.enums import UserRole


class UserBase(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    email: str
    is_active: bool = True
    role: UserRole = UserRole.VIEWER

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(email_pattern, v):
            raise ValueError("Email must be a valid email address")
        return v.lower()


class UserCreate(UserBase):
    password: str

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 4:
            raise ValueError("Password must be at least 4 characters long")

        has_alpha = any(c.isalpha() for c in v)
        has_digit = any(c.isdigit() for c in v)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in v)

        if not has_alpha:
            raise ValueError("Password must contain at least one alphabetic character")
        if not has_digit:
            raise ValueError("Password must contain at least one numeric character")
        if not has_special:
            raise ValueError(
                "Password must contain at least one special character (!@#$%^&*()_+-=[]{}|;:,.<>?)"
            )

        return v


class UserInDB(UserBase):
    hashed_password: str


class UserOut(UserBase):
    id: UUID4
    created_at: datetime
    updated_at: datetime


class TokenBase(BaseModel):
    access_token: str
    token_type: str = "Bearer"


class TokenCreate(TokenBase):
    ip_address: str | None = None


class TokenOut(TokenBase):
    pass


class AuthCreds(BaseModel):
    email: str
    password: str


class MessageResponse(BaseModel):
    """Generic message response schema."""

    message: str

    model_config = ConfigDict(
        json_schema_extra={"example": {"message": "Operation completed successfully"}}
    )
