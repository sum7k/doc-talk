"""Custom exception hierarchy for the application.

This module defines a structured exception hierarchy that:
1. Separates domain exceptions from HTTP exceptions
2. Allows for consistent error handling and logging
3. Provides clear error messages and status codes
"""

from typing import Any, Dict, Optional

import structlog
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

logger = structlog.get_logger()


# ============================================================================
# Base Domain Exceptions
# ============================================================================


class DomainException(Exception):
    """Base exception for all domain-level errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


class AuthException(DomainException):
    """Base exception for authentication-related errors."""

    pass


class RepositoryException(DomainException):
    """Base exception for repository/data-layer errors."""

    pass


class ValidationException(DomainException):
    """Base exception for validation errors."""

    pass


# ============================================================================
# Specific Domain Exceptions
# ============================================================================


class UserAlreadyExistsException(AuthException):
    """Raised when attempting to register a user that already exists."""

    def __init__(self, email: str):
        super().__init__(
            message=f"User with email {email} already exists", details={"email": email}
        )


class InvalidCredentialsException(AuthException):
    """Raised when authentication credentials are invalid."""

    def __init__(self):
        super().__init__(message="Invalid email or password")


class TokenExpiredException(AuthException):
    """Raised when a token has expired."""

    def __init__(self):
        super().__init__(message="Token has expired")


class TokenInvalidException(AuthException):
    """Raised when a token is invalid or malformed."""

    def __init__(self):
        super().__init__(message="Invalid or malformed token")


class InsufficientPermissionsException(AuthException):
    """Raised when user lacks required permissions."""

    def __init__(self, required_role: str):
        super().__init__(
            message=f"Insufficient permissions. Required role: {required_role}",
            details={"required_role": required_role},
        )


class ResourceNotFoundException(RepositoryException):
    """Raised when a requested resource is not found."""

    def __init__(self, resource_type: str, resource_id: Any):
        super().__init__(
            message=f"{resource_type} with id {resource_id} not found",
            details={"resource_type": resource_type, "resource_id": str(resource_id)},
        )


class ReadinessError(Exception):
    """Raised when service readiness checks fail."""

    pass


# ============================================================================
# HTTP Exception Factory
# ============================================================================


def domain_exception_to_http(
    exc: DomainException, correlation_id: Optional[str] = None
) -> HTTPException:
    """Convert domain exceptions to HTTP exceptions with appropriate status codes."""

    status_map = {
        UserAlreadyExistsException: status.HTTP_400_BAD_REQUEST,
        InvalidCredentialsException: status.HTTP_401_UNAUTHORIZED,
        TokenExpiredException: status.HTTP_401_UNAUTHORIZED,
        TokenInvalidException: status.HTTP_401_UNAUTHORIZED,
        InsufficientPermissionsException: status.HTTP_403_FORBIDDEN,
        ResourceNotFoundException: status.HTTP_404_NOT_FOUND,
        ValidationException: status.HTTP_422_UNPROCESSABLE_ENTITY,
    }

    status_code = status_map.get(type(exc), status.HTTP_500_INTERNAL_SERVER_ERROR)

    headers = {}
    if isinstance(
        exc, (InvalidCredentialsException, TokenExpiredException, TokenInvalidException)
    ):
        headers["WWW-Authenticate"] = "Bearer"

    if correlation_id:
        headers["X-Correlation-ID"] = correlation_id

    return HTTPException(
        status_code=status_code,
        detail={
            "message": exc.message,
            "details": exc.details,
            "correlation_id": correlation_id,
        },
        headers=headers if headers else None,
    )


# ============================================================================
# Global Exception Handlers
# ============================================================================


async def domain_exception_handler(
    request: Request, exc: DomainException
) -> JSONResponse:
    """Global handler for domain exceptions."""
    correlation_id = request.headers.get("X-Correlation-ID")

    logger.error(
        "domain_exception",
        exception_type=type(exc).__name__,
        message=exc.message,
        details=exc.details,
        path=request.url.path,
    )

    http_exc = domain_exception_to_http(exc, correlation_id)

    return JSONResponse(
        status_code=http_exc.status_code,
        content=http_exc.detail,
        headers=http_exc.headers,
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global handler for unexpected exceptions."""
    correlation_id = request.headers.get("X-Correlation-ID")

    logger.exception(
        "unhandled_exception",
        exception_type=type(exc).__name__,
        path=request.url.path,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "message": "An unexpected error occurred",
            "correlation_id": correlation_id,
        },
        headers={"X-Correlation-ID": correlation_id} if correlation_id else None,
    )


# ============================================================================
# Backward Compatibility (deprecated - use domain exceptions instead)
# ============================================================================


UnauthorizedException = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not authenticated",
    headers={"WWW-Authenticate": "Bearer"},
)

AlreadyRegisteredException = HTTPException(
    status_code=status.HTTP_400_BAD_REQUEST,
    detail="Username already registered",
)
