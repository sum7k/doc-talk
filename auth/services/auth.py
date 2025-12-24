import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession

from auth.models.domain import CreateTokenDTO, CreateUserDTO, UserDTO
from auth.models.mappers import UserMapper
from auth.models.schemas import AuthCreds, TokenBase, TokenOut, UserCreate, UserOut
from auth.repositories.auth import TokenRepository, UserRepository
from core.database import get_db_session
from core.exceptions import AlreadyRegisteredException, UnauthorizedException
from core.settings import JWTConfig, Settings, get_settings

security = HTTPBearer()
AuthHeaderDep = Annotated[HTTPAuthorizationCredentials, Depends(security)]


class PasswordService:
    """Service for password hashing and verification."""

    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    async def verify_password(self, password: str, hashed_password: str) -> bool:
        result: bool = self.pwd_context.verify(password, hashed_password)
        return result

    async def get_password_hash(self, password: str) -> str:
        """Hash a plain password."""
        hashed: str = self.pwd_context.hash(password)
        return hashed


class TokenService:
    """Service for JWT token creation, validation, and management.

    Note: This service is separate from TokenRepository to maintain
    single responsibility principle.
    """

    tracer = trace.get_tracer(__name__)

    def __init__(self, token_repository: TokenRepository, jwt_config: JWTConfig):
        self.token_repository = token_repository
        self.jwt_config = jwt_config
        self.secret_key = jwt_config.secret_key
        self.algorithm = jwt_config.algorithm
        self.default_timedelta = timedelta(
            minutes=jwt_config.access_token_expire_minutes
        )

    async def create_access_token(
        self, data: UserDTO, expires_delta: timedelta | None = None
    ) -> str:
        """Create a new JWT access token for a user."""
        with self.tracer.start_as_current_span("token.create_access_token") as span:
            span.set_attribute("user.id", str(data.id))

            if expires_delta is None:
                expires_delta = self.default_timedelta
            expiry = datetime.now(timezone.utc) + expires_delta
            span.set_attribute(
                "auth.expires_minutes", expires_delta.total_seconds() / 60
            )

            # Create CreateTokenDTO with required fields
            create_token_dto = CreateTokenDTO(
                user_id=data.id,
                expires_at=expiry,
                is_active=True,
            )

            created_token = await self.token_repository.create(create_token_dto)
            token_id = created_token.id

            # Create JWT payload with all claims
            serialized_data = {
                "id": str(data.id),
                "exp": int(expiry.timestamp()),
                "iss": "fastapi-skeleton",
                "sub": str(token_id),
            }
            encoded_jwt: str = jwt.encode(
                serialized_data, self.secret_key, self.algorithm
            )
            return encoded_jwt

    async def deactivate(self, token_id_str: str) -> None:
        """Deactivate a token by ID."""
        token_id = uuid.UUID(token_id_str)
        await self.token_repository.update(token_id, {"is_active": False})

    def decode(self, encoded_token: str) -> dict[str, Any]:
        """Decode a JWT token."""
        try:
            decoded: dict[str, Any] = jwt.decode(
                encoded_token, self.secret_key, algorithms=[self.algorithm]
            )
            return decoded
        except JWTError:
            raise UnauthorizedException

    async def validate(self, token_id_str: str) -> bool:
        """Validate that a token exists and is active."""
        token_id = uuid.UUID(token_id_str)
        token = await self.token_repository.get(token_id)
        return token is not None and token.is_active is True


class AuthService:
    """Authentication service handling user registration, login, and token management."""

    tracer = trace.get_tracer(__name__)

    def __init__(
        self,
        session: AsyncSession,
        password_service: PasswordService,
        token_service: TokenService,
    ):
        self.session = session
        self.password_service = password_service
        self.token_service = token_service
        self.user_repo = UserRepository(session)
        self.user_mapper = UserMapper

    async def register_user(self, user: UserCreate) -> UserOut:
        with self.tracer.start_as_current_span("auth.register") as span:
            if await self.user_repo.get_by_email(user.email):
                span.set_status(Status(StatusCode.ERROR, "User already registered"))
                raise AlreadyRegisteredException
            hashed_password = await self.password_service.get_password_hash(
                user.password
            )
            create_user_dto = CreateUserDTO(
                email=user.email,
                hashed_password=hashed_password,
                role=user.role,
                is_active=user.is_active,
            )
            user_dto = await self.user_repo.create(create_user_dto)
            span.set_status(Status(StatusCode.OK), "user_registered")

            return UserOut(
                email=user_dto.email,
                id=user_dto.id,
                created_at=user_dto.created_at,
                updated_at=user_dto.updated_at,  # type: ignore
            )

    async def authenticate_user(self, login_user: AuthCreds) -> TokenOut:
        with self.tracer.start_as_current_span("auth.authenticate") as span:
            if not (user := await self.user_repo.get_by_email(login_user.email)):
                span.set_status(Status(StatusCode.ERROR, "User not found"))
                raise UnauthorizedException
            if not await self.password_service.verify_password(
                login_user.password,
                user.hashed_password,
            ):
                span.set_status(Status(StatusCode.ERROR, "Invalid credentials"))
                raise UnauthorizedException
            jwt = await self.token_service.create_access_token(user)
            span.set_status(Status(StatusCode.OK), "user_authenticated")
            return TokenOut(access_token=jwt)

    async def get_current_user(self, token: TokenBase) -> UserOut:
        with self.tracer.start_as_current_span("auth.current_user") as span:
            payload = self.token_service.decode(token.access_token)
            if not await self.token_service.validate(str(payload.get("sub"))):
                span.set_status(Status(StatusCode.ERROR, "invalid_token"))
                raise UnauthorizedException
            if not (id := payload.get("id")):
                span.set_status(Status(StatusCode.ERROR, "missing_user_id"))
                raise UnauthorizedException
            if not (user_dto := await self.user_repo.get(id)):
                span.set_status(Status(StatusCode.ERROR, "user_not_found"))
                raise UnauthorizedException
            span.set_status(Status(StatusCode.OK), "user_authenticated")
            return UserOut(
                email=user_dto.email,
                id=user_dto.id,
                role=user_dto.role,
                created_at=user_dto.created_at,
                updated_at=user_dto.updated_at,
            )

    async def logout(self, credentials: HTTPAuthorizationCredentials) -> None:
        payload = self.token_service.decode(credentials.credentials)
        await self.token_service.deactivate(str(payload.get("sub")))


def get_password_service() -> PasswordService:
    """Dependency to get password service instance."""
    return PasswordService()


def get_token_service(
    session: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> TokenService:
    """Dependency to get token service instance."""
    token_repository = TokenRepository(session)
    jwt_config = settings.get_jwt_config()
    return TokenService(token_repository, jwt_config)


def get_auth_service(
    session: AsyncSession = Depends(get_db_session),
    password_service: PasswordService = Depends(get_password_service),
    token_service: TokenService = Depends(get_token_service),
) -> AuthService:
    """Dependency to get auth service instance."""
    return AuthService(session, password_service, token_service)


async def get_current_user_service(
    credentials: AuthHeaderDep,
    auth_service: AuthService = Depends(get_auth_service),
) -> UserOut:
    """Dependency to get current authenticated user."""
    token = TokenBase(access_token=credentials.credentials)
    return await auth_service.get_current_user(token)


AuthServiceDep = Depends(get_auth_service)
CurUserDep = Annotated[UserOut, Depends(get_current_user_service)]
