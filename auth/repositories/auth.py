from opentelemetry import trace
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth.models.db import Token, User
from auth.models.domain import TokenDTO, UserDTO
from auth.models.mappers import TokenMapper, UserMapper
from core.db.repository import Repository


class TokenRepository(Repository[TokenDTO, Token]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, Token, TokenMapper)


class UserRepository(Repository[UserDTO, User]):
    tracer = trace.get_tracer(__name__)

    def __init__(self, session: AsyncSession):
        super().__init__(session, User, UserMapper)

    async def get_by_email(self, email: str) -> UserDTO | None:
        with self.tracer.start_as_current_span("user_repository.get_by_email") as span:
            span.set_attribute("user.email", email)
            query = await self.session.execute(select(User).where(User.email == email))
            user = query.scalar_one_or_none()
            if user:
                return self.mapper.from_db(user)  # type: ignore[no-any-return]
            return None
