from typing import Any, Generic, Type, TypeVar

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from sqlalchemy import select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.ext.asyncio import AsyncSession

from core.db.interfaces import IMapper, IRepository

TDto = TypeVar("TDto")
TModel = TypeVar("TModel")


class Repository(IRepository[TDto], Generic[TDto, TModel]):
    """Generic repository implementation for CRUD operations."""

    tracer = trace.get_tracer(__name__)

    def __init__(
        self,
        session: AsyncSession,
        model_class: Type[TModel],
        mapper: IMapper,
    ):
        self.session = session
        self._model_class = model_class
        self.mapper = mapper

    async def list(self, skip: int, take: int) -> list[TDto]:
        with self.tracer.start_as_current_span(
            "repository.list",
            attributes={
                "db.model": self._model_class.__name__,
                "db.skip": skip,
                "db.limit": take,
            },
        ) as span:
            result = await self.session.execute(
                select(self._model_class).offset(skip).limit(take)
            )
            items = [self.mapper.from_db(r) for r in result.scalars().all()]
            span.set_attribute("db.result_count", len(items))
            return items

    async def get(self, uid: Any) -> TDto:
        with self.tracer.start_as_current_span(
            "repository.get",
            attributes={
                "db.model": self._model_class.__name__,
                "db.id": str(uid),
            },
        ) as span:
            try:
                entity = await self._get(uid)
                return self.mapper.from_db(entity)  # type: ignore[no-any-return]
            except NoResultFound as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, "Entity not found"))
                raise ValueError(f"No {self._model_class.__name__} found with id:{uid}")

    async def create(self, record: Any) -> TDto:
        """Create a new entity from a Create DTO.

        Args:
            record: Create DTO (e.g., CreateUserDTO, CreateTokenDTO)

        Returns:
            Full DTO of the created entity
        """
        with self.tracer.start_as_current_span(
            "repository.create",
            attributes={
                "db.model": self._model_class.__name__,
            },
        ) as span:
            try:
                entity = self.mapper.to_db_new(record)
                self.session.add(entity)
                await self.session.flush()  # Flush to get the ID
                await self.session.refresh(entity)
                span.set_attribute("db.created_id", str(entity.id))  # type: ignore
                created_dto = self.mapper.from_db(entity)
                return created_dto  # type: ignore[no-any-return]
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, "Create failed"))
                raise

    async def update(self, uid: Any, attrs: dict[str, Any]) -> TDto:
        with self.tracer.start_as_current_span(
            "repository.update",
            attributes={
                "db.model": self._model_class.__name__,
                "db.id": str(uid),
                "db.update_fields": list(attrs.keys()),
            },
        ) as span:
            try:
                entity = await self._get(uid)
            except NoResultFound as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, "Entity not found"))
                raise ValueError(f"{self._model_class.__name__} not found.")

            for key, value in attrs.items():
                if hasattr(entity, key):  # Only update fields that exist on ORM
                    setattr(entity, key, value)

            try:
                await self.session.flush()
                await self.session.refresh(entity)
                return self.mapper.from_db(entity)  # type: ignore[no-any-return]
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, "Update failed"))
                raise

    async def delete(self, uid: int) -> None:
        with self.tracer.start_as_current_span(
            "repository.delete",
            attributes={
                "db.model": self._model_class.__name__,
                "db.id": str(uid),
            },
        ) as span:
            try:
                entity = await self._get(uid)
            except NoResultFound:
                span.set_attribute("db.deleted", False)
                return  # No-op if record doesn't exist

            try:
                await self.session.delete(entity)
                await self.session.flush()
                span.set_attribute("db.deleted", True)
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, "Delete failed"))
                raise

    async def _get(self, entity_id: Any) -> TModel:
        result = await self.session.execute(
            select(self._model_class).where(self._model_class.id == entity_id)  # type: ignore
        )
        one: TModel = result.scalar_one()
        return one
