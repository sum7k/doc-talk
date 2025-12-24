"""Generic mapper for converting between database models and domain DTOs.

This module provides a generic, type-safe mapper that automatically handles
conversions using dataclass field introspection, eliminating repetitive code.
"""

from dataclasses import fields, is_dataclass
from typing import Any, Generic, Type, TypeVar

TDto = TypeVar("TDto")
TModel = TypeVar("TModel")


class GenericMapper(Generic[TDto, TModel]):
    """Generic mapper that uses dataclass field introspection for conversions.

    Automatically maps fields between DTOs and ORM models based on matching
    attribute names, eliminating manual field-by-field mapping code.

    Usage:
        # Create a mapper instance
        UserMapper = GenericMapper[UserDTO, User](UserDTO, User)

        # Convert ORM to DTO
        user_dto = UserMapper.from_db(user_model)

        # Create new ORM model
        new_user = UserMapper.to_db_new(create_user_dto)

        # Update existing model
        UserMapper.apply_update(user_model, update_user_dto)
    """

    def __init__(self, dto_class: Type[TDto], model_class: Type[TModel]):
        """Initialize mapper with DTO and Model types.

        Args:
            dto_class: Dataclass type for DTOs
            model_class: SQLAlchemy ORM model type
        """
        self.dto_class = dto_class
        self.model_class = model_class

    def from_db(self, model: TModel) -> TDto:
        """Convert ORM model to DTO by mapping matching field names.

        Args:
            model: SQLAlchemy ORM model instance

        Returns:
            DTO instance with fields populated from model

        Raises:
            ValueError: If model is None
            TypeError: If DTO class is not a dataclass
        """
        if not model:
            raise ValueError(f"Cannot map None {self.model_class.__name__}")

        if not is_dataclass(self.dto_class):
            raise TypeError(f"{self.dto_class.__name__} must be a dataclass")

        # Get all DTO fields and map from model attributes
        dto_fields = {f.name for f in fields(self.dto_class)}
        kwargs = {}

        for field_name in dto_fields:
            if hasattr(model, field_name):
                kwargs[field_name] = getattr(model, field_name)

        return self.dto_class(**kwargs)  # type: ignore[return-value]

    def to_db_new(self, create_dto: Any) -> TModel:
        """Convert Create DTO to new ORM model instance.

        Only maps fields that exist in the Create DTO (not id, timestamps).
        The database will assign id and timestamps automatically.

        Args:
            create_dto: Create DTO (e.g., CreateUserDTO, CreateTokenDTO)

        Returns:
            New ORM model instance without id or timestamps

        Raises:
            ValueError: If DTO is None
            TypeError: If create_dto is not a dataclass
        """
        if not create_dto:
            raise ValueError("Cannot map None create_dto")

        if not is_dataclass(create_dto):
            raise TypeError(f"{type(create_dto).__name__} must be a dataclass")

        # Get all fields from the Create DTO
        dto_fields = {f.name: getattr(create_dto, f.name) for f in fields(create_dto)}

        # Filter out None values and create model
        kwargs = {k: v for k, v in dto_fields.items() if v is not None}

        return self.model_class(**kwargs)

    def apply_update(self, model: TModel, update_dto: Any) -> None:
        """Apply Update DTO fields to existing model.

        Only updates fields that are explicitly set (not None) in the DTO.

        Args:
            model: Existing ORM model to update
            update_dto: Update DTO with optional fields to change

        Raises:
            TypeError: If update_dto is not a dataclass
        """
        if not is_dataclass(update_dto):
            raise TypeError(f"{type(update_dto).__name__} must be a dataclass")

        # Update only non-None fields from the Update DTO
        for field in fields(update_dto):
            value = getattr(update_dto, field.name)
            if value is not None and hasattr(model, field.name):
                setattr(model, field.name, value)
