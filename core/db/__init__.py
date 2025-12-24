"""Database and persistence layer infrastructure.

This module contains generic infrastructure for database operations,
including repositories, mappers, and interfaces that can be reused
across different domains.
"""

from core.db.interfaces import IMapper, IRepository
from core.db.mappers import GenericMapper
from core.db.repository import Repository

__all__ = [
    "IMapper",
    "IRepository",
    "GenericMapper",
    "Repository",
]
