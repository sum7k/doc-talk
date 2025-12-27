from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel


class FileType(str, Enum):
    HTML = "html"
    PDF = "pdf"
    DOCX = "docx"
    MD = "md"
    TEXT = "text"


class FileData(BaseModel):
    file_name: str
    file_type: FileType
    binary_content: bytes


class ChunkingStrategy(Enum):
    DEFAULT = "default"
    DENSE_TEXT = "dense_text"
    QA = "qa"


@dataclass(frozen=True)
class ChunkProfile:
    """Configuration for chunking strategy."""

    strategy: ChunkingStrategy
    chunk_length: int
    overlap: int


CHUNKING_PROFILES: dict[ChunkingStrategy, ChunkProfile] = {
    ChunkingStrategy.DEFAULT: ChunkProfile(
        strategy=ChunkingStrategy.DEFAULT, chunk_length=500, overlap=50
    ),
    ChunkingStrategy.DENSE_TEXT: ChunkProfile(
        strategy=ChunkingStrategy.DENSE_TEXT, chunk_length=800, overlap=150
    ),
    ChunkingStrategy.QA: ChunkProfile(
        strategy=ChunkingStrategy.QA, chunk_length=300, overlap=20
    ),
}
