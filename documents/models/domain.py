from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Literal, Optional

from pydantic import Field

from documents.models.db import DocumentStatus
from llm_kit.llms import LLMResponse

# ============================================================================
# Chunk DTOs
# ============================================================================


@dataclass(frozen=True)
class CreateChunkDTO:
    """DTO for creating a new chunk."""

    page_id: str
    page_index: int
    chunk_index: int
    offset_start: int
    text_content: str
    offset_end: int
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ChunkDTO:
    """Immutable DTO for a text chunk."""

    id: str
    page_id: str
    chunk_index: int
    offset_start: int
    offset_end: int
    text_content: str
    metadata: dict[str, Any] | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


# ============================================================================
# Page DTOs
# ============================================================================


@dataclass(frozen=True)
class CreatePageDTO:
    """DTO for creating a new page."""

    document_id: str
    page_number: int
    text_content: str
    text_length: int
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class PageDTO:
    """Immutable DTO for a document page."""

    id: str
    document_id: str
    page_number: int
    text_content: str
    text_length: int
    metadata: dict[str, Any] | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    chunks: tuple[ChunkDTO, ...] = field(default_factory=tuple)


# ============================================================================
# Document DTOs
# ============================================================================


@dataclass(frozen=True)
class CreateDocumentDTO:
    """DTO for creating a new document."""

    source_name: str
    chunk_size: int
    chunk_overlap: int
    file_path: str | None = None
    display_title: str | None = None
    status: DocumentStatus = DocumentStatus.UPLOADED


@dataclass(frozen=True)
class DocumentDTO:
    """Immutable DTO for a document."""

    id: str
    source_name: str
    status: DocumentStatus
    chunk_size: int
    chunk_overlap: int
    file_path: str | None = None
    display_title: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    pages: tuple[PageDTO, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PageSchema:
    """Schema for a single extracted page from a document."""

    page_number: int = Field(
        ..., ge=1, description="1-based page number in the document."
    )

    text: str = Field(
        ..., min_length=1, description="Extracted text content of the page."
    )

    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional extraction metadata (parser, confidence, warnings, etc.).",
    )


@dataclass(frozen=True)
class EmbeddingContext:
    """Context information for embeddings."""

    provider: str
    model: str
    version: str
    namespace: str

@dataclass(frozen=True)
class CitationDTO:
    """Citation information for a document."""
    document_id: str
    title: str | None
    page_number: int
    snippet: str

@dataclass(frozen=True)
class ChatResponseDTO:
    """Result of a document query."""

    citations: list[CitationDTO]
    llm_response: LLMResponse