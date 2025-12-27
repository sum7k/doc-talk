"""Domain-specific mapper instances for document models.

This module creates specific mapper instances for Document, Page, and Chunk entities
using the generic mapper from core infrastructure, with custom handling for nested relationships.
"""

from documents.models.db import Chunk, Document, Page
from documents.models.domain import (
    ChunkDTO,
    CreateChunkDTO,
    CreateDocumentDTO,
    CreatePageDTO,
    DocumentDTO,
    PageDTO,
)
from core.db.mappers import GenericMapper


class ChunkMapper(GenericMapper[ChunkDTO, Chunk]):
    """Mapper for Chunk entity with metadata field mapping."""

    def __init__(self) -> None:
        super().__init__(ChunkDTO, Chunk)

    def from_db(self, model: Chunk) -> ChunkDTO:
        """Convert Chunk ORM model to DTO, handling metadata_ field."""
        if not model:
            raise ValueError("Cannot map None Chunk")

        return ChunkDTO(
            id=model.id,
            page_id=model.page_id,
            chunk_index=model.chunk_index,
            offset_start=model.offset_start,
            offset_end=model.offset_end,
            text_content=model.text_content,
            metadata=model.metadata_,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    def to_db_new(self, create_dto: CreateChunkDTO) -> Chunk:
        """Convert CreateChunkDTO to new Chunk ORM model."""
        if not create_dto:
            raise ValueError("Cannot map None create_dto")

        return Chunk(
            page_id=create_dto.page_id,
            page_index=create_dto.page_index,
            chunk_index=create_dto.chunk_index,
            offset_start=create_dto.offset_start,
            offset_end=create_dto.offset_end,
            text_content=create_dto.text_content,
            metadata_=create_dto.metadata,
        )


class PageMapper(GenericMapper[PageDTO, Page]):
    """Mapper for Page entity with nested chunks and metadata field mapping."""

    def __init__(self) -> None:
        super().__init__(PageDTO, Page)
        self._chunk_mapper = ChunkMapper()

    def from_db(self, model: Page, include_chunks: bool = False) -> PageDTO:
        """Convert Page ORM model to DTO.

        Args:
            model: Page ORM model
            include_chunks: Whether to include nested chunks

        Returns:
            PageDTO instance
        """
        if not model:
            raise ValueError("Cannot map None Page")

        chunks: tuple[ChunkDTO, ...] = ()
        if include_chunks and model.chunks:
            chunks = tuple(self._chunk_mapper.from_db(c) for c in model.chunks)

        return PageDTO(
            id=model.id,
            document_id=model.document_id,
            page_number=model.page_number,
            text_content=model.text_content,
            text_length=model.text_length,
            metadata=model.metadata_,
            created_at=model.created_at,
            updated_at=model.updated_at,
            chunks=chunks,
        )

    def to_db_new(self, create_dto: CreatePageDTO) -> Page:
        """Convert CreatePageDTO to new Page ORM model."""
        if not create_dto:
            raise ValueError("Cannot map None create_dto")

        return Page(
            document_id=create_dto.document_id,
            page_number=create_dto.page_number,
            text_content=create_dto.text_content,
            text_length=create_dto.text_length,
            metadata_=create_dto.metadata,
        )


class DocumentMapper(GenericMapper[DocumentDTO, Document]):
    """Mapper for Document entity with nested pages."""

    def __init__(self) -> None:
        super().__init__(DocumentDTO, Document)
        self._page_mapper = PageMapper()

    def from_db(
        self, model: Document, include_pages: bool = False, include_chunks: bool = False
    ) -> DocumentDTO:
        """Convert Document ORM model to DTO.

        Args:
            model: Document ORM model
            include_pages: Whether to include nested pages
            include_chunks: Whether to include chunks within pages

        Returns:
            DocumentDTO instance
        """
        if not model:
            raise ValueError("Cannot map None Document")

        pages: tuple[PageDTO, ...] = ()
        if include_pages and model.pages:
            pages = tuple(
                self._page_mapper.from_db(p, include_chunks=include_chunks)
                for p in model.pages
            )

        return DocumentDTO(
            id=model.id,
            source_name=model.source_name,
            status=model.status,
            chunk_size=model.chunk_size,
            chunk_overlap=model.chunk_overlap,
            file_path=model.file_path,
            display_title=model.display_title,
            created_at=model.created_at,
            updated_at=model.updated_at,
            pages=pages,
        )

    def to_db_new(self, create_dto: CreateDocumentDTO) -> Document:
        """Convert CreateDocumentDTO to new Document ORM model."""
        if not create_dto:
            raise ValueError("Cannot map None create_dto")

        return Document(
            source_name=create_dto.source_name,
            chunk_size=create_dto.chunk_size,
            chunk_overlap=create_dto.chunk_overlap,
            file_path=create_dto.file_path,
            display_title=create_dto.display_title,
            status=create_dto.status,
        )
