from typing import Any

from opentelemetry import trace
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ai.documents.models.db import Chunk, Document, DocumentStatus, Page
from ai.documents.models.domain import ChunkDTO, DocumentDTO, PageDTO
from ai.documents.models.mappers import ChunkMapper, DocumentMapper, PageMapper
from core.db.repository import Repository


class DocumentRepository(Repository[DocumentDTO, Document]):
    """Repository for Document entities with support for nested relationships."""

    tracer = trace.get_tracer(__name__)

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, Document, DocumentMapper())

    async def get_with_pages(
        self, document_id: str, include_chunks: bool = False
    ) -> DocumentDTO | None:
        """Get a document with its pages loaded.

        Args:
            document_id: Document ID
            include_chunks: Whether to also load chunks for each page

        Returns:
            DocumentDTO with pages (and optionally chunks) or None
        """
        with self.tracer.start_as_current_span(
            "document_repository.get_with_pages",
            attributes={"document.id": document_id, "include_chunks": include_chunks},
        ):
            query = select(Document).where(Document.id == document_id)

            if include_chunks:
                query = query.options(
                    selectinload(Document.pages).selectinload(Page.chunks)
                )
            else:
                query = query.options(selectinload(Document.pages))

            result = await self.session.execute(query)
            document = result.scalar_one_or_none()

            if document:
                mapper: DocumentMapper = self.mapper  # type: ignore
                return mapper.from_db(
                    document, include_pages=True, include_chunks=include_chunks
                )
            return None

    async def get_by_source_name(self, source_name: str) -> DocumentDTO | None:
        """Get a document by its source name.

        Args:
            source_name: The source name (e.g., filename)

        Returns:
            DocumentDTO or None
        """
        with self.tracer.start_as_current_span(
            "document_repository.get_by_source_name",
            attributes={"document.source_name": source_name},
        ):
            query = select(Document).where(Document.source_name == source_name)
            result = await self.session.execute(query)
            document = result.scalar_one_or_none()

            if document:
                dto: DocumentDTO = self.mapper.from_db(document)
                return dto
            return None

    async def update_status(
        self, document_id: str, status: DocumentStatus
    ) -> DocumentDTO:
        """Update the status of a document.

        Args:
            document_id: Document ID
            status: New status

        Returns:
            Updated DocumentDTO
        """
        with self.tracer.start_as_current_span(
            "document_repository.update_status",
            attributes={"document.id": document_id, "document.status": status.value},
        ):
            return await self.update(document_id, {"status": status})


class PageRepository(Repository[PageDTO, Page]):
    """Repository for Page entities."""

    tracer = trace.get_tracer(__name__)

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, Page, PageMapper())

    async def get_by_document(
        self, document_id: str, include_chunks: bool = False
    ) -> list[PageDTO]:
        """Get all pages for a document.

        Args:
            document_id: Document ID
            include_chunks: Whether to load chunks

        Returns:
            List of PageDTO
        """
        with self.tracer.start_as_current_span(
            "page_repository.get_by_document",
            attributes={"document.id": document_id, "include_chunks": include_chunks},
        ):
            query = (
                select(Page)
                .where(Page.document_id == document_id)
                .order_by(Page.page_number)
            )

            if include_chunks:
                query = query.options(selectinload(Page.chunks))

            result = await self.session.execute(query)
            pages = result.scalars().all()

            mapper: PageMapper = self.mapper  # type: ignore
            return [mapper.from_db(p, include_chunks=include_chunks) for p in pages]

    async def get_with_chunks(self, page_id: str) -> PageDTO | None:
        """Get a page with its chunks loaded.

        Args:
            page_id: Page ID

        Returns:
            PageDTO with chunks or None
        """
        with self.tracer.start_as_current_span(
            "page_repository.get_with_chunks",
            attributes={"page.id": page_id},
        ):
            query = (
                select(Page)
                .where(Page.id == page_id)
                .options(selectinload(Page.chunks))
            )
            result = await self.session.execute(query)
            page = result.scalar_one_or_none()

            if page:
                mapper: PageMapper = self.mapper  # type: ignore
                return mapper.from_db(page, include_chunks=True)
            return None

    async def bulk_create(self, pages: list[Any]) -> list[PageDTO]:
        """Create multiple pages in a single transaction.

        Args:
            pages: List of Create DTOs

        Returns:
            List of created PageDTO
        """
        with self.tracer.start_as_current_span(
            "page_repository.bulk_create",
            attributes={"page.count": len(pages)},
        ):
            created = []
            for page_dto in pages:
                entity = self.mapper.to_db_new(page_dto)
                self.session.add(entity)
                await self.session.flush()
                await self.session.refresh(entity)
                created.append(self.mapper.from_db(entity))
            return created


class ChunkRepository(Repository[ChunkDTO, Chunk]):
    """Repository for Chunk entities."""

    tracer = trace.get_tracer(__name__)

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, Chunk, ChunkMapper())

    async def get_by_page(self, page_id: str) -> list[ChunkDTO]:
        """Get all chunks for a page.

        Args:
            page_id: Page ID

        Returns:
            List of ChunkDTO ordered by offset_start
        """
        with self.tracer.start_as_current_span(
            "chunk_repository.get_by_page",
            attributes={"page.id": page_id},
        ):
            query = (
                select(Chunk)
                .where(Chunk.page_id == page_id)
                .order_by(Chunk.offset_start)
            )
            result = await self.session.execute(query)
            chunks = result.scalars().all()

            return [self.mapper.from_db(c) for c in chunks]

    async def bulk_create(self, chunks: list[Any]) -> list[ChunkDTO]:
        """Create multiple chunks in a single transaction.

        Args:
            chunks: List of Create DTOs

        Returns:
            List of created ChunkDTO
        """
        with self.tracer.start_as_current_span(
            "chunk_repository.bulk_create",
            attributes={"chunk.count": len(chunks)},
        ):
            created = []
            for chunk_dto in chunks:
                entity = self.mapper.to_db_new(chunk_dto)
                self.session.add(entity)
                await self.session.flush()
                await self.session.refresh(entity)
                created.append(self.mapper.from_db(entity))
            return created

    async def get_by_document(self, document_id: str) -> list[ChunkDTO]:
        """Get all chunks for a document (across all pages).

        Args:
            document_id: Document ID

        Returns:
            List of ChunkDTO ordered by page number and offset_start
        """
        with self.tracer.start_as_current_span(
            "chunk_repository.get_by_document",
            attributes={"document.id": document_id},
        ):
            query = (
                select(Chunk)
                .join(Page, Chunk.page_id == Page.id)
                .where(Page.document_id == document_id)
                .order_by(Page.page_number, Chunk.offset_start)
            )
            result = await self.session.execute(query)
            chunks = result.scalars().all()

            return [self.mapper.from_db(c) for c in chunks]
