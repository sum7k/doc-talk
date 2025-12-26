import asyncio
from hashlib import sha256

import structlog
from llm_kit import EmbeddingsClient
from llm_kit.chunking.chunking import chunk_text
from llm_kit.vectorstores.base import VectorStore
from llm_kit.vectorstores.types import VectorItem
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from ai.documents.models.db import DocumentStatus
from ai.documents.models.domain import (
    ChunkDTO,
    CreateChunkDTO,
    CreateDocumentDTO,
    CreatePageDTO,
    DocumentDTO,
    EmbeddingContext,
    PageDTO,
    PageSchema,
)
from ai.documents.models.schemas import ChunkProfile, FileData
from ai.documents.parsers.factory import ParserFactory
from ai.documents.repositories.document import (
    ChunkRepository,
    DocumentRepository,
    PageRepository,
)

logger = structlog.get_logger(__name__)


class DocumentIngestionService:
    """Service for ingesting documents into the vector store.

    Handles the complete ingestion pipeline:
    1. Save document metadata
    2. Parse document into pages
    3. Chunk pages into smaller segments
    4. Generate embeddings for chunks
    5. Store vectors in the vector store
    """

    tracer = trace.get_tracer(__name__)

    def __init__(
        self,
        embedding_client: EmbeddingsClient,
        embedding_context: EmbeddingContext,
        vector_store: VectorStore,
        document_store: DocumentRepository,
        page_store: PageRepository,
        chunk_store: ChunkRepository,
        parser_factory: ParserFactory = ParserFactory(),
    ) -> None:
        self._embedding_client = embedding_client
        self._embedding_context = embedding_context
        self._vector_store = vector_store
        self._document_store = document_store
        self._page_store = page_store
        self._chunk_store = chunk_store
        self._parser_factory = parser_factory

    async def ingest(self, file_data: FileData, chunk_profile: ChunkProfile) -> str:
        """Ingest a document into the system.

        Args:
            file_data: File data containing filename and binary content.
            chunk_profile: Profile containing chunk length and overlap settings.

        Returns:
            The document ID of the ingested document.

        Raises:
            Exception: If ingestion fails at any stage.
        """
        content_size = len(file_data.binary_content) if file_data.binary_content else 0

        with self.tracer.start_as_current_span("ingestion.ingest") as span:
            span.set_attribute("document.source_name", file_data.file_name)
            span.set_attribute("document.content_size", content_size)

            logger.info(
                "ingestion_started",
                source_name=file_data.file_name,
                content_size=content_size,
            )

            document: DocumentDTO | None = None
            try:
                document = await self._save_document(file_data, chunk_profile)
                span.set_attribute("document.id", document.id)

                page_chunks = await self._process_document(
                    document, file_data, chunk_profile
                )

                total_chunks = len(page_chunks)
                page_count = len(set(page.id for page, _ in page_chunks))
                span.set_attribute("document.page_count", page_count)
                span.set_attribute("document.chunk_count", total_chunks)

                await self._generate_and_store_embeddings(document, page_chunks)

                await self._mark_document_ready(document)
                span.set_status(Status(StatusCode.OK))

                logger.info(
                    "ingestion_completed",
                    document_id=document.id,
                    page_count=page_count,
                    chunk_count=total_chunks,
                )
                return document.id

            except Exception as e:
                await self._handle_ingestion_failure(span, document, file_data, e)
                raise

    async def _process_document(
        self, document: DocumentDTO, file_data: FileData, chunk_profile: ChunkProfile
    ) -> list[tuple[PageDTO, ChunkDTO]]:
        """Parse document and create chunks.

        Returns:
            List of (page, chunk) pairs.
        """
        parser = self._parser_factory.get_parser(file_data.file_type)
        pages: list[PageSchema] = await asyncio.to_thread(
            parser.parse, file_data.binary_content, file_data.file_name
        )
        page_dtos = await self._save_pages(document, pages)

        logger.info(
            "pages_parsed",
            document_id=document.id,
            page_count=len(pages),
        )

        page_chunks: list[tuple[PageDTO, ChunkDTO]] = []
        for page_dto in page_dtos:
            chunks = await self._save_chunks(page_dto, chunk_profile)
            page_chunks.extend((page_dto, chunk) for chunk in chunks)

        logger.info(
            "chunks_created",
            document_id=document.id,
            total_chunks=len(page_chunks),
        )

        return page_chunks

    async def _generate_and_store_embeddings(
        self,
        document: DocumentDTO,
        page_chunks: list[tuple[PageDTO, ChunkDTO]],
    ) -> None:
        """Generate embeddings and store in vector store."""
        texts = [chunk.text_content for _, chunk in page_chunks]

        embeddings = await asyncio.to_thread(self._embedding_client.embed, texts)

        vectors = [
            VectorItem(
                id=chunk.id,
                vector=embedding.vector,
                metadata=self._build_chunk_metadata(self._embedding_context, document, page, chunk),
            )
            for embedding, (page, chunk) in zip(embeddings, page_chunks)
        ]

        await asyncio.to_thread(
            self._vector_store.upsert,
            items=vectors,
            namespace=self._embedding_context.namespace,
        )

        logger.info(
            "embeddings_stored",
            document_id=document.id,
            vector_count=len(vectors),
        )

    @staticmethod
    def _build_chunk_metadata(
        embedding_context: EmbeddingContext,
        document: DocumentDTO,
        page: PageDTO,
        chunk: ChunkDTO,
    ) -> dict:
        """Build metadata dictionary for a chunk."""
        return {
            "embedding_model_name": embedding_context.model,
            "embedding_provider": embedding_context.provider,
            "embedding_version": embedding_context.version,
            "embedding_namespace": embedding_context.namespace,
            "document_id": document.id,
            "page_id": page.id,
            "chunk_id": chunk.id,
            "page_number": page.page_number,
            "chunk_index": chunk.chunk_index,
            "chunk_offset_start": chunk.offset_start,
            "chunk_offset_end": chunk.offset_end,
            "chunk_content_hash": sha256(chunk.text_content.encode("utf-8")).hexdigest(),
        }

    async def _mark_document_ready(self, document: DocumentDTO) -> None:
        """Mark document as ready for querying."""
        await self._document_store.update(document.id, {"status": DocumentStatus.READY})

    async def _handle_ingestion_failure(
        self,
        span,
        document: DocumentDTO | None,
        file_data: FileData,
        error: Exception,
    ) -> None:
        """Handle ingestion failure by logging and updating document status."""
        span.record_exception(error)
        span.set_status(Status(StatusCode.ERROR, "Ingestion failed"))

        logger.error(
            "ingestion_failed",
            filename=file_data.file_name,
            document_id=document.id if document else None,
            error=str(error),
            exc_info=True,
        )

        if document is not None:
            await self._document_store.update(
                document.id, {"status": DocumentStatus.FAILED}
            )

    async def _save_chunks(
        self, page: PageDTO, chunk_profile: ChunkProfile
    ) -> list[ChunkDTO]:
        """Chunk page text and save to database."""
        with self.tracer.start_as_current_span("ingestion.save_chunks") as span:
            span.set_attribute("page.id", page.id)
            span.set_attribute("page.page_number", page.page_number)

            raw_chunks = chunk_text(
                page.text_content,
                chunk_size=chunk_profile.chunk_length,
                overlap=chunk_profile.overlap,
                metadata=page.metadata or {},
            )
            span.set_attribute("page.chunk_count", len(raw_chunks))

            chunk_dtos: list[ChunkDTO] = []
            for chunk_index, chunk in enumerate(raw_chunks):
                create_dto = CreateChunkDTO(
                    page_id=page.id,
                    page_index=page.page_number,
                    chunk_index=chunk_index,
                    text_content=chunk.text,
                    metadata=chunk.metadata,
                    offset_start=chunk.offset_start,
                    offset_end=chunk.offset_end,
                )
                chunk_dto = await self._chunk_store.create(create_dto)
                chunk_dtos.append(chunk_dto)

            return chunk_dtos

    async def _save_pages(
        self, document: DocumentDTO, pages: list[PageSchema]
    ) -> list[PageDTO]:
        """Save parsed pages to database."""
        with self.tracer.start_as_current_span("ingestion.save_pages") as span:
            span.set_attribute("document.id", document.id)
            span.set_attribute("pages.count", len(pages))

            page_dtos: list[PageDTO] = []
            for page in pages:
                create_dto = CreatePageDTO(
                    document_id=document.id,
                    page_number=page.page_number,
                    text_content=page.text,
                    text_length=len(page.text),
                    metadata=page.metadata,
                )
                page_dto = await self._page_store.create(create_dto)
                page_dtos.append(page_dto)

            return page_dtos

    async def _save_document(
        self, file_data: FileData, chunk_profile: ChunkProfile
    ) -> DocumentDTO:
        """Create document record in database."""
        with self.tracer.start_as_current_span("ingestion.save_document") as span:
            span.set_attribute("document.source_name", file_data.file_name)

            create_dto = CreateDocumentDTO(
                source_name=file_data.file_name,
                display_title=file_data.file_name,
                binary_content=file_data.binary_content,
                chunk_size=chunk_profile.chunk_length,
                chunk_overlap=chunk_profile.overlap,
                status=DocumentStatus.INGESTING,
            )
            document = await self._document_store.create(create_dto)
            span.set_attribute("document.id", document.id)

            logger.info(
                "document_created",
                document_id=document.id,
                source_name=file_data.file_name,
            )
            return document
