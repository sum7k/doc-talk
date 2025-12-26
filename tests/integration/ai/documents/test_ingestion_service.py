# tests/integration/test_ingestion_service.py
import os
import time

import pytest
import pytest_asyncio
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from testcontainers.postgres import PostgresContainer

from ai.documents.models.schemas import CHUNKING_PROFILES, ChunkingStrategy, ChunkProfile
from ai.documents.repositories.document import (
    ChunkRepository,
    DocumentRepository,
    PageRepository,
)
from ai.documents.services.ingestion import DocumentIngestionService
from core.database import Base

# Ensure Docker socket is accessible (needed for Colima and other non-standard Docker setups)
if not os.environ.get("DOCKER_HOST"):
    # Try common locations
    for socket_path in [
        "/var/run/docker.sock",
        os.path.expanduser("~/.colima/default/docker.sock"),
        os.path.expanduser("~/.docker/run/docker.sock"),
    ]:
        if os.path.exists(socket_path):
            os.environ["DOCKER_HOST"] = f"unix://{socket_path}"
            break

# Disable Ryuk container for Colima compatibility
# Ryuk requires mounting Docker socket which doesn't work well with Colima
os.environ["TESTCONTAINERS_RYUK_DISABLED"] = "true"

pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest.fixture(scope="module")
def postgres_container():
    """Spin up a PostgreSQL container with pgvector extension."""
    container = PostgresContainer("pgvector/pgvector:pg16")
    container.with_env("POSTGRES_DB", "test_db")
    container.with_env("POSTGRES_USER", "test_user")
    container.with_env("POSTGRES_PASSWORD", "test_password")

    with container:
        # Wait for PostgreSQL to be ready
        container.get_connection_url()  # This will wait for the container
        yield container


@pytest.fixture(scope="module")
def sync_url(postgres_container):
    """Get sync connection URL for pgvector store."""
    url = postgres_container.get_connection_url()
    # Convert to plain postgresql:// for psycopg
    return url.replace("postgresql+psycopg2://", "postgresql://")


@pytest.fixture(scope="module")
def _init_vector_schema(sync_url):
    """Initialize pgvector extension and vector_items table."""
    import psycopg

    # Retry connection a few times to wait for container
    max_retries = 10
    for attempt in range(max_retries):
        try:
            with psycopg.connect(sync_url) as conn:
                conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS vector_items (
                        namespace TEXT NOT NULL,
                        id TEXT NOT NULL,
                        embedding VECTOR(16) NOT NULL,
                        metadata JSONB NOT NULL,
                        PRIMARY KEY (namespace, id)
                    );
                """
                )
                conn.commit()
            break
        except psycopg.OperationalError:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                raise


@pytest_asyncio.fixture(scope="module")
async def async_engine(postgres_container, _init_vector_schema):
    """Create async engine connected to test container."""
    # Convert sync URL to async - testcontainers returns psycopg2 URL
    sync_url = postgres_container.get_connection_url()
    # Replace both possible sync drivers with asyncpg
    async_url = sync_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://")
    async_url = async_url.replace("postgresql://", "postgresql+asyncpg://")

    # Give PostgreSQL a moment to finish starting up
    time.sleep(2)

    engine = create_async_engine(async_url, echo=True)

    # Retry connection a few times if needed
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Create tables
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            break  # Success
        except (OperationalError, OSError):
            if attempt < max_retries - 1:
                print(f"Connection attempt {attempt + 1} failed, retrying...")
                time.sleep(1)
            else:
                raise  # Final attempt failed

    yield engine

    await engine.dispose()


@pytest.fixture(scope="module")
def session_maker(async_engine):
    """Create session maker bound to the async engine."""
    return async_sessionmaker(async_engine, class_=AsyncSession)


@pytest.fixture
def vector_store(sync_url, _init_vector_schema):
    """Create PgVectorStore connected to test container."""
    from llm_kit.vectorstores.pgvectorstore import PgVectorStore

    store = PgVectorStore(sync_url)
    yield store

    # Clean up vectors after each test
    import psycopg

    with psycopg.connect(sync_url) as conn:
        conn.execute("DELETE FROM vector_items WHERE namespace = 'doc-talk';")
        conn.commit()

    store.close()


@pytest.fixture
def ingestion_service(session_maker, fake_embedding_client, vector_store):
    """Create ingestion service with real DB repositories and real PDF parser."""
    from ai.documents.models.domain import EmbeddingContext

    # Create a new session for this test
    session = session_maker()

    return DocumentIngestionService(
        embedding_client=fake_embedding_client,
        embedding_context=EmbeddingContext(
            provider="fake",
            model="fake-model",
            version="v1",
            namespace="doc-talk",
        ),
        vector_store=vector_store,
        document_store=DocumentRepository(session),
        page_store=PageRepository(session),
        chunk_store=ChunkRepository(session),
    )


# Fake embedding client for testing (produces deterministic embeddings based on text hash)
class FakeEmbeddingsClient:
    def __init__(self, dim: int = 16):
        self.dim = dim
        self.call_count = 0
        self.texts_embedded: list[str] = []

    def embed(self, texts: list[str]):
        import hashlib

        import numpy as np

        self.call_count += 1
        self.texts_embedded.extend(texts)

        embeddings = []
        for text in texts:
            h = hashlib.sha256(text.encode()).digest()
            vec = np.frombuffer(h, dtype=np.uint8)[: self.dim].astype(float)
            vec = vec / np.linalg.norm(vec)
            embeddings.append(type("Embedding", (), {"vector": vec.tolist()})())
        return embeddings


@pytest.fixture
def fake_embedding_client():
    """Fake embeddings client that produces deterministic embeddings."""
    return FakeEmbeddingsClient(dim=16)


@pytest.fixture
def default_chunk_profile():
    """Default chunk profile for tests."""
    return CHUNKING_PROFILES[ChunkingStrategy.DEFAULT]


@pytest.fixture
def sample_file_data():
    """Load real PDF file data for testing."""
    from pathlib import Path

    from ai.documents.models.schemas import FileData, FileType

    # Load real PDF from fixtures
    fixtures_dir = Path(__file__).parent.parent.parent.parent / "fixtures"
    pdf_content = (fixtures_dir / "sample.pdf").read_bytes()

    return FileData(
        file_name="sample.pdf",
        file_type=FileType.PDF,
        binary_content=pdf_content,
    )


# =============================================================================
# Focused Integration Tests
# =============================================================================


# --- Chunk Persistence Tests ---


async def test_chunks_are_persisted_to_database(ingestion_service, sample_file_data, default_chunk_profile):
    """Verify chunks are saved with correct content and offsets."""
    document_id = await ingestion_service.ingest(sample_file_data, default_chunk_profile)

    chunks = await ingestion_service._chunk_store.get_by_document(document_id)

    assert len(chunks) == 2  # One chunk per page (text fits in single chunk)
    assert all(chunk.text_content for chunk in chunks)
    assert all(chunk.offset_start == 0 for chunk in chunks)
    assert all(chunk.offset_end > 0 for chunk in chunks)


async def test_chunks_linked_to_correct_pages(ingestion_service, sample_file_data, default_chunk_profile):
    """Verify each chunk is linked to its source page."""
    document_id = await ingestion_service.ingest(sample_file_data, default_chunk_profile)

    pages = await ingestion_service._page_store.get_by_document(document_id)
    chunks = await ingestion_service._chunk_store.get_by_document(document_id)

    page_ids = {p.id for p in pages}
    chunk_page_ids = {c.page_id for c in chunks}

    assert chunk_page_ids.issubset(page_ids)


# --- Embedding Generation Tests ---


async def test_embeddings_generated_once_per_chunk(
    ingestion_service, sample_file_data, fake_embedding_client, default_chunk_profile
):
    """Verify embed() is called exactly once with all chunk texts."""
    document_id = await ingestion_service.ingest(sample_file_data, default_chunk_profile)

    chunks = await ingestion_service._chunk_store.get_by_document(document_id)

    # Embedding client should be called once (batched)
    assert fake_embedding_client.call_count == 1

    # All chunk texts should have been embedded
    assert len(fake_embedding_client.texts_embedded) == len(chunks)


async def test_embedding_texts_match_chunk_content(
    ingestion_service, sample_file_data, fake_embedding_client, default_chunk_profile
):
    """Verify the texts sent to embedding match chunk content."""
    document_id = await ingestion_service.ingest(sample_file_data, default_chunk_profile)

    chunks = await ingestion_service._chunk_store.get_by_document(document_id)
    chunk_texts = {c.text_content for c in chunks}
    embedded_texts = set(fake_embedding_client.texts_embedded)

    assert chunk_texts == embedded_texts


# --- Vector Storage Tests ---


async def test_vectors_stored_with_chunk_ids(
    ingestion_service, sample_file_data, vector_store, fake_embedding_client, default_chunk_profile
):
    """Verify vector IDs match chunk IDs."""
    document_id = await ingestion_service.ingest(sample_file_data, default_chunk_profile)

    chunks = await ingestion_service._chunk_store.get_by_document(document_id)
    query_vec = fake_embedding_client.embed([chunks[0].text_content])[0].vector

    results = vector_store.query(namespace="doc-talk", vector=query_vec, top_k=10)

    result_ids = {r.id for r in results}
    chunk_ids = {c.id for c in chunks}

    assert result_ids == chunk_ids


async def test_vectors_stored_with_correct_metadata(
    ingestion_service, sample_file_data, vector_store, fake_embedding_client, default_chunk_profile
):
    """Verify vector metadata contains document_id, page_id, page_number."""
    document_id = await ingestion_service.ingest(sample_file_data, default_chunk_profile)

    chunks = await ingestion_service._chunk_store.get_by_document(document_id)
    query_vec = fake_embedding_client.embed([chunks[0].text_content])[0].vector

    results = vector_store.query(namespace="doc-talk", vector=query_vec, top_k=10)

    for result in results:
        assert result.metadata["document_id"] == document_id
        assert "page_id" in result.metadata
        assert "page_number" in result.metadata
        assert "chunk_offset_start" in result.metadata
        assert "chunk_offset_end" in result.metadata


# --- Vector Retrieval Tests ---


async def test_retrieval_returns_results(
    ingestion_service, sample_file_data, vector_store, fake_embedding_client, default_chunk_profile
):
    """Verify querying returns stored vectors."""
    await ingestion_service.ingest(sample_file_data, default_chunk_profile)

    # Query with embedding of known text
    query_vec = fake_embedding_client.embed(["Sample PDF Document"])[0].vector

    results = vector_store.query(namespace="doc-talk", vector=query_vec, top_k=5)

    assert len(results) > 0


async def test_retrieval_scores_are_valid(
    ingestion_service, sample_file_data, vector_store, fake_embedding_client, default_chunk_profile
):
    """Verify retrieval scores are between 0 and 1 (cosine similarity)."""
    document_id = await ingestion_service.ingest(sample_file_data, default_chunk_profile)

    chunks = await ingestion_service._chunk_store.get_by_document(document_id)
    # Query with exact chunk text should give high score
    query_vec = fake_embedding_client.embed([chunks[0].text_content])[0].vector

    results = vector_store.query(namespace="doc-talk", vector=query_vec, top_k=5)

    for result in results:
        assert 0 <= result.score <= 1

    # First result (exact match) should have score close to 1
    assert results[0].score > 0.99


# --- Pipeline Boundary Tests ---


async def test_document_status_transitions_to_ready(
    ingestion_service, sample_file_data, default_chunk_profile
):
    """Verify document status changes to READY after successful ingestion."""
    from ai.documents.models.db import DocumentStatus

    document_id = await ingestion_service.ingest(sample_file_data, default_chunk_profile)

    document = await ingestion_service._document_store.get(document_id)

    assert document.status == DocumentStatus.READY


async def test_page_count_matches_pdf(ingestion_service, sample_file_data, default_chunk_profile):
    """Verify number of pages matches the actual PDF structure."""
    document_id = await ingestion_service.ingest(sample_file_data, default_chunk_profile)

    pages = await ingestion_service._page_store.get_by_document(document_id)

    # Our sample.pdf has 2 pages
    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert pages[1].page_number == 2


async def test_chunk_count_matches_vector_count(
    ingestion_service, sample_file_data, vector_store, fake_embedding_client, default_chunk_profile
):
    """Verify every chunk has a corresponding vector."""
    document_id = await ingestion_service.ingest(sample_file_data, default_chunk_profile)

    chunks = await ingestion_service._chunk_store.get_by_document(document_id)
    query_vec = fake_embedding_client.embed([chunks[0].text_content])[0].vector

    results = vector_store.query(namespace="doc-talk", vector=query_vec, top_k=100)

    assert len(results) == len(chunks)


# --- Chunk Metadata Integrity Tests ---


async def test_content_hash_stored_correctly(
    ingestion_service, sample_file_data, vector_store, fake_embedding_client, default_chunk_profile
):
    """Verify chunk content hash in vector metadata matches computed hash from chunk text."""
    from hashlib import sha256

    document_id = await ingestion_service.ingest(sample_file_data, default_chunk_profile)

    chunks = await ingestion_service._chunk_store.get_by_document(document_id)
    query_vec = fake_embedding_client.embed([chunks[0].text_content])[0].vector
    results = vector_store.query(namespace="doc-talk", vector=query_vec, top_k=100)

    # Build a map of chunk_id -> chunk for easy lookup
    chunk_by_id = {c.id: c for c in chunks}

    for result in results:
        chunk = chunk_by_id[result.id]
        expected_hash = sha256(chunk.text_content.encode("utf-8")).hexdigest()
        assert result.metadata["chunk_content_hash"] == expected_hash


async def test_page_number_stored_correctly(
    ingestion_service, sample_file_data, vector_store, fake_embedding_client, default_chunk_profile
):
    """Verify page_number in vector metadata matches the source page's page_number."""
    document_id = await ingestion_service.ingest(sample_file_data, default_chunk_profile)

    chunks = await ingestion_service._chunk_store.get_by_document(document_id)
    pages = await ingestion_service._page_store.get_by_document(document_id)
    query_vec = fake_embedding_client.embed([chunks[0].text_content])[0].vector
    results = vector_store.query(namespace="doc-talk", vector=query_vec, top_k=100)

    # Build a map of page_id -> page for easy lookup
    page_by_id = {p.id: p for p in pages}
    chunk_by_id = {c.id: c for c in chunks}

    for result in results:
        chunk = chunk_by_id[result.id]
        page = page_by_id[chunk.page_id]
        assert result.metadata["page_number"] == page.page_number


async def test_chunk_index_stored_correctly(
    ingestion_service, sample_file_data, vector_store, fake_embedding_client, default_chunk_profile
):
    """Verify chunk_index in vector metadata matches the chunk's chunk_index."""
    document_id = await ingestion_service.ingest(sample_file_data, default_chunk_profile)

    chunks = await ingestion_service._chunk_store.get_by_document(document_id)
    query_vec = fake_embedding_client.embed([chunks[0].text_content])[0].vector
    results = vector_store.query(namespace="doc-talk", vector=query_vec, top_k=100)

    chunk_by_id = {c.id: c for c in chunks}

    for result in results:
        chunk = chunk_by_id[result.id]
        assert result.metadata["chunk_index"] == chunk.chunk_index


async def test_offsets_stored_correctly(
    ingestion_service, sample_file_data, vector_store, fake_embedding_client, default_chunk_profile
):
    """Verify chunk offsets in vector metadata match the chunk's offset_start and offset_end."""
    document_id = await ingestion_service.ingest(sample_file_data, default_chunk_profile)

    chunks = await ingestion_service._chunk_store.get_by_document(document_id)
    query_vec = fake_embedding_client.embed([chunks[0].text_content])[0].vector
    results = vector_store.query(namespace="doc-talk", vector=query_vec, top_k=100)

    chunk_by_id = {c.id: c for c in chunks}

    for result in results:
        chunk = chunk_by_id[result.id]
        assert result.metadata["chunk_offset_start"] == chunk.offset_start
        assert result.metadata["chunk_offset_end"] == chunk.offset_end

