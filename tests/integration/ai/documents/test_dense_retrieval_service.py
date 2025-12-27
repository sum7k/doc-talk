# tests/integration/ai/documents/test_dense_retrieval_service.py
import uuid

import pytest
import pytest_asyncio
from qdrant_client.models import Distance

from documents.models.domain import EmbeddingContext
from documents.services.dense_retrieval import DenseRetrievalService

pytestmark = pytest.mark.asyncio(loop_scope="module")

# Sample texts for embedding and retrieval testing
SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "Python is a popular programming language for data science.",
    "Natural language processing enables computers to understand human language.",
    "Deep learning uses neural networks with many layers.",
    "The weather is sunny and warm today.",
    "Cats are independent and curious animals.",
    "The stock market experienced significant volatility.",
    "Quantum computing promises exponential speedups for certain problems.",
    "Coffee is one of the most popular beverages worldwide.",
    "Electric vehicles are becoming more affordable.",
    "The human brain contains approximately 86 billion neurons.",
]

# all-MiniLM-L6-v2 produces 384-dimensional embeddings
EMBEDDING_DIM = 384
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture(scope="module")
def vector_store():
    """Create QdrantVectorStore with in-memory storage."""
    from llm_kit.vectorstores.qdrantvectorstore import QdrantVectorStore

    store = QdrantVectorStore(
        url=None,  # In-memory mode
        collection_name="test-embeddings",
        vector_size=EMBEDDING_DIM,
        distance=Distance.COSINE,
    )
    yield store


@pytest.fixture(scope="module")
def embedding_client():
    """Create a real sentence-transformers embedding client using all-MiniLM-L6-v2."""
    from llm_kit.embeddings.factory import EmbeddingsConfig, create_embeddings_client

    client = create_embeddings_client(
        EmbeddingsConfig(
            provider="local",
            model=MODEL_NAME,
        )
    )
    return client


@pytest.fixture(scope="module")
def embedding_context():
    """Create embedding context for testing."""
    return EmbeddingContext(
        provider="local",
        model=MODEL_NAME,
        version="v1",
        namespace="test-dense-retrieval",
    )


@pytest_asyncio.fixture(scope="module")
async def seeded_vector_store(vector_store, embedding_client, embedding_context):
    """Embed sample texts and insert them into the vector store."""
    from llm_kit.vectorstores.types import VectorItem

    # Embed all sample texts
    embeddings = await embedding_client.embed(texts=SAMPLE_TEXTS)

    # Create vector items with metadata (using UUIDs for Qdrant in-memory mode)
    items = []
    for idx, (text, embedding) in enumerate(zip(SAMPLE_TEXTS, embeddings)):
        item = VectorItem(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"sample-{idx}")),
            vector=embedding.vector,
            metadata={
                "text": text,
                "index": idx,
                "embedding_model_name": embedding_context.model,
                "embedding_version": embedding_context.version,
            },
        )
        items.append(item)

    # Upsert all items into the vector store
    await vector_store.upsert(
        namespace=embedding_context.namespace,
        items=items,
    )

    yield vector_store

    # Cleanup
    await vector_store.close()


@pytest_asyncio.fixture
async def dense_retrieval_service(
    seeded_vector_store, embedding_client, embedding_context
):
    """Create DenseRetrievalService with real dependencies."""
    return DenseRetrievalService(
        embedding_client=embedding_client,
        embedding_context=embedding_context,
        vector_store=seeded_vector_store,
    )


# =============================================================================
# Integration Tests
# =============================================================================


class TestDenseRetrievalServiceIntegration:
    """Integration tests for DenseRetrievalService with real embeddings and Qdrant."""

    async def test_retrieve_returns_results(self, dense_retrieval_service):
        """Verify retrieval returns results for a valid query."""
        query = "What is machine learning?"
        results = await dense_retrieval_service.retrieve(query, top_k=3)

        assert len(results) > 0
        assert len(results) <= 3

    async def test_retrieve_returns_relevant_results_for_ml_query(
        self, dense_retrieval_service
    ):
        """Verify ML-related queries return ML-related documents."""
        query = "artificial intelligence and machine learning"
        results = await dense_retrieval_service.retrieve(query, top_k=3)

        # Get the text from the top result's metadata
        top_texts = [r.metadata.get("text", "") for r in results]

        # At least one result should be ML/AI related
        ml_keywords = [
            "machine learning",
            "artificial intelligence",
            "neural",
            "deep learning",
        ]
        has_relevant_result = any(
            any(keyword in text.lower() for keyword in ml_keywords)
            for text in top_texts
        )
        assert has_relevant_result, f"Expected ML-related results, got: {top_texts}"

    async def test_retrieve_returns_relevant_results_for_animals_query(
        self, dense_retrieval_service
    ):
        """Verify animal-related queries return animal-related documents."""
        query = "pets and animals behavior"
        results = await dense_retrieval_service.retrieve(query, top_k=3)

        top_texts = [r.metadata.get("text", "") for r in results]

        # Check for animal-related content
        animal_keywords = ["fox", "dog", "cats", "animals"]
        has_relevant_result = any(
            any(keyword in text.lower() for keyword in animal_keywords)
            for text in top_texts
        )
        assert has_relevant_result, f"Expected animal-related results, got: {top_texts}"

    async def test_retrieve_returns_relevant_results_for_programming_query(
        self, dense_retrieval_service
    ):
        """Verify programming-related queries return programming-related documents."""
        query = "coding and software development"
        results = await dense_retrieval_service.retrieve(query, top_k=3)

        top_texts = [r.metadata.get("text", "") for r in results]

        # Check for programming-related content
        programming_keywords = ["python", "programming", "language", "data science"]
        has_relevant_result = any(
            any(keyword in text.lower() for keyword in programming_keywords)
            for text in top_texts
        )
        assert has_relevant_result, (
            f"Expected programming-related results, got: {top_texts}"
        )

    async def test_retrieve_respects_top_k_parameter(self, dense_retrieval_service):
        """Verify top_k parameter limits the number of results."""
        query = "technology and science"

        results_1 = await dense_retrieval_service.retrieve(query, top_k=1)
        results_5 = await dense_retrieval_service.retrieve(query, top_k=5)
        results_10 = await dense_retrieval_service.retrieve(query, top_k=10)

        assert len(results_1) == 1
        assert len(results_5) == 5
        assert len(results_10) == 10

    async def test_retrieve_results_have_expected_structure(
        self, dense_retrieval_service
    ):
        """Verify results have the expected QueryResult structure."""
        query = "coffee and beverages"
        results = await dense_retrieval_service.retrieve(query, top_k=1)

        assert len(results) == 1
        result = results[0]

        # Check QueryResult has expected attributes
        assert hasattr(result, "id")
        assert hasattr(result, "score")
        assert hasattr(result, "metadata")

        # Check metadata contains expected fields
        assert "text" in result.metadata
        assert "embedding_model_name" in result.metadata
        assert "embedding_version" in result.metadata

    async def test_retrieve_results_ordered_by_similarity(
        self, dense_retrieval_service
    ):
        """Verify results are ordered by similarity score (descending)."""
        query = "neural networks and deep learning"
        results = await dense_retrieval_service.retrieve(query, top_k=5)

        scores = [r.score for r in results]

        # Scores should be in descending order (most similar first)
        assert scores == sorted(scores, reverse=True), (
            "Results should be ordered by similarity"
        )

    async def test_retrieve_with_exact_text_match_returns_high_score(
        self, dense_retrieval_service
    ):
        """Verify querying with exact text returns a high similarity score."""
        # Use one of the sample texts as the query
        exact_query = "Machine learning is a subset of artificial intelligence."
        results = await dense_retrieval_service.retrieve(exact_query, top_k=1)

        assert len(results) == 1
        # The exact match should have a very high similarity score
        assert results[0].score > 0.9, (
            f"Expected high score for exact match, got {results[0].score}"
        )
        assert results[0].metadata["text"] == exact_query

    async def test_retrieve_different_queries_return_different_results(
        self, dense_retrieval_service
    ):
        """Verify different queries return different top results."""
        weather_results = await dense_retrieval_service.retrieve(
            "sunny weather forecast", top_k=1
        )
        finance_results = await dense_retrieval_service.retrieve(
            "stock market trading", top_k=1
        )

        # Top results should be different
        assert weather_results[0].id != finance_results[0].id

    async def test_retrieve_handles_empty_query(self, dense_retrieval_service):
        """Verify service handles empty query gracefully."""
        # Empty string still gets embedded and returns results
        results = await dense_retrieval_service.retrieve("", top_k=3)
        assert len(results) == 3

    async def test_retrieve_handles_long_query(self, dense_retrieval_service):
        """Verify service handles long queries."""
        long_query = " ".join(["machine learning"] * 50)
        results = await dense_retrieval_service.retrieve(long_query, top_k=3)

        assert len(results) == 3
