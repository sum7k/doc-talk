import asyncio
import structlog

from llm_kit import QueryResult, Embedding, VectorStore, EmbeddingsClient
from ai.documents.models.domain import EmbeddingContext

logger = structlog.get_logger(__name__)


class DenseRetrievalService:

    def __init__(
        self,
        embedding_client: EmbeddingsClient,
        embedding_context: EmbeddingContext,
        vector_store: VectorStore,
    ):
        self.embedding_client = embedding_client
        self.embedding_context = embedding_context
        self.vector_store = vector_store
        self.filters = {
            "embedding_model_name": self.embedding_context.model,
            "embedding_version": self.embedding_context.version,
        }
        logger.info(
            "dense_retrieval_service_initialized",
            namespace=self.embedding_context.namespace,
            model=self.embedding_context.model,
            version=self.embedding_context.version,
        )

    async def retrieve(self, query: str, top_k: int = 5) -> list[QueryResult]:
        logger.info(
            "dense_retrieval_started",
            query_length=len(query),
            top_k=top_k,
            namespace=self.embedding_context.namespace,
        )

        query_embeddings: list[Embedding] = self.embedding_client.embed(
            texts=[query],
        )
        logger.debug(
            "query_embedding_generated",
            vector_dimension=len(query_embeddings[0].vector),
        )

        results: list[QueryResult] = await asyncio.to_thread(
            self.vector_store.query,
            namespace=self.embedding_context.namespace,
            vector=query_embeddings[0].vector,
            top_k=top_k,
            filters=self.filters,
        )

        logger.info(
            "dense_retrieval_completed",
            results_count=len(results),
            top_k=top_k,
        )

        return results
