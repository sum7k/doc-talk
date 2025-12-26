from functools import lru_cache
from fastapi import Depends
from llm_kit.embeddings.factory import create_embeddings_client, EmbeddingsConfig
from llm_kit.embeddings.base import EmbeddingsClient
from ai.documents.models.domain import EmbeddingContext
from ai.documents.repositories.document import (
    ChunkRepository,
    DocumentRepository,
    PageRepository,
)
from core import settings
from core.database import DBSessionDep
from core.settings import PgVectorConfig, QdrantConfig, QdrantConfig, SettingsDep
from .documents.services.ingestion import DocumentIngestionService
from llm_kit.vectorstores.pgvectorstore import PgVectorStore
from llm_kit.vectorstores.qdrantvectorstore import QdrantVectorStore
from qdrant_client.models import Distance

@lru_cache
def get_embedding_client(
    settings: SettingsDep,
):
    config = settings.embeddings_config
    return create_embeddings_client(
        EmbeddingsConfig(
            provider=config.provider,
            model=config.model,
            timeout=config.timeout,
            batch_size=config.batch_size,
            api_key=config.api_key,
        )
    )


def get_embedding_context(
    settings: SettingsDep,
):
    config = settings.embeddings_config
    return EmbeddingContext(
        provider=config.provider,
        model=config.model,
        version=getattr(config, "version", "v1"),
        namespace=getattr(config, "namespace", "doc-talk"),
    )

@lru_cache
def get_vector_store(
    settings: SettingsDep,
):
    cfg = settings.vector_store
    if isinstance(cfg, PgVectorConfig):
        pg_vector_cfg: PgVectorConfig = cfg
        return PgVectorStore(
            dsn=pg_vector_cfg.dsn,
            pool_min_size=pg_vector_cfg.pool_min_size,
            pool_max_size=pg_vector_cfg.pool_max_size,
        )
    if isinstance(cfg, QdrantConfig):
        qdrant_cfg: QdrantConfig = cfg
        distance_map = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dot": Distance.DOT,
        }
        return QdrantVectorStore(
            url=qdrant_cfg.url,
            api_key=qdrant_cfg.api_key,
            collection_name=qdrant_cfg.collection_name,
            vector_size=qdrant_cfg.vector_size,
            distance=distance_map[qdrant_cfg.distance],
            on_disk=qdrant_cfg.on_disk,
        )
    raise ValueError(f"Unsupported vector store backend: {cfg.backend}")


def get_document_repository(
    session: DBSessionDep,
):
    return DocumentRepository(
        session=session,
    )


def get_page_repository(
    session: DBSessionDep,
):
    return PageRepository(
        session=session,
    )


def get_chunk_repository(
    session: DBSessionDep,
):
    return ChunkRepository(
        session=session,
    )


def get_ingestion_service(
    session: DBSessionDep,
    embedding_client: EmbeddingsClient = Depends(get_embedding_client),
    embedding_context: EmbeddingContext = Depends(get_embedding_context),
    vector_store = Depends(get_vector_store),
    document_store: DocumentRepository = Depends(get_document_repository),
    page_store: PageRepository = Depends(get_page_repository),
    chunk_store: ChunkRepository = Depends(get_chunk_repository),
):
    return DocumentIngestionService(
        embedding_client=embedding_client,
        embedding_context=embedding_context,
        vector_store=vector_store,
        document_store=document_store,
        page_store=page_store,
        chunk_store=chunk_store,
    )
