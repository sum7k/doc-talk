from typing_extensions import Annotated

from fastapi import Depends
from llm_kit.embeddings.base import EmbeddingsClient
from llm_kit.embeddings.factory import EmbeddingsConfig, create_embeddings_client
from llm_kit.prompts.prompts_library import PromptsLibrary
from llm_kit.vectorstores.base import VectorStore
from llm_kit.vectorstores.pgvectorstore import PgVectorStore
from llm_kit.vectorstores.qdrantvectorstore import QdrantVectorStore
from llm_kit.llms.factory import create_llm_client
from llm_kit.llms.config import LLMConfig
from qdrant_client.models import Distance

from documents.models.domain import EmbeddingContext
from documents.repositories.document import (
    ChunkRepository,
    DocumentRepository,
    PageRepository,
)
from documents.repositories.file import FileRepository
from documents.services.chat import DocumentChatService
from documents.services.dense_retrieval import DenseRetrievalService
from core.database import DBSessionDep
from core.settings import PgVectorConfig, QdrantConfig, SettingsDep

from documents.services.document import DocumentService

# Singleton instances
_embedding_client: EmbeddingsClient | None = None
_prompts_library: PromptsLibrary | None = None
_vector_store: VectorStore | None = None


def get_embedding_client(
    settings: SettingsDep,
) -> EmbeddingsClient:
    global _embedding_client
    if _embedding_client is None:
        config = settings.embeddings
        _embedding_client = create_embeddings_client(
            EmbeddingsConfig(
                provider=config.provider,
                model=config.model,
                timeout=config.timeout,
                batch_size=config.batch_size,
                api_key=config.api_key,
            )
        )
    return _embedding_client


def get_prompts_library(
    settings: SettingsDep,
) -> PromptsLibrary:
    global _prompts_library
    if _prompts_library is None:
        _prompts_library = PromptsLibrary("documents/prompts")
    return _prompts_library


def get_embedding_context(
    settings: SettingsDep,
) -> EmbeddingContext:
    config = settings.embeddings
    return EmbeddingContext(
        provider=config.provider,
        model=config.model,
        version=getattr(config, "version", "v1"),
        namespace=getattr(config, "namespace", "doc-talk"),
    )


def get_vector_store(
    settings: SettingsDep,
) -> VectorStore:
    global _vector_store
    if _vector_store is not None:
        return _vector_store
    
    # Use Qdrant if URL or path is configured, otherwise PgVector
    if settings.qdrant.url or settings.qdrant.path:
        cfg = settings.qdrant
        distance_map = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dot": Distance.DOT,
        }
        _vector_store = QdrantVectorStore(
            url=cfg.url,
            path=cfg.path,
            api_key=cfg.api_key,
            collection_name=cfg.collection_name,
            vector_size=cfg.vector_size,
            distance=distance_map[cfg.distance],
            on_disk=cfg.on_disk,
        )
    elif settings.pgvector.dsn:
        cfg = settings.pgvector
        _vector_store = PgVectorStore(
            dsn=cfg.dsn,
            pool_min_size=cfg.pool_min_size,
            pool_max_size=cfg.pool_max_size,
        )
    else:
        raise ValueError("No vector store configured. Set QDRANT__URL, QDRANT__PATH, or PG_VECTOR_DSN in .env file.")
    
    return _vector_store


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


def get_file_repository(
    settings: SettingsDep,
) -> FileRepository:
    file_storage_path = f"{settings.data_dir}/uploaded_files"
    return FileRepository(file_storage_path)


def get_ingestion_service(
    session: DBSessionDep,
    embedding_client: EmbeddingsClient = Depends(get_embedding_client),
    embedding_context: EmbeddingContext = Depends(get_embedding_context),
    vector_store=Depends(get_vector_store),
    document_store: DocumentRepository = Depends(get_document_repository),
    page_store: PageRepository = Depends(get_page_repository),
    chunk_store: ChunkRepository = Depends(get_chunk_repository),
    file_store: FileRepository = Depends(get_file_repository),
) -> DocumentService:
    return DocumentService(
        embedding_client=embedding_client,
        embedding_context=embedding_context,
        vector_store=vector_store,
        document_store=document_store,
        page_store=page_store,
        chunk_store=chunk_store,
        file_store=file_store,
    )


def get_dense_retrieval_service(
    embedding_client: EmbeddingsClient = Depends(get_embedding_client),
    embedding_context: EmbeddingContext = Depends(get_embedding_context),
    vector_store=Depends(get_vector_store),
):
    return DenseRetrievalService(
        embedding_client,
        embedding_context,
        vector_store,
    )


def get_llm_client(
    settings: SettingsDep,
):
    llm_config: LLMConfig = LLMConfig(**settings.llm.model_dump())
    return create_llm_client(llm_config)


def get_chat_service(
    document_repository: DocumentRepository = Depends(get_document_repository),
    chunk_repository: ChunkRepository = Depends(get_chunk_repository),
    page_repository: PageRepository = Depends(get_page_repository),
    dense_retrieval=Depends(get_dense_retrieval_service),
    llm_client=Depends(get_llm_client),
    prompt_library=Depends(get_prompts_library),
):
    return DocumentChatService(
        document_repository=document_repository,
        chunk_repository=chunk_repository,
        page_repository=page_repository,
        dense_retrieval=dense_retrieval,
        llm_client=llm_client,
        system_prompt=prompt_library.get("document_search", "1.0"),
    )

DocumentRepositoryDep = Annotated[DocumentRepository, Depends(get_document_repository)]
ChatServiceDep = Annotated[DocumentChatService, Depends(get_chat_service)]
DocumentServiceDep = Annotated[DocumentService, Depends(get_ingestion_service)]