import asyncio
from opentelemetry import trace

import structlog
from documents.models.domain import ChatResponseDTO, CitationDTO
from documents.repositories.document import ChunkRepository, DocumentRepository, PageRepository
from documents.services.dense_retrieval import DenseRetrievalService
from llm_kit.llms import LLMClient, Message, Role
from llm_kit.prompts.prompt import Prompt

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)


class DocumentChatService:
    def __init__(
        self,
        document_repository: DocumentRepository,
        chunk_repository: ChunkRepository,
        page_repository: PageRepository,
        dense_retrieval: DenseRetrievalService,
        llm_client: LLMClient,
        system_prompt: Prompt,
    ):
        self.document_repository = document_repository
        self.chunk_repository = chunk_repository
        self.page_repository = page_repository
        self.llm_client = llm_client
        self.dense_retrieval = dense_retrieval
        self.system_prompt = system_prompt
        logger.info("document_chat_service_initialized")

    async def generate_response(self, query: str) -> ChatResponseDTO:
        logger.info("generate_response_started", query_length=len(query))
        with tracer.start_as_current_span("document_chat.generate_response") as span:
            span.set_attribute("query.length", len(query))

            with tracer.start_as_current_span("document_chat.retrieval"):
                with tracer.start_as_current_span(
                    "document_chat.retrieve_vectors"
                ) as span:
                    span.set_attribute("retrieval.top_k", 15)
                    # Get relevant documents
                    relevant_vectors = await self.dense_retrieval.retrieve(
                        query, top_k=15
                    )
                relevant_vector_ids = [vec.id for vec in relevant_vectors]

                with tracer.start_as_current_span("document_chat.fetch_chunks"):
                    relevant_chunks = await self.chunk_repository.get_many(
                        relevant_vector_ids
                    )

            # Build citations from retrieved chunks
            citations: list[CitationDTO] = []
            for chunk in relevant_chunks:
                # Get page info for the chunk
                page = await self.page_repository.get(chunk.page_id)
                if page:
                    document = await self.document_repository.get(page.document_id)
                    citations.append(
                        CitationDTO(
                            document_id=page.document_id,
                            title=document.display_title if document else None,
                            page_number=page.page_number,
                            snippet=chunk.text_content[:200] + "..." if len(chunk.text_content) > 200 else chunk.text_content,
                        )
                    )

            # Prepare context for the chat model
            context = "\n\n".join(
                f"[chunk_id={chunk.id}]\n{chunk.text_content}"
                for chunk in relevant_chunks
            )

            messages = [
                Message(role=Role.SYSTEM, content=self.system_prompt.template),
                Message(role=Role.SYSTEM, content=f"CONTEXT:\n{context}"),
                Message(role=Role.USER, content=query),
            ]

            with tracer.start_as_current_span("document_chat.call_llm") as span:
                # Generate response using the chat model
                llm_response = await self.llm_client.complete(messages=messages)

            span.set_attribute("response.length", len(llm_response.content or ""))
            logger.info(
                "generate_response_completed",
                response_length=len(llm_response.content or ""),
                tool_call_count=len(llm_response.tool_calls),
                finish_reason=llm_response.finish_reason,
                citation_count=len(citations),
            )
            
            return ChatResponseDTO(
                citations=citations,
                llm_response=llm_response,
            )
