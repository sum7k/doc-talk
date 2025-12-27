import os
import re
from fastapi import APIRouter, File, Form, UploadFile
from .dependencies import DocumentServiceDep, ChatServiceDep
from documents.models.schemas import ChunkingStrategy
from documents.models.schemas import CHUNKING_PROFILES, FileData, FileType
from core.utils import normalize_filename
router = APIRouter(prefix="/documents", tags=["Documents"])

@router.post("/upload")
async def upload_document_controller(
    ingestion_service: DocumentServiceDep,
    file: UploadFile = File(...),
    file_type: FileType = Form(...),
    chunking_strategy: ChunkingStrategy = Form(...),
) -> dict:
    """Upload a document for ingestion.
    
    Args:
        file: The document file to upload
        file_type: Explicit file type (html, pdf, docx, md, text)
        chunking_strategy: Chunking strategy to use (default, dense_text, qa)
        ingestion_service: Document ingestion service
    """
    # Read file content
    content = await file.read()
    filename = normalize_filename(file.filename or "unknown")
    
    # Create FileData instance
    file_data = FileData(
        file_name=filename,
        file_type=file_type,
        binary_content=content,
    )
    
    # Get chunking profile
    chunk_profile = CHUNKING_PROFILES[chunking_strategy]
    
    # Ingest the document
    document_id = await ingestion_service.ingest(file_data, chunk_profile)
    
    return {
        "document_id": document_id,
        "filename": filename,
        "file_type": file_type.value,
        "chunking_strategy": chunking_strategy.value,
    }

@router.post("/chat")
async def chat_with_document_controller(query: str, chat_service: ChatServiceDep):
    
    response = await chat_service.generate_response(query)
    return {
        "answer": response.llm_response.content,
        "citations": [
            {
                "document_id": c.document_id,
                "title": c.title,
                "page_number": c.page_number,
                "snippet": c.snippet,
            }
            for c in response.citations
        ],
        "finish_reason": response.llm_response.finish_reason,
    }

@router.get("/list")
async def list_documents_controller(document_service: DocumentServiceDep, skip: int = 0, take: int = 10) -> dict:
    """List all ingested documents.
    
    Args:
        document_service: Document ingestion service
    """
    documents = await document_service.list_documents(skip=skip, take=take)
    return {"documents": documents}

@router.delete("/{document_id}")
async def delete_document_controller(document_service: DocumentServiceDep, document_id: str) -> dict:
    """Delete a document by its ID.
    
    Args:
        document_id: The ID of the document to delete
        document_service: Document ingestion service
    """
    await document_service.delete_document(document_id)
    return {"message": f"Document {document_id} deleted successfully."}