import io

import structlog
from docx import Document
from opentelemetry import trace

from ai.documents.models.domain import PageSchema

logger = structlog.get_logger()
tracer = trace.get_tracer(__name__)


class DocxParser:
    """Parser for DOCX documents using python-docx."""

    def parse(self, binary: bytes, source_name: str) -> list[PageSchema]:
        with tracer.start_as_current_span("docx_parser.parse") as span:
            span.set_attribute("parser.source_name", source_name)
            span.set_attribute("parser.type", "docx")

            logger.info("parsing_docx", source_name=source_name)

            doc = Document(io.BytesIO(binary))

            # Extract all paragraphs as a single page (DOCX doesn't have strict pages)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            text = "\n\n".join(paragraphs)

            pages: list[PageSchema] = []
            if text.strip():
                pages.append(
                    PageSchema(
                        page_number=1,
                        text=text,
                        metadata={
                            "parser": "python-docx",
                            "source": source_name,
                            "paragraph_count": len(paragraphs),
                        },
                    )
                )

            span.set_attribute("parser.page_count", len(pages))
            span.set_attribute("parser.paragraph_count", len(paragraphs))
            logger.info(
                "docx_parsed",
                source_name=source_name,
                paragraph_count=len(paragraphs),
            )

            return pages
