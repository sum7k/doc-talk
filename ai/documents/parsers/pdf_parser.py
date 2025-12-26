import io

import structlog
from opentelemetry import trace
from pypdf import PdfReader

from ai.documents.models.domain import PageSchema

logger = structlog.get_logger()
tracer = trace.get_tracer(__name__)


class PDFParser:
    """Parser for PDF documents using pypdf."""

    def parse(self, binary: bytes, source_name: str) -> list[PageSchema]:
        with tracer.start_as_current_span("pdf_parser.parse") as span:
            span.set_attribute("parser.source_name", source_name)
            span.set_attribute("parser.type", "pdf")

            logger.info("parsing_pdf", source_name=source_name)

            reader = PdfReader(io.BytesIO(binary))
            pages: list[PageSchema] = []

            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(
                        PageSchema(
                            page_number=page_num,
                            text=text,
                            metadata={
                                "parser": "pypdf",
                                "source": source_name,
                            },
                        )
                    )

            span.set_attribute("parser.page_count", len(pages))
            logger.info(
                "pdf_parsed",
                source_name=source_name,
                page_count=len(pages),
            )

            return pages
