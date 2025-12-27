import structlog
from opentelemetry import trace

from documents.models.domain import PageSchema

logger = structlog.get_logger()
tracer = trace.get_tracer(__name__)


class TextParser:
    """Parser for plain text and markdown documents."""

    def parse(self, binary: bytes, source_name: str) -> list[PageSchema]:
        with tracer.start_as_current_span("text_parser.parse") as span:
            span.set_attribute("parser.source_name", source_name)
            span.set_attribute("parser.type", "text")

            logger.info("parsing_text", source_name=source_name)

            # Decode text content, trying common encodings
            text = self._decode_text(binary)

            pages: list[PageSchema] = []
            if text.strip():
                pages.append(
                    PageSchema(
                        page_number=1,
                        text=text,
                        metadata={
                            "parser": "text",
                            "source": source_name,
                        },
                    )
                )

            span.set_attribute("parser.page_count", len(pages))
            span.set_attribute("parser.text_length", len(text))
            logger.info(
                "text_parsed",
                source_name=source_name,
                text_length=len(text),
            )

            return pages

    def _decode_text(self, binary: bytes) -> str:
        """Decode binary content to text, trying multiple encodings."""
        encodings = ["utf-8", "utf-16", "latin-1", "cp1252"]

        for encoding in encodings:
            try:
                return binary.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                continue

        # Fallback: decode with errors replaced
        return binary.decode("utf-8", errors="replace")
