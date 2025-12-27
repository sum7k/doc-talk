import structlog
from bs4 import BeautifulSoup
from opentelemetry import trace

from documents.models.domain import PageSchema

logger = structlog.get_logger()
tracer = trace.get_tracer(__name__)


class HtmlParser:
    """Parser for HTML documents using BeautifulSoup."""

    def parse(self, binary: bytes, source_name: str) -> list[PageSchema]:
        with tracer.start_as_current_span("html_parser.parse") as span:
            span.set_attribute("parser.source_name", source_name)
            span.set_attribute("parser.type", "html")

            logger.info("parsing_html", source_name=source_name)

            # Decode HTML content
            html_content = self._decode_html(binary)

            # Parse HTML and extract text
            soup = BeautifulSoup(html_content, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()

            # Get text content
            text = soup.get_text(separator="\n", strip=True)

            # Extract title if available
            title = soup.title.string if soup.title else None

            pages: list[PageSchema] = []
            if text.strip():
                pages.append(
                    PageSchema(
                        page_number=1,
                        text=text,
                        metadata={
                            "parser": "beautifulsoup",
                            "source": source_name,
                            "title": title,
                        },
                    )
                )

            span.set_attribute("parser.page_count", len(pages))
            span.set_attribute("parser.text_length", len(text))
            logger.info(
                "html_parsed",
                source_name=source_name,
                text_length=len(text),
                title=title,
            )

            return pages

    def _decode_html(self, binary: bytes) -> str:
        """Decode HTML binary content to string."""
        encodings = ["utf-8", "utf-16", "latin-1", "cp1252"]

        for encoding in encodings:
            try:
                return binary.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                continue

        return binary.decode("utf-8", errors="replace")
