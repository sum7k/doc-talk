# tests/integration/ai/documents/test_parsers.py
"""Integration tests for document parsers using real test files."""

from pathlib import Path

import pytest

from documents.models.schemas import FileType
from documents.parsers.docx_parser import DocxParser
from documents.parsers.factory import ParserFactory
from documents.parsers.html_parser import HtmlParser
from documents.parsers.pdf_parser import PDFParser
from documents.parsers.text_parser import TextParser

# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent.parent.parent.parent / "fixtures"


@pytest.fixture
def pdf_content() -> bytes:
    """Load sample PDF file."""
    return (FIXTURES_DIR / "sample.pdf").read_bytes()


@pytest.fixture
def docx_content() -> bytes:
    """Load sample DOCX file."""
    return (FIXTURES_DIR / "sample.docx").read_bytes()


@pytest.fixture
def html_content() -> bytes:
    """Load sample HTML file."""
    return (FIXTURES_DIR / "sample.html").read_bytes()


@pytest.fixture
def markdown_content() -> bytes:
    """Load sample Markdown file."""
    return (FIXTURES_DIR / "sample.md").read_bytes()


@pytest.fixture
def text_content() -> bytes:
    """Load sample text file."""
    return (FIXTURES_DIR / "sample.txt").read_bytes()


class TestParserFactory:
    """Tests for ParserFactory."""

    def test_factory_returns_correct_parser_for_pdf(self):
        factory = ParserFactory()
        parser = factory.get_parser(FileType.PDF)
        assert isinstance(parser, PDFParser)

    def test_factory_returns_correct_parser_for_docx(self):
        factory = ParserFactory()
        parser = factory.get_parser(FileType.DOCX)
        assert isinstance(parser, DocxParser)

    def test_factory_returns_correct_parser_for_html(self):
        factory = ParserFactory()
        parser = factory.get_parser(FileType.HTML)
        assert isinstance(parser, HtmlParser)

    def test_factory_returns_correct_parser_for_markdown(self):
        factory = ParserFactory()
        parser = factory.get_parser(FileType.MD)
        assert isinstance(parser, TextParser)

    def test_factory_returns_correct_parser_for_text(self):
        factory = ParserFactory()
        parser = factory.get_parser(FileType.TEXT)
        assert isinstance(parser, TextParser)


class TestPDFParser:
    """Tests for PDFParser with real PDF files."""

    def test_parse_extracts_pages(self, pdf_content):
        parser = PDFParser()
        pages = parser.parse(pdf_content, "sample.pdf")

        assert len(pages) == 2
        assert pages[0].page_number == 1
        assert pages[1].page_number == 2

    def test_parse_extracts_text_content(self, pdf_content):
        parser = PDFParser()
        pages = parser.parse(pdf_content, "sample.pdf")

        assert "Sample PDF Document" in pages[0].text
        assert "Page 1" in pages[0].text
        assert "integration testing" in pages[0].text

        assert "Page 2" in pages[1].text
        assert "second page" in pages[1].text

    def test_parse_includes_metadata(self, pdf_content):
        parser = PDFParser()
        pages = parser.parse(pdf_content, "sample.pdf")

        assert pages[0].metadata["parser"] == "pypdf"
        assert pages[0].metadata["source"] == "sample.pdf"


class TestDocxParser:
    """Tests for DocxParser with real DOCX files."""

    def test_parse_returns_single_page(self, docx_content):
        parser = DocxParser()
        pages = parser.parse(docx_content, "sample.docx")

        # DOCX doesn't have strict pages, so all content is in one page
        assert len(pages) == 1
        assert pages[0].page_number == 1

    def test_parse_extracts_text_content(self, docx_content):
        parser = DocxParser()
        pages = parser.parse(docx_content, "sample.docx")

        text = pages[0].text
        assert "Sample DOCX Document" in text
        assert "integration testing" in text
        assert "Features Section" in text
        assert "Conclusion" in text

    def test_parse_includes_metadata(self, docx_content):
        parser = DocxParser()
        pages = parser.parse(docx_content, "sample.docx")

        assert pages[0].metadata["parser"] == "python-docx"
        assert pages[0].metadata["source"] == "sample.docx"
        assert "paragraph_count" in pages[0].metadata


class TestHtmlParser:
    """Tests for HtmlParser with real HTML files."""

    def test_parse_returns_single_page(self, html_content):
        parser = HtmlParser()
        pages = parser.parse(html_content, "sample.html")

        assert len(pages) == 1
        assert pages[0].page_number == 1

    def test_parse_extracts_text_content(self, html_content):
        parser = HtmlParser()
        pages = parser.parse(html_content, "sample.html")

        text = pages[0].text
        assert "Sample HTML Document" in text
        assert "Features" in text
        assert "BeautifulSoup" in text

    def test_parse_removes_script_content(self, html_content):
        parser = HtmlParser()
        pages = parser.parse(html_content, "sample.html")

        text = pages[0].text
        assert "console.log" not in text
        assert "This script should be removed" not in text

    def test_parse_removes_style_content(self, html_content):
        parser = HtmlParser()
        pages = parser.parse(html_content, "sample.html")

        text = pages[0].text
        assert "font-family" not in text

    def test_parse_includes_metadata_with_title(self, html_content):
        parser = HtmlParser()
        pages = parser.parse(html_content, "sample.html")

        assert pages[0].metadata["parser"] == "beautifulsoup"
        assert pages[0].metadata["source"] == "sample.html"
        assert pages[0].metadata["title"] == "Sample HTML Document"


class TestTextParser:
    """Tests for TextParser with real text and markdown files."""

    def test_parse_markdown_returns_single_page(self, markdown_content):
        parser = TextParser()
        pages = parser.parse(markdown_content, "sample.md")

        assert len(pages) == 1
        assert pages[0].page_number == 1

    def test_parse_markdown_extracts_full_content(self, markdown_content):
        parser = TextParser()
        pages = parser.parse(markdown_content, "sample.md")

        text = pages[0].text
        assert "# Sample Markdown Document" in text
        assert "## Features" in text
        assert "```python" in text
        assert "def hello_world():" in text

    def test_parse_text_returns_single_page(self, text_content):
        parser = TextParser()
        pages = parser.parse(text_content, "sample.txt")

        assert len(pages) == 1
        assert pages[0].page_number == 1

    def test_parse_text_extracts_full_content(self, text_content):
        parser = TextParser()
        pages = parser.parse(text_content, "sample.txt")

        text = pages[0].text
        assert "Sample Plain Text Document" in text
        assert "Section 1: Introduction" in text
        assert "Section 2: Content" in text
        assert "Section 3: Conclusion" in text

    def test_parse_includes_metadata(self, text_content):
        parser = TextParser()
        pages = parser.parse(text_content, "sample.txt")

        assert pages[0].metadata["parser"] == "text"
        assert pages[0].metadata["source"] == "sample.txt"


class TestParserWithFactory:
    """Integration tests using ParserFactory with real files."""

    @pytest.mark.parametrize(
        "file_type,fixture_name,expected_content",
        [
            (FileType.PDF, "pdf_content", "Sample PDF Document"),
            (FileType.DOCX, "docx_content", "Sample DOCX Document"),
            (FileType.HTML, "html_content", "Sample HTML Document"),
            (FileType.MD, "markdown_content", "Sample Markdown Document"),
            (FileType.TEXT, "text_content", "Sample Plain Text Document"),
        ],
    )
    def test_factory_parser_extracts_content(
        self, file_type, fixture_name, expected_content, request
    ):
        """Test that factory-created parsers correctly extract content."""
        content = request.getfixturevalue(fixture_name)

        factory = ParserFactory()
        parser = factory.get_parser(file_type)
        pages = parser.parse(content, f"sample.{file_type.value}")

        assert len(pages) >= 1
        # Check that expected content is somewhere in the parsed output
        all_text = " ".join(page.text for page in pages)
        assert expected_content in all_text

    def test_all_parsers_return_valid_page_schema(
        self,
        pdf_content,
        docx_content,
        html_content,
        markdown_content,
        text_content,
    ):
        """Test that all parsers return valid PageSchema objects."""
        factory = ParserFactory()
        test_cases = [
            (FileType.PDF, pdf_content, "sample.pdf"),
            (FileType.DOCX, docx_content, "sample.docx"),
            (FileType.HTML, html_content, "sample.html"),
            (FileType.MD, markdown_content, "sample.md"),
            (FileType.TEXT, text_content, "sample.txt"),
        ]

        for file_type, content, source_name in test_cases:
            parser = factory.get_parser(file_type)
            pages = parser.parse(content, source_name)

            for page in pages:
                assert page.page_number >= 1
                assert len(page.text) > 0
                assert isinstance(page.metadata, dict)
                assert "parser" in page.metadata
                assert "source" in page.metadata
