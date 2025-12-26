from ai.documents.models.schemas import FileType

from .docx_parser import DocxParser
from .html_parser import HtmlParser
from .parser import Parser
from .pdf_parser import PDFParser
from .text_parser import TextParser


class ParserFactory:
    """Factory for creating document parsers based on file type."""

    _parsers: dict[FileType, type[Parser]] = {
        FileType.PDF: PDFParser,
        FileType.DOCX: DocxParser,
        FileType.HTML: HtmlParser,
        FileType.MD: TextParser,
        FileType.TEXT: TextParser,
    }

    def get_parser(self, file_type: FileType) -> Parser:
        """Get a parser instance for the given file type.

        Args:
            file_type: The type of file to parse.

        Returns:
            A parser instance capable of parsing the file type.

        Raises:
            ValueError: If no parser is available for the file type.
        """
        parser_class = self._parsers.get(file_type)
        if parser_class is None:
            raise ValueError(f"No parser available for file type: {file_type}")
        return parser_class()
