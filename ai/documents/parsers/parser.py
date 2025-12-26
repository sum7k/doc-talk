from typing import Protocol

import structlog

from ai.documents.models.domain import PageSchema

logger = structlog.get_logger()


class Parser(Protocol):
    def parse(self, binary: bytes, source_name: str) -> list[PageSchema]: ...
