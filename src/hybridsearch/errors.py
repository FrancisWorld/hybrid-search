"""Structured error hierarchy. Every error carries a stable machine-readable ``code``."""

from __future__ import annotations


class SearchError(Exception):
    code = "search_error"

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": str(self)}


class ValidationError(SearchError):
    """A document, filter or request is malformed."""

    code = "validation_error"


class QueryError(SearchError):
    """The query string could not be parsed."""

    code = "query_syntax_error"

    def __init__(self, message: str, position: int | None = None) -> None:
        super().__init__(message if position is None else f"{message} (at offset {position})")
        self.position = position


class CorruptIndexError(SearchError):
    """On-disk data failed an integrity check."""

    code = "corrupt_index"


class IndexLockedError(SearchError):
    """Another process already holds the index directory."""

    code = "index_locked"


class IndexClosedError(SearchError):
    code = "index_closed"
