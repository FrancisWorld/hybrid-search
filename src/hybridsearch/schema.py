"""Index schema and document validation."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from hybridsearch.errors import ValidationError

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
MAX_ID_LENGTH = 256
MAX_TEXT_CHARS = 1_000_000
MAX_KEYWORD_VALUES = 64


@dataclass(frozen=True)
class Schema:
    """Which fields exist and how they are indexed.

    ``text_fields`` map to BM25 boosts; ``keyword_fields`` support exact filters
    and facets; ``number_fields`` support range filters.
    """

    language: str = "en"
    text_fields: Mapping[str, float] = field(default_factory=lambda: {"title": 2.0, "body": 1.0})
    keyword_fields: tuple[str, ...] = ()
    number_fields: tuple[str, ...] = ()
    embedding_dim: int = 256

    def __post_init__(self) -> None:
        names = [*self.text_fields, *self.keyword_fields, *self.number_fields]
        if not self.text_fields:
            raise ValidationError("schema needs at least one text field")
        for name in names:
            if not _NAME_RE.match(name):
                raise ValidationError(f"invalid field name {name!r}")
        if len(set(names)) != len(names):
            raise ValidationError("field names must be unique across text, keyword and number")
        if any(boost <= 0 or not math.isfinite(boost) for boost in self.text_fields.values()):
            raise ValidationError("text field boosts must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "text_fields": dict(self.text_fields),
            "keyword_fields": list(self.keyword_fields),
            "number_fields": list(self.number_fields),
            "embedding_dim": self.embedding_dim,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Schema:
        try:
            return cls(
                language=str(data["language"]),
                text_fields={str(k): float(v) for k, v in data["text_fields"].items()},
                keyword_fields=tuple(str(x) for x in data.get("keyword_fields", ())),
                number_fields=tuple(str(x) for x in data.get("number_fields", ())),
                embedding_dim=int(data["embedding_dim"]),
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValidationError(f"invalid schema: {exc}") from exc


@dataclass(frozen=True)
class Document:
    id: str
    fields: Mapping[str, str]
    keywords: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    numbers: Mapping[str, float] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """All text fields joined, used for the embedding."""
        return "\n".join(self.fields.values())

    def to_dict(self) -> dict[str, Any]:
        meta: dict[str, Any] = {k: list(v) for k, v in self.keywords.items()}
        meta.update(self.numbers)
        return {"id": self.id, "fields": dict(self.fields), "meta": meta}


def parse_document(raw: object, schema: Schema) -> Document:
    """Validate an untrusted mapping against ``schema``."""
    if not isinstance(raw, Mapping):
        raise ValidationError("document must be an object")
    unknown = set(raw) - {"id", "fields", "meta"}
    if unknown:
        raise ValidationError(f"unknown document keys: {sorted(unknown)}")
    doc_id = raw.get("id")
    if not isinstance(doc_id, str) or not doc_id.strip() or len(doc_id) > MAX_ID_LENGTH:
        raise ValidationError(f"id must be a non-empty string of at most {MAX_ID_LENGTH} chars")

    raw_fields = raw.get("fields", {})
    if not isinstance(raw_fields, Mapping):
        raise ValidationError("fields must be an object")
    fields: dict[str, str] = {}
    for name, value in raw_fields.items():
        if name not in schema.text_fields:
            raise ValidationError(f"unknown text field {name!r}")
        if not isinstance(value, str):
            raise ValidationError(f"field {name!r} must be a string")
        if len(value) > MAX_TEXT_CHARS:
            raise ValidationError(f"field {name!r} exceeds {MAX_TEXT_CHARS} characters")
        fields[name] = value
    if not any(v.strip() for v in fields.values()):
        raise ValidationError("document needs at least one non-empty text field")

    raw_meta = raw.get("meta", {})
    if not isinstance(raw_meta, Mapping):
        raise ValidationError("meta must be an object")
    keywords: dict[str, tuple[str, ...]] = {}
    numbers: dict[str, float] = {}
    for name, value in raw_meta.items():
        if name in schema.keyword_fields:
            values = value if isinstance(value, list) else [value]
            if len(values) > MAX_KEYWORD_VALUES or not all(
                isinstance(v, str) and v for v in values
            ):
                raise ValidationError(f"meta {name!r} must be non-empty string(s)")
            keywords[name] = tuple(dict.fromkeys(values))
        elif name in schema.number_fields:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValidationError(f"meta {name!r} must be a number")
            if not math.isfinite(value):
                raise ValidationError(f"meta {name!r} must be finite")
            numbers[name] = float(value)
        else:
            raise ValidationError(f"unknown meta field {name!r}")
    return Document(doc_id, fields, keywords, numbers)
