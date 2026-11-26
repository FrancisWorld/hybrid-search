from __future__ import annotations

import pytest

from hybridsearch.errors import ValidationError
from hybridsearch.schema import Schema, parse_document


def test_valid_document_normalises_meta(schema: Schema) -> None:
    doc = parse_document(
        {
            "id": "x",
            "fields": {"title": "T", "body": "B"},
            "meta": {"category": "coffee", "tags": ["a", "a", "b"], "year": 2020},
        },
        schema,
    )
    assert doc.keywords == {"category": ("coffee",), "tags": ("a", "b")}
    assert doc.numbers == {"year": 2020.0}
    assert doc.text == "T\nB"
    assert doc.to_dict()["meta"] == {"category": ["coffee"], "tags": ["a", "b"], "year": 2020.0}


@pytest.mark.parametrize(
    ("raw", "fragment"),
    [
        ("nope", "must be an object"),
        ({"fields": {"body": "x"}}, "id must be"),
        ({"id": "", "fields": {"body": "x"}}, "id must be"),
        ({"id": "x" * 300, "fields": {"body": "x"}}, "id must be"),
        ({"id": "a", "fields": {"body": "x"}, "extra": 1}, "unknown document keys"),
        ({"id": "a", "fields": {"nope": "x"}}, "unknown text field"),
        ({"id": "a", "fields": {"body": 3}}, "must be a string"),
        ({"id": "a", "fields": {"body": "  "}}, "at least one non-empty"),
        ({"id": "a", "fields": []}, "fields must be an object"),
        ({"id": "a", "fields": {"body": "x"}, "meta": []}, "meta must be an object"),
        ({"id": "a", "fields": {"body": "x"}, "meta": {"zzz": 1}}, "unknown meta field"),
        ({"id": "a", "fields": {"body": "x"}, "meta": {"year": "2020"}}, "must be a number"),
        ({"id": "a", "fields": {"body": "x"}, "meta": {"year": True}}, "must be a number"),
        ({"id": "a", "fields": {"body": "x"}, "meta": {"year": float("inf")}}, "finite"),
        ({"id": "a", "fields": {"body": "x"}, "meta": {"category": 5}}, "non-empty string"),
        ({"id": "a", "fields": {"body": "x"}, "meta": {"tags": ["ok", ""]}}, "non-empty string"),
    ],
)
def test_invalid_documents(schema: Schema, raw: object, fragment: str) -> None:
    with pytest.raises(ValidationError, match=fragment):
        parse_document(raw, schema)


def test_schema_validation_and_roundtrip(schema: Schema) -> None:
    assert Schema.from_dict(schema.to_dict()) == schema
    with pytest.raises(ValidationError, match="at least one text field"):
        Schema(text_fields={})
    with pytest.raises(ValidationError, match="invalid field name"):
        Schema(text_fields={"bad name": 1.0})
    with pytest.raises(ValidationError, match="unique"):
        Schema(text_fields={"a": 1.0}, keyword_fields=("a",))
    with pytest.raises(ValidationError, match="positive"):
        Schema(text_fields={"a": 0.0})
    with pytest.raises(ValidationError, match="invalid schema"):
        Schema.from_dict({"language": "en"})
