from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from hybridsearch.api import create_app
from hybridsearch.config import Settings
from hybridsearch.index import Index

DOCS = {
    "documents": [
        {
            "id": "e1",
            "fields": {"title": "Espresso machine", "body": "Nine bars of pressure for espresso."},
            "meta": {"category": "coffee", "year": 2021},
        },
        {
            "id": "v1",
            "fields": {"title": "Volcano <script>alert(1)</script>", "body": "Lava and ash."},
            "meta": {"category": "earth", "year": 2020},
        },
    ]
}


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(index_dir=tmp_path / "ix", fsync=False, embedding_dim=64)


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings=settings)) as c:
        yield c


def test_health_and_stats(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok", "documents": 0}
    stats = client.get("/stats").json()
    assert stats["documents"] == 0
    assert stats["language"] == "en"


def test_document_crud(client: TestClient) -> None:
    assert client.post("/documents", json=DOCS).json() == {"indexed": 2}
    got = client.get("/documents/e1").json()
    assert got["fields"]["title"] == "Espresso machine"
    assert client.get("/documents/missing").status_code == 404
    assert client.delete("/documents/e1").json() == {"deleted": True}
    assert client.delete("/documents/e1").status_code == 404
    assert client.get("/healthz").json()["documents"] == 1


def test_search_modes_facets_filters_and_shape(client: TestClient) -> None:
    client.post("/documents", json=DOCS)
    body = {"query": "espresso pressure", "mode": "hybrid", "facets": ["category"]}
    res = client.post("/search", json=body).json()
    assert res["hits"][0]["id"] == "e1"
    assert res["facets"]["category"][0] == {"value": "coffee", "count": 1}
    assert "<mark>" in res["hits"][0]["snippet"]
    assert res["hits"][0]["lexical_rank"] == 1
    assert res["took_ms"] >= 0
    filtered = client.post(
        "/search",
        json={"query": "lava espresso", "mode": "lexical", "filters": {"year": {"gte": 2021}}},
    ).json()
    assert [h["id"] for h in filtered["hits"]] == ["e1"]


@pytest.mark.parametrize(
    ("payload", "status", "code"),
    [
        ({"query": '"open'}, 400, "query_syntax_error"),
        ({"query": "x", "k": 0}, 400, "validation_error"),
        ({"query": "x", "filters": {"zzz": 1}}, 400, "validation_error"),
        ({"query": "x", "facets": ["year"]}, 400, "validation_error"),
        ({"query": "x", "mode": "fuzzy"}, 422, None),
        ({"query": "x", "surprise": True}, 422, None),
    ],
)
def test_error_mapping(
    client: TestClient, payload: dict[str, Any], status: int, code: str | None
) -> None:
    res = client.post("/search", json=payload)
    assert res.status_code == status
    if code:
        assert res.json()["error"]["code"] == code


def test_invalid_documents_rejected(client: TestClient) -> None:
    bad = {"documents": [{"id": "x", "fields": {"nope": "y"}}]}
    res = client.post("/documents", json=bad)
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "validation_error"
    assert client.post("/documents", json={"documents": []}).status_code == 422
    assert client.get("/healthz").json()["documents"] == 0


def test_flush_and_merge_endpoints(client: TestClient) -> None:
    client.post("/documents", json=DOCS)
    flushed = client.post("/flush").json()
    assert flushed["memtable_documents"] == 0
    assert len(flushed["segments"]) == 1
    merged = client.post("/merge", params={"force": "true"}).json()
    assert merged["merges"] == 0


def test_ui_renders_results_and_escapes_html(client: TestClient) -> None:
    client.post("/documents", json=DOCS)
    page = client.get("/", params={"q": "volcano lava", "mode": "lexical"})
    assert page.status_code == 200
    assert "<mark>Volcano</mark>" in page.text
    assert "<script>alert(1)</script>" not in page.text
    assert "&lt;script&gt;" in page.text
    assert "category" in page.text
    empty = client.get("/")
    assert "2 documents" in empty.text
    assert "No results" in client.get("/", params={"q": "zzzzqqq", "mode": "lexical"}).text
    err = client.get("/", params={"q": '"broken', "mode": "lexical"})
    assert err.status_code == 200
    assert "unterminated" in err.text
    filtered = client.get(
        "/", params={"q": "espresso volcano", "mode": "lexical", "category": "coffee"}
    )
    assert "Espresso" in filtered.text
    assert "Volcano" not in filtered.text


def test_lifespan_flushes_on_shutdown(settings: Settings) -> None:
    with TestClient(create_app(settings=settings)) as c:
        c.post("/documents", json=DOCS)
        assert c.get("/stats").json()["memtable_documents"] == 2
    with Index.open(settings.index_dir) as ix:
        assert ix.stats()["memtable_documents"] == 0
        assert len(ix.stats()["segments"]) == 1
        assert ix.count() == 2


def test_injected_index_is_not_closed_by_the_app(settings: Settings) -> None:
    ix = Index.open(settings.index_dir, settings.schema(), options=settings.index_options())
    with TestClient(create_app(index=ix)) as c:
        c.post("/documents", json=DOCS)
    assert ix.count() == 2
    ix.close()


def test_settings_from_env(tmp_path: Path) -> None:
    s = Settings.from_env(
        {
            "HYBRID_SEARCH_DIR": str(tmp_path),
            "HYBRID_SEARCH_FSYNC": "0",
            "HYBRID_SEARCH_LANGUAGE": "pt",
            "HYBRID_SEARCH_KEYWORDS": "a, b",
            "HYBRID_SEARCH_NUMBERS": "",
            "HYBRID_SEARCH_DIM": "64",
        }
    )
    assert (s.index_dir, s.fsync, s.language) == (tmp_path, False, "pt")
    assert s.keyword_fields == ("a", "b")
    assert s.number_fields == ()
    assert s.schema().embedding_dim == 64
    assert Settings.from_env({}).fsync is True
