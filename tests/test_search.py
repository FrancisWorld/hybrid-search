"""End-to-end search behaviour: modes, fusion, filters, facets, highlighting, pagination."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from hybridsearch.errors import QueryError, ValidationError
from hybridsearch.index import Index, IndexOptions
from hybridsearch.schema import Schema
from hybridsearch.search import SearchRequest
from hybridsearch.segment import SegmentOptions
from tests.helpers import random_docs

DOCS: list[tuple[str, str, str, str, int]] = [
    (
        "e1",
        "Espresso machine pressure",
        "A good espresso machine holds nine bars of pressure.",
        "coffee",
        2021,
    ),
    (
        "e2",
        "Grinder burrs",
        "Burr grinders give an even grind for espresso and filter coffee.",
        "coffee",
        2019,
    ),
    (
        "e3",
        "Moka pot guide",
        "The moka pot brews strong coffee on the stove without a machine.",
        "coffee",
        2023,
    ),
    ("v1", "Volcano eruptions", "Lava and ash leave the crater during an eruption.", "earth", 2020),
    (
        "v2",
        "Island volcanoes",
        "Volcanic islands rise from the ocean as magma cools.",
        "earth",
        2022,
    ),
    (
        "m1",
        "Marathon training",
        "Long runs build the pace needed to finish a marathon.",
        "sport",
        2018,
    ),
    (
        "m2",
        "Running shoes",
        "Runners replace their shoes every six hundred kilometres.",
        "sport",
        2023,
    ),
    ("s1", "Solar panels", "Solar panels turn sunlight into power for the grid.", "energy", 2021),
]


@pytest.fixture
def ix(tmp_path: Path) -> Iterator[Index]:
    schema = Schema(
        text_fields={"title": 2.0, "body": 1.0},
        keyword_fields=("category",),
        number_fields=("year",),
        embedding_dim=128,
    )
    index = Index.open(tmp_path / "ix", schema, options=IndexOptions(fsync=False))
    index.put_many(
        {
            "id": i,
            "fields": {"title": t, "body": b},
            "meta": {"category": c, "year": y},
        }
        for i, t, b, c, y in DOCS
    )
    yield index
    index.close()


def search(ix: Index, query: str, **kw: Any) -> Any:
    return ix.search(SearchRequest(query, **kw))


def test_lexical_ranking_prefers_title_matches_and_stems(ix: Index) -> None:
    res = search(ix, "espresso machine", mode="lexical")
    assert [h.id for h in res.hits] == ["e1", "e2", "e3"]  # title+body hit > body-only hits
    assert res.hits[0].lexical_rank == 1
    assert res.hits[0].vector_rank is None
    assert res.hits[0].lexical_score is not None
    assert res.hits[0].lexical_score > res.hits[1].lexical_score
    assert {h.id for h in search(ix, "running", mode="lexical").hits} == {"m1", "m2"}  # stem: run
    assert {h.id for h in search(ix, "runners", mode="lexical").hits} == {"m2"}
    assert {h.id for h in search(ix, "volcano", mode="lexical").hits} == {"v1", "v2"}


def test_vector_finds_typos_that_lexical_misses_and_hybrid_recovers_them(ix: Index) -> None:
    query = "espreso presure"
    assert search(ix, query, mode="lexical").hits == []
    vector = search(ix, query, mode="vector")
    assert vector.hits[0].id == "e1"
    hybrid = search(ix, query, mode="hybrid")
    assert hybrid.hits[0].id == "e1"
    assert hybrid.hits[0].lexical_rank is None
    assert hybrid.hits[0].vector_rank == 1


def test_hybrid_fuses_with_rrf_formula(ix: Index) -> None:
    res = search(ix, "espresso coffee grinder", mode="hybrid")
    both = [h for h in res.hits if h.lexical_rank and h.vector_rank]
    assert both
    for h in res.hits:
        expected = sum(1 / (60 + r) for r in (h.lexical_rank, h.vector_rank) if r)
        assert h.score == pytest.approx(expected)
    assert [h.score for h in res.hits] == sorted((h.score for h in res.hits), reverse=True)
    custom = search(ix, "espresso coffee grinder", mode="hybrid", rrf_k=1)
    top = custom.hits[0]
    assert top.score == pytest.approx(
        sum(1 / (1 + r) for r in (top.lexical_rank, top.vector_rank) if r)
    )


def test_filters_apply_to_every_mode(ix: Index) -> None:
    for mode in ("lexical", "vector", "hybrid"):
        res = search(
            ix,
            "machine pressure coffee",
            mode=mode,
            filters={"category": "coffee", "year": {"gte": 2021}},
        )
        assert res.hits
        assert {h.id for h in res.hits} <= {"e1", "e3"}
    res = search(
        ix,
        "ocean lava",
        mode="hybrid",
        filters={"category": ["earth", "sport"], "year": {"lt": 2022}},
    )
    assert {h.id for h in res.hits} <= {"v1", "m1"}
    assert search(ix, "coffee", filters={"year": {"gt": 3000}}).hits == []


def test_facets_count_the_filtered_match_set(ix: Index) -> None:
    res = search(
        ix, "coffee OR espresso OR volcano OR marathon", mode="lexical", facets=["category"]
    )
    assert res.facets["category"] == [
        {"value": "coffee", "count": 3},
        {"value": "earth", "count": 2},
        {"value": "sport", "count": 1},
    ]
    assert res.total == 6
    filtered = search(
        ix,
        "coffee OR espresso OR volcano OR marathon",
        mode="lexical",
        facets=["category"],
        filters={"year": {"gte": 2021}},
    )
    assert filtered.facets["category"] == [
        {"value": "coffee", "count": 2},
        {"value": "earth", "count": 1},
    ]
    limited = search(
        ix, "coffee OR volcano OR marathon", mode="lexical", facets=["category"], facet_size=1
    )
    assert len(limited.facets["category"]) == 1


def test_facet_and_filter_validation(ix: Index) -> None:
    with pytest.raises(ValidationError, match="cannot facet"):
        search(ix, "coffee", facets=["year"])
    with pytest.raises(ValidationError, match="unknown field"):
        search(ix, "coffee", filters={"nope": 1})


def test_highlights_and_snippets(ix: Index) -> None:
    res = search(ix, "espresso machine", mode="lexical")
    hit = res.hits[0]
    assert hit.highlights["title"] == "<mark>Espresso</mark> <mark>machine</mark> pressure"
    assert "<mark>espresso</mark>" in hit.highlights["body"]
    assert "<mark>" in hit.snippet
    phrase = search(ix, '"machine pressure"', mode="lexical")
    assert [h.id for h in phrase.hits] == ["e1"]
    assert (
        phrase.hits[0].highlights["title"] == "Espresso <mark>machine</mark> <mark>pressure</mark>"
    )
    plain = search(ix, "espresso", mode="lexical", highlight=False)
    assert plain.hits[0].highlights == {}
    assert plain.hits[0].snippet == ""


def test_snippet_for_vector_hit_without_lexical_match_falls_back_to_text(ix: Index) -> None:
    res = search(ix, "espreso presure", mode="vector")
    assert res.hits[0].snippet
    assert "<mark>" not in res.hits[0].snippet


def test_pagination_is_consistent(ix: Index) -> None:
    full = search(
        ix, "coffee volcano marathon solar espresso island", mode="lexical", k=8, candidates=50
    )
    p1 = search(
        ix, "coffee volcano marathon solar espresso island", mode="lexical", k=3, candidates=50
    )
    p2 = search(
        ix,
        "coffee volcano marathon solar espresso island",
        mode="lexical",
        k=3,
        offset=3,
        candidates=50,
    )
    assert [h.id for h in p1.hits + p2.hits] == [h.id for h in full.hits][:6]
    assert p1.total == full.total


def test_query_language_through_the_engine(ix: Index) -> None:
    assert {h.id for h in search(ix, "+coffee -moka", mode="lexical").hits} == {"e2"}
    assert {h.id for h in search(ix, "title:grinder", mode="lexical").hits} == {"e2"}
    assert {h.id for h in search(ix, "body:grinder", mode="lexical").hits} == {"e2"}
    assert search(ix, "the of and", mode="lexical").hits == []  # only stop words
    assert search(ix, "!!!", mode="vector").hits == []  # no features -> zero vector
    with pytest.raises(QueryError, match="unterminated"):
        search(ix, '"broken', mode="hybrid")
    with pytest.raises(QueryError, match="unknown field"):
        search(ix, "nofield:x", mode="lexical")
    # vector mode treats the text as free text, so operators are never a syntax error
    search(ix, '"broken', mode="vector")


def test_negative_terms_do_not_steer_the_vector_side(ix: Index) -> None:
    res = search(ix, "coffee -espresso", mode="hybrid")
    assert "e1" not in {h.id for h in res.hits if h.lexical_rank}


def test_min_similarity_filters_weak_vector_hits(ix: Index) -> None:
    strict = search(ix, "espresso machine", mode="vector", min_similarity=0.99)
    assert strict.hits == []
    loose = search(ix, "espresso machine", mode="vector", min_similarity=0.0)
    assert len(loose.hits) >= 3


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"mode": "fuzzy"}, "unknown mode"),
        ({"k": 0}, "k must"),
        ({"k": 101, "candidates": 1000}, "k must"),
        ({"offset": -1}, "offset"),
        ({"candidates": 0}, "candidates"),
        ({"k": 10, "offset": 95, "candidates": 100}, "cannot exceed"),
        ({"rrf_k": -1}, "rrf_k"),
        ({"query": "  "}, "empty"),
        ({"query": "a" * 2049, "mode": "vector"}, "longer than"),
    ],
)
def test_request_validation(kwargs: dict[str, Any], fragment: str) -> None:
    params: dict[str, Any] = {"query": "x"}
    params.update(kwargs)
    with pytest.raises(ValidationError, match=fragment):
        SearchRequest(**params)


def test_hnsw_segments_with_filters_and_tombstones(tmp_path: Path) -> None:
    schema = Schema(
        text_fields={"title": 2.0, "body": 1.0},
        keyword_fields=("category", "tags"),
        number_fields=("year",),
        embedding_dim=64,
    )
    options = IndexOptions(
        fsync=False, memtable_max_docs=100, segment=SegmentOptions(hnsw_min_docs=50)
    )
    with Index.open(tmp_path / "ix", schema, options=options) as index:
        docs = random_docs(schema, 300, seed=4)
        index.put_many([d.to_dict() for d in docs])
        index.flush()
        assert any(s["hnsw"] for s in index.stats()["segments"])
        victims = [d.id for d in docs[:30]]
        index.delete_many(victims)
        res = index.search(
            SearchRequest(
                "espresso pressure grinder",
                mode="vector",
                k=20,
                filters={"category": "coffee"},
                min_similarity=-1.0,
            )
        )
        assert len(res.hits) == 20
        assert all(h.document["meta"]["category"] == ["coffee"] for h in res.hits)
        assert not {h.id for h in res.hits} & set(victims)
