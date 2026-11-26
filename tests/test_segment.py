from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hybridsearch.analysis import LanguageAnalyzer
from hybridsearch.embedder import HashingEmbedder
from hybridsearch.errors import CorruptIndexError, ValidationError
from hybridsearch.filters import evaluate, parse_filters
from hybridsearch.merge import MergePolicy, merge_segments
from hybridsearch.schema import Schema, parse_document
from hybridsearch.segment import (
    Segment,
    SegmentOptions,
    analyze_documents,
    serialize_segment,
    write_segment_file,
)
from hybridsearch.segment_format import pack_sections, unpack_sections
from tests.helpers import build_segment, random_docs


def test_segment_roundtrip_postings_positions_and_stored(
    schema: Schema, analyzer: LanguageAnalyzer, embedder: HashingEmbedder
) -> None:
    docs = [
        parse_document(
            {"id": "a", "fields": {"title": "Espresso", "body": "pressure and more pressure"}},
            schema,
        ),
        parse_document({"id": "b", "fields": {"body": "pressure"}, "meta": {"year": 2020}}, schema),
    ]
    seg = build_segment(docs, schema, analyzer, embedder)
    assert seg.n_docs == 2
    assert seg.ids == ["a", "b"]
    plist = seg.postings("body", "pressur", with_positions=True)
    assert plist is not None
    assert plist.doc_ids.tolist() == [0, 1]
    assert plist.freqs.tolist() == [2, 1]
    assert plist.positions == [[0, 3], [0]]
    assert seg.postings("body", "absent") is None
    assert seg.doc_freq("body", "pressur") == 2
    assert seg.doc_freq("title", "pressur") == 0
    assert seg.stored(1)["meta"] == {"year": 2020.0}
    assert seg.field_lengths("body").tolist() == [3, 1]  # "and" is a stop word
    assert seg.total_len == {"title": 1, "body": 4}
    assert seg.id_to_local == {"a": 0, "b": 1}


def test_filters_over_columns(
    schema: Schema, analyzer: LanguageAnalyzer, embedder: HashingEmbedder
) -> None:
    docs = random_docs(schema, 60, seed=1)
    seg = build_segment(docs, schema, analyzer, embedder)

    def expect(pred) -> list[int]:  # type: ignore[no-untyped-def]
        return [i for i, d in enumerate(docs) if pred(d)]

    mask = evaluate(parse_filters({"category": "coffee"}, schema), seg)
    assert np.flatnonzero(mask).tolist() == expect(lambda d: d.keywords["category"] == ("coffee",))
    mask = evaluate(
        parse_filters({"category": ["coffee", "earth"], "year": {"gte": 2020, "lt": 2024}}, schema),
        seg,
    )
    assert np.flatnonzero(mask).tolist() == expect(
        lambda d: (
            d.keywords["category"][0] in ("coffee", "earth") and 2020 <= d.numbers["year"] < 2024
        )
    )
    mask = evaluate(parse_filters({"tags": "a"}, schema), seg)
    assert np.flatnonzero(mask).tolist() == expect(lambda d: "a" in d.keywords.get("tags", ()))
    mask = evaluate(parse_filters({"year": [2016, 2017]}, schema), seg)
    assert np.flatnonzero(mask).tolist() == expect(lambda d: d.numbers["year"] in (2016, 2017))
    assert evaluate((), seg).all()
    mask = evaluate(parse_filters({"year": {"eq": 2018, "gt": 2000}}, schema), seg)
    assert np.flatnonzero(mask).tolist() == expect(lambda d: d.numbers["year"] == 2018)


@pytest.mark.parametrize(
    ("raw", "fragment"),
    [
        ({"nope": "x"}, "unknown field"),
        ({"category": {"gt": 3}}, "range operator"),
        ({"category": 3}, "must be a string"),
        ({"year": "x"}, "must be a number"),
        ({"year": {"between": 3}}, "unknown filter operator"),
        ({"year": {"in": []}}, "non-empty list"),
        ({"year": {}}, "empty filter"),
        ({"year": True}, "must be a number"),
    ],
)
def test_filter_validation(schema: Schema, raw: dict[str, object], fragment: str) -> None:
    with pytest.raises(ValidationError, match=fragment):
        parse_filters(raw, schema)


def test_missing_numbers_never_match_range(
    schema: Schema, analyzer: LanguageAnalyzer, embedder: HashingEmbedder
) -> None:
    docs = [
        parse_document({"id": "a", "fields": {"body": "x"}}, schema),
        parse_document({"id": "b", "fields": {"body": "x"}, "meta": {"year": 2000}}, schema),
    ]
    seg = build_segment(docs, schema, analyzer, embedder)
    assert np.flatnonzero(
        evaluate(parse_filters({"year": {"lt": 3000}}, schema), seg)
    ).tolist() == [1]


def test_hnsw_only_built_above_threshold(
    schema: Schema, analyzer: LanguageAnalyzer, embedder: HashingEmbedder
) -> None:
    docs = random_docs(schema, 40)
    assert build_segment(docs, schema, analyzer, embedder).hnsw is None
    big = build_segment(docs, schema, analyzer, embedder, SegmentOptions(hnsw_min_docs=10))
    assert big.hnsw is not None
    q = embedder.embed(["espresso pressure"])[0]
    ids, _ = big.vector_search(q, 5, None, 32)
    assert len(ids) == 5


def test_file_roundtrip_and_close(
    schema: Schema, analyzer: LanguageAnalyzer, embedder: HashingEmbedder, tmp_path: Path
) -> None:
    docs = random_docs(schema, 30)
    data = analyze_documents(docs, embedder.embed([d.text for d in docs]), schema, analyzer)
    blob = serialize_segment(data, schema, SegmentOptions(hnsw_min_docs=10))
    path = tmp_path / "seg-1.seg"
    write_segment_file(path, blob)
    assert not list(tmp_path.glob("*.tmp"))
    seg = Segment.open(path)
    assert seg.ids == [d.id for d in docs]
    assert seg.stored(7)["id"] == "d7"
    seg.close()
    seg.close()


def test_corruption_is_detected(
    schema: Schema, analyzer: LanguageAnalyzer, embedder: HashingEmbedder
) -> None:
    docs = random_docs(schema, 5)
    data = analyze_documents(docs, embedder.embed([d.text for d in docs]), schema, analyzer)
    blob = serialize_segment(data, schema, SegmentOptions())
    with pytest.raises(CorruptIndexError, match="bad magic"):
        Segment(b"junk" * 10)
    with pytest.raises(CorruptIndexError, match="truncated"):
        Segment(blob[:-3])
    flipped = bytearray(blob)
    flipped[len(blob) // 2] ^= 0xFF
    with pytest.raises(CorruptIndexError, match="checksum"):
        Segment(bytes(flipped))
    toc_flipped = bytearray(blob)
    toc_flipped[-30] ^= 0xFF
    with pytest.raises(CorruptIndexError):
        Segment(bytes(toc_flipped))


def test_empty_segment_and_shape_mismatch_rejected(
    schema: Schema, analyzer: LanguageAnalyzer, embedder: HashingEmbedder
) -> None:
    with pytest.raises(ValueError, match="empty segment"):
        serialize_segment(
            analyze_documents([], embedder.embed([]), schema, analyzer), schema, SegmentOptions()
        )
    docs = random_docs(schema, 3)
    bad = analyze_documents(docs, np.zeros((3, 5), dtype=np.float32), schema, analyzer)
    with pytest.raises(ValueError, match="vector matrix"):
        serialize_segment(bad, schema, SegmentOptions())


def test_container_sections_aligned() -> None:
    blob = pack_sections({"a": b"123", "b": b"4567890"})
    sections = unpack_sections(blob)
    assert bytes(sections["a"]) == b"123"
    assert bytes(sections["b"]) == b"4567890"
    assert unpack_sections(blob, verify=False).keys() == {"a", "b"}


def test_merge_drops_deleted_and_matches_fresh_build(
    schema: Schema, analyzer: LanguageAnalyzer, embedder: HashingEmbedder
) -> None:
    docs_a = random_docs(schema, 25, seed=1, prefix="a")
    docs_b = random_docs(schema, 25, seed=2, prefix="b")
    seg_a = build_segment(docs_a, schema, analyzer, embedder)
    seg_b = build_segment(docs_b, schema, analyzer, embedder)
    live_a = np.ones(25, dtype=bool)
    live_a[[0, 7, 24]] = False
    live_b = np.ones(25, dtype=bool)
    live_b[[3, 4]] = False

    data = merge_segments([(seg_a, live_a), (seg_b, live_b)])
    merged = Segment(serialize_segment(data, schema, SegmentOptions()))

    survivors = [d for d, ok in zip(docs_a, live_a, strict=True) if ok] + [
        d for d, ok in zip(docs_b, live_b, strict=True) if ok
    ]
    fresh = build_segment(survivors, schema, analyzer, embedder)
    assert merged.ids == fresh.ids
    assert merged.total_len == fresh.total_len
    assert sorted(merged.terms()) == sorted(fresh.terms())
    for key in fresh.terms():
        a = merged.postings(*key, with_positions=True)
        b = fresh.postings(*key, with_positions=True)
        assert a is not None
        assert b is not None
        assert a.doc_ids.tolist() == b.doc_ids.tolist()
        assert a.positions == b.positions
    assert np.array_equal(merged.vectors, fresh.vectors)
    assert [merged.stored(i) for i in range(merged.n_docs)] == [
        fresh.stored(i) for i in range(fresh.n_docs)
    ]


def test_merge_guards(
    schema: Schema, analyzer: LanguageAnalyzer, embedder: HashingEmbedder
) -> None:
    with pytest.raises(ValueError, match="nothing"):
        merge_segments([])
    seg = build_segment(random_docs(schema, 3), schema, analyzer, embedder)
    other_schema = Schema(language="en", embedding_dim=64)
    other = build_segment(
        [parse_document({"id": "x", "fields": {"body": "hello"}}, other_schema)],
        other_schema,
        analyzer,
        embedder,
    )
    with pytest.raises(ValueError, match="different schemas"):
        merge_segments([(seg, np.ones(3, dtype=bool)), (other, np.ones(1, dtype=bool))])
    empty = merge_segments([(seg, np.zeros(3, dtype=bool))])
    assert empty.stored == []


def test_merge_policy_tiers_and_selection() -> None:
    policy = MergePolicy(factor=4, floor_docs=100, max_merge=6)
    assert [policy.tier(n) for n in (1, 399, 400, 1599, 1600)] == [0, 0, 1, 1, 2]
    assert policy.select([10, 20, 30]) == []
    assert policy.select([10, 500, 20, 30, 40, 700]) == [0, 2, 3, 4]
    assert policy.select([1] * 10) == [0, 1, 2, 3, 4, 5]
    assert policy.select([450, 460, 470, 480, 5, 6]) == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="merge factor"):
        MergePolicy(factor=1)
