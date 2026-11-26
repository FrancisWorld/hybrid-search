"""BM25, boolean and phrase evaluation checked against a naive reference."""

from __future__ import annotations

import itertools
import math
import random
from collections import Counter

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hybridsearch.analysis import LanguageAnalyzer, get_analyzer
from hybridsearch.embedder import HashingEmbedder
from hybridsearch.errors import QueryError
from hybridsearch.query import analyze_query, parse_query
from hybridsearch.schema import Document, Schema, parse_document
from hybridsearch.searcher import LexicalSearcher
from hybridsearch.snapshot import Snapshot
from tests.helpers import VOCAB, make_snapshot, random_docs

K1, B = 1.2, 0.75


class Naive:
    """Straight-from-the-formula BM25 over raw documents, no index structures."""

    def __init__(self, docs: list[Document], schema: Schema, analyzer: LanguageAnalyzer) -> None:
        self.schema = schema
        self.tokens: dict[str, dict[str, list[tuple[str, int]]]] = {}
        for d in docs:
            self.tokens[d.id] = {
                f: [(t.term, t.position) for t in analyzer.analyze(d.fields.get(f, ""))]
                for f in schema.text_fields
            }
        self.n = len(docs)
        self.avg = {
            f: sum(len(t[f]) for t in self.tokens.values()) / self.n for f in schema.text_fields
        }

    def df(self, field: str, term: str) -> int:
        return sum(any(t == term for t, _ in toks[field]) for toks in self.tokens.values())

    def score_term(self, doc: str, term: str, field: str | None = None) -> float:
        total = 0.0
        for f in [field] if field else list(self.schema.text_fields):
            toks = self.tokens[doc][f]
            tf = Counter(t for t, _ in toks)[term]
            if not tf:
                continue
            df = self.df(f, term)
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            norm = K1 * (1 - B + B * len(toks) / self.avg[f])
            total += self.schema.text_fields[f] * idf * tf * (K1 + 1) / (tf + norm)
        return total

    def phrase_freq(self, doc: str, terms: list[str], offsets: list[int], field: str) -> int:
        toks = self.tokens[doc][field]
        by_pos: dict[int, str] = {p: t for t, p in toks}
        return sum(
            all(by_pos.get(p + off) == t for t, off in zip(terms, offsets, strict=True))
            for p in by_pos
            if by_pos[p] == terms[0]
        )


def run(
    snapshot: Snapshot, schema: Schema, query: str, analyzer: LanguageAnalyzer
) -> dict[str, float]:
    node = analyze_query(parse_query(query), analyzer)
    assert node is not None
    searcher = LexicalSearcher(snapshot, schema)
    out: dict[str, float] = {}
    for view in snapshot.views:
        docs, scores = searcher.evaluate(view, node)
        for d, s in zip(docs.tolist(), scores.tolist(), strict=True):
            out[view.segment.ids[d]] = s
    return out


@pytest.fixture
def corpus(schema: Schema) -> list[Document]:
    return random_docs(schema, 90, seed=11)


@pytest.mark.parametrize("splits", [[90], [30, 30, 30], [5, 80, 5]])
def test_term_scores_match_naive_bm25_regardless_of_segmentation(
    schema: Schema,
    analyzer: LanguageAnalyzer,
    embedder: HashingEmbedder,
    corpus: list[Document],
    splits: list[int],
) -> None:
    bounds = np.cumsum([0, *splits])
    groups = [corpus[a:b] for a, b in itertools.pairwise(bounds.tolist())]
    snap = make_snapshot(groups, schema, analyzer, embedder)
    naive = Naive(corpus, schema, analyzer)
    rng = random.Random(5)
    for _ in range(25):
        words = rng.sample(VOCAB, k=rng.randint(1, 3))
        got = run(snap, schema, " ".join(words), analyzer)
        stems = [t.term for w in words for t in analyzer.analyze(w)]
        expected = {
            d.id: sum(naive.score_term(d.id, t) for t in stems)
            for d in corpus
            if any(naive.score_term(d.id, t) > 0 for t in stems)
        }
        assert got.keys() == expected.keys()
        for doc_id, score in expected.items():
            assert got[doc_id] == pytest.approx(score, rel=1e-9)


def test_field_restricted_term(
    schema: Schema, analyzer: LanguageAnalyzer, embedder: HashingEmbedder, corpus: list[Document]
) -> None:
    snap = make_snapshot([corpus], schema, analyzer, embedder)
    naive = Naive(corpus, schema, analyzer)
    got = run(snap, schema, "title:espresso", analyzer)
    stem = analyzer.analyze("espresso")[0].term
    assert got
    for doc_id, score in got.items():
        assert score == pytest.approx(naive.score_term(doc_id, stem, "title"))
    with pytest.raises(QueryError, match="unknown field"):
        run(snap, schema, "nofield:espresso", analyzer)


def test_phrase_matches_and_scores_against_reference(
    schema: Schema, analyzer: LanguageAnalyzer, embedder: HashingEmbedder, corpus: list[Document]
) -> None:
    snap = make_snapshot([corpus[:40], corpus[40:]], schema, analyzer, embedder)
    naive = Naive(corpus, schema, analyzer)
    phrases = []
    for doc in corpus[:12]:
        words = doc.fields["body"].split()
        phrases += [" ".join(words[:2]), " ".join(words[3:6])]
    for phrase in phrases:
        got = run(snap, schema, f'body:"{phrase}"', analyzer)
        stems = [t.term for t in analyzer.analyze(phrase)]
        offsets = list(range(len(stems)))
        expected = {d.id: naive.phrase_freq(d.id, stems, offsets, "body") for d in corpus}
        expected = {k: v for k, v in expected.items() if v}
        assert got.keys() == expected.keys(), phrase
        assert expected
        assert all(math.isfinite(s) and s > 0 for s in got.values())


def test_phrase_respects_order_and_stopword_gaps(
    schema: Schema, analyzer: LanguageAnalyzer, embedder: HashingEmbedder
) -> None:
    docs = [
        parse_document({"id": "a", "fields": {"body": "the king of spain arrived"}}, schema),
        parse_document({"id": "b", "fields": {"body": "the king and the spain arrived"}}, schema),
        parse_document({"id": "c", "fields": {"body": "spain king"}}, schema),
        parse_document({"id": "d", "fields": {"body": "king spain"}}, schema),
    ]
    snap = make_snapshot([docs], schema, analyzer, embedder)
    assert set(run(snap, schema, '"king of spain"', analyzer)) == {"a"}
    assert set(run(snap, schema, '"king spain"', analyzer)) == {"d"}
    assert set(run(snap, schema, '"spain king"', analyzer)) == {"c"}
    assert set(run(snap, schema, "king spain", analyzer)) == {"a", "b", "c", "d"}


def test_boolean_semantics_against_set_logic(
    schema: Schema, analyzer: LanguageAnalyzer, embedder: HashingEmbedder, corpus: list[Document]
) -> None:
    snap = make_snapshot([corpus[:50], corpus[50:]], schema, analyzer, embedder)

    def ids(q: str) -> set[str]:
        return set(run(snap, schema, q, analyzer))

    a, b, c = ids("lava"), ids("ocean"), ids("island")
    assert a
    assert b
    assert c
    assert ids("lava ocean") == a | b
    assert ids("lava AND ocean") == a & b
    assert ids("+lava ocean") == a
    assert ids("lava -ocean") == a - b
    assert ids("lava NOT ocean") == a - b
    assert ids("(lava OR ocean) AND NOT island") == (a | b) - c
    assert ids("lava AND ocean AND island") == a & b & c
    assert ids("+lava -ocean island") == a - b


def test_must_with_should_boosts_scores(
    schema: Schema, analyzer: LanguageAnalyzer, embedder: HashingEmbedder, corpus: list[Document]
) -> None:
    snap = make_snapshot([corpus], schema, analyzer, embedder)
    plain = run(snap, schema, "lava", analyzer)
    boosted = run(snap, schema, "+lava ocean", analyzer)
    assert plain.keys() == boosted.keys()
    assert all(boosted[k] >= plain[k] - 1e-12 for k in plain)
    assert any(boosted[k] > plain[k] for k in plain)


def test_deleted_documents_never_match_and_stats_follow(
    schema: Schema, analyzer: LanguageAnalyzer, embedder: HashingEmbedder, corpus: list[Document]
) -> None:
    full = make_snapshot([corpus], schema, analyzer, embedder)
    hits = run(full, schema, "espresso", analyzer)
    victims = [full.views[0].segment.id_to_local[i] for i in list(hits)[:3]]
    pruned = make_snapshot([corpus], schema, analyzer, embedder, deleted={0: victims})
    after = run(pruned, schema, "espresso", analyzer)
    assert set(hits) - set(after) == {corpus[v].id for v in victims}
    assert pruned.n_docs == full.n_docs - 3
    assert pruned.total_len("body") < full.total_len("body")


def test_unanalysed_node_is_a_programming_error(
    schema: Schema, analyzer: LanguageAnalyzer, embedder: HashingEmbedder, corpus: list[Document]
) -> None:
    snap = make_snapshot([corpus], schema, analyzer, embedder)
    with pytest.raises(TypeError, match="not analysed"):
        LexicalSearcher(snap, schema).evaluate(snap.views[0], parse_query("lava"))


def test_empty_index_snapshot(schema: Schema) -> None:
    searcher = LexicalSearcher(Snapshot(()), schema)
    assert searcher.idf("body", "x") == pytest.approx(math.log(1 + 0.5 / 0.5))


@settings(max_examples=40, deadline=None)
@given(
    st.lists(st.sampled_from(VOCAB[:8]), min_size=1, max_size=3),
    st.lists(st.sampled_from(VOCAB[:8]), min_size=0, max_size=2),
    st.integers(1, 5),
)
def test_must_and_must_not_match_naive_set_logic(
    must_words: list[str], not_words: list[str], seed: int
) -> None:
    schema = Schema(
        text_fields={"title": 2.0, "body": 1.0},
        keyword_fields=("category", "tags"),
        number_fields=("year",),
        embedding_dim=16,
    )
    analyzer_ = get_analyzer("en")
    emb = HashingEmbedder(analyzer_, dim=16)
    docs = random_docs(schema, 40, seed=seed)
    snap = make_snapshot([docs[:15], docs[15:]], schema, analyzer_, emb)
    query = " ".join(f"+{w}" for w in must_words) + "".join(f" -{w}" for w in not_words)
    got = set(run(snap, schema, query, analyzer_))
    naive = Naive(docs, schema, analyzer_)

    def has(doc: str, word: str) -> bool:
        stem = analyzer_.analyze(word)[0].term
        return naive.score_term(doc, stem) > 0

    expected = {
        d.id
        for d in docs
        if all(has(d.id, w) for w in must_words) and not any(has(d.id, w) for w in not_words)
    }
    assert got == expected
