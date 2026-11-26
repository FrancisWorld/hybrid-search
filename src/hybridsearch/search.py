"""Search orchestration: lexical, vector and hybrid retrieval with filters and facets."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from hybridsearch.analysis import Analyzer
from hybridsearch.embedder import Embedder
from hybridsearch.errors import ValidationError
from hybridsearch.filters import evaluate, parse_filters
from hybridsearch.fusion import reciprocal_rank_fusion
from hybridsearch.highlight import (
    Targets,
    collect_targets,
    find_spans,
    render_fragment,
    terms_only_targets,
)
from hybridsearch.query import analyze_query, parse_query, positive_text
from hybridsearch.schema import Schema
from hybridsearch.searcher import Bm25Params, LexicalSearcher
from hybridsearch.snapshot import Snapshot

Mode = Literal["lexical", "vector", "hybrid"]
MAX_CANDIDATES = 1000
MAX_PAGE = 100
Key = tuple[int, int]  # (view index, local doc id)


@dataclass(frozen=True)
class SearchRequest:
    query: str
    mode: Mode = "hybrid"
    k: int = 10
    offset: int = 0
    filters: Mapping[str, object] | None = None
    facets: Sequence[str] = ()
    facet_size: int = 10
    highlight: bool = True
    candidates: int = 100
    rrf_k: int = 60
    min_similarity: float = 0.1

    def __post_init__(self) -> None:
        if self.mode not in ("lexical", "vector", "hybrid"):
            raise ValidationError(f"unknown mode {self.mode!r}")
        if not 1 <= self.k <= MAX_PAGE:
            raise ValidationError(f"k must be between 1 and {MAX_PAGE}")
        if self.offset < 0:
            raise ValidationError("offset must be >= 0")
        if not 1 <= self.candidates <= MAX_CANDIDATES:
            raise ValidationError(f"candidates must be between 1 and {MAX_CANDIDATES}")
        if self.offset + self.k > self.candidates:
            raise ValidationError("offset + k cannot exceed candidates")
        if self.rrf_k < 0 or self.facet_size < 1:
            raise ValidationError("rrf_k must be >= 0 and facet_size >= 1")
        if not self.query.strip():
            raise ValidationError("query must not be empty")


@dataclass
class Hit:
    id: str
    score: float
    document: dict[str, Any]
    lexical_rank: int | None = None
    vector_rank: int | None = None
    lexical_score: float | None = None
    vector_similarity: float | None = None
    highlights: dict[str, str] = field(default_factory=dict)
    snippet: str = ""


@dataclass
class SearchResult:
    hits: list[Hit]
    total: int
    mode: Mode
    facets: dict[str, list[dict[str, Any]]]
    took_ms: float


@dataclass
class _Ranked:
    keys: list[Key]
    scores: dict[Key, float]
    match_masks: list[NDArray[np.bool_]]  # per view: every doc that matched


def run_search(
    snapshot: Snapshot,
    schema: Schema,
    analyzer: Analyzer,
    embedder: Embedder,
    request: SearchRequest,
    bm25: Bm25Params | None = None,
) -> SearchResult:
    started = time.perf_counter()
    conditions = parse_filters(request.filters, schema)
    for name in request.facets:
        if name not in schema.keyword_fields:
            raise ValidationError(
                f"cannot facet on {name!r}; keyword fields: {list(schema.keyword_fields)}"
            )
    views = snapshot.views
    allowed = [v.live & evaluate(conditions, v.segment) for v in views]

    targets = Targets()
    plain = request.query
    lexical: _Ranked | None = None
    if request.mode != "vector":
        raw = parse_query(request.query)
        plain = positive_text(raw)
        node = analyze_query(raw, analyzer)
        targets = collect_targets(node)
        lexical = _lexical(snapshot, schema, bm25, node, allowed, request.candidates)
    else:
        targets = terms_only_targets(t.term for t in analyzer.analyze(request.query))
    vector = (
        _vector(snapshot, embedder, plain, allowed, request) if request.mode != "lexical" else None
    )

    ranked: list[tuple[Key, float]]
    if lexical is not None and vector is not None:
        ranked = reciprocal_rank_fusion([lexical.keys, vector.keys], k=request.rrf_k)
    else:
        single = lexical or vector
        assert single is not None
        ranked = [(key, single.scores[key]) for key in single.keys]

    pools = [p for p in (lexical, vector) if p is not None]
    facet_masks = [
        np.logical_or.reduce([p.match_masks[i] for p in pools]) for i in range(len(views))
    ]
    total = int(sum(m.sum() for m in facet_masks))
    page = ranked[request.offset : request.offset + request.k]

    lex_rank = {k: r for r, k in enumerate(lexical.keys, 1)} if lexical else {}
    vec_rank = {k: r for r, k in enumerate(vector.keys, 1)} if vector else {}
    hits = []
    for key, score in page:
        segment = views[key[0]].segment
        hit = Hit(
            id=segment.ids[key[1]],
            score=score,
            document=segment.stored(key[1]),
            lexical_rank=lex_rank.get(key),
            vector_rank=vec_rank.get(key),
            lexical_score=lexical.scores.get(key) if lexical else None,
            vector_similarity=vector.scores.get(key) if vector else None,
        )
        if request.highlight:
            _highlight(hit, schema, analyzer, targets)
        hits.append(hit)

    facets = _facets(snapshot, request, facet_masks)
    return SearchResult(hits, total, request.mode, facets, (time.perf_counter() - started) * 1000)


def _lexical(
    snapshot: Snapshot,
    schema: Schema,
    bm25: Bm25Params | None,
    node: Any,
    allowed: list[NDArray[np.bool_]],
    limit: int,
) -> _Ranked:
    masks = [np.zeros(v.segment.n_docs, dtype=bool) for v in snapshot.views]
    if node is None:
        return _Ranked([], {}, masks)
    searcher = LexicalSearcher(snapshot, schema, bm25)
    seg_ids: list[NDArray[np.int64]] = []
    docs: list[NDArray[np.int64]] = []
    scores: list[NDArray[np.float64]] = []
    for i, view in enumerate(snapshot.views):
        d, s = searcher.evaluate(view, node)
        keep = allowed[i][d]
        d, s = d[keep], s[keep]
        masks[i][d] = True
        seg_ids.append(np.full(len(d), i, dtype=np.int64))
        docs.append(d)
        scores.append(s)
    if not docs:
        return _Ranked([], {}, masks)
    all_segs, all_docs, all_scores = (np.concatenate(x) for x in (seg_ids, docs, scores))
    order = np.argsort(-all_scores, kind="stable")[:limit]
    keys = [(int(all_segs[i]), int(all_docs[i])) for i in order]
    return _Ranked(keys, {k: float(all_scores[i]) for k, i in zip(keys, order, strict=True)}, masks)


def _vector(
    snapshot: Snapshot,
    embedder: Embedder,
    text: str,
    allowed: list[NDArray[np.bool_]],
    request: SearchRequest,
) -> _Ranked:
    masks = [np.zeros(v.segment.n_docs, dtype=bool) for v in snapshot.views]
    query = embedder.embed([text])[0]
    if not query.any():
        return _Ranked([], {}, masks)
    found: list[tuple[float, Key]] = []
    ef = max(64, request.candidates)
    for i, view in enumerate(snapshot.views):
        ids, dists = view.segment.vector_search(query, request.candidates, allowed[i], ef)
        for local, dist in zip(ids.tolist(), dists.tolist(), strict=True):
            similarity = 1.0 - dist
            if similarity >= request.min_similarity:
                found.append((similarity, (i, local)))
    found.sort(key=lambda item: -item[0])
    found = found[: request.candidates]
    for _, (i, local) in found:
        masks[i][local] = True
    return _Ranked([k for _, k in found], {k: s for s, k in found}, masks)


def _highlight(hit: Hit, schema: Schema, analyzer: Analyzer, targets: Targets) -> None:
    best_marks = -1
    for name in schema.text_fields:
        text = hit.document["fields"].get(name, "")
        if not text:
            continue
        terms, phrases = targets.for_field(name)
        spans = find_spans(text, analyzer, terms, phrases)
        if not spans:
            continue
        fragment, marks = render_fragment(text, spans)
        hit.highlights[name] = fragment
        if marks > best_marks:
            best_marks, hit.snippet = marks, fragment
    if not hit.snippet:
        longest = max(hit.document["fields"].values(), key=len, default="")
        hit.snippet = render_fragment(longest, [])[0]


def _facets(
    snapshot: Snapshot, request: SearchRequest, masks: list[NDArray[np.bool_]]
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for name in request.facets:
        counts: Counter[str] = Counter()
        for view, mask in zip(snapshot.views, masks, strict=True):
            for value, local_ids in view.segment.keyword_values(name).items():
                n = int(mask[local_ids].sum())
                if n:
                    counts[value] += n
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[: request.facet_size]
        out[name] = [{"value": v, "count": c} for v, c in ordered]
    return out
