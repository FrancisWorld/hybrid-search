"""Lexical retrieval: BM25 scoring, boolean combination and phrase matching.

Per segment a query node evaluates to ``(doc_ids, scores)``: sorted local ids
and float scores. Boolean nodes combine them with sorted-array set operations.

Scoring is Lucene-style BM25 per field, ``idf * tf (k1+1) / (tf + k1 (1 - b + b dl/avgdl))``
with ``idf = ln(1 + (N - df + 0.5) / (df + 0.5))``, multiplied by the field boost and
summed over fields. Collection statistics (N, df, avgdl) are global across segments,
so scores do not depend on how documents happen to be split into segments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hybridsearch.errors import QueryError
from hybridsearch.query import Bool, Node, PhraseNode, TermNode
from hybridsearch.schema import Schema
from hybridsearch.snapshot import SegmentView, Snapshot

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]
Scored = tuple[IntArray, FloatArray]
_EMPTY: Scored = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64))


@dataclass(frozen=True)
class Bm25Params:
    k1: float = 1.2
    b: float = 0.75


def _union(a: Scored, b: Scored) -> Scored:
    if len(a[0]) == 0:
        return b
    if len(b[0]) == 0:
        return a
    docs, inverse = np.unique(np.concatenate((a[0], b[0])), return_inverse=True)
    scores = np.bincount(inverse, weights=np.concatenate((a[1], b[1])), minlength=len(docs))
    return docs, scores


def _intersect(a: Scored, b: Scored) -> Scored:
    docs, ia, ib = np.intersect1d(a[0], b[0], assume_unique=True, return_indices=True)
    return docs, a[1][ia] + b[1][ib]


def _difference(a: Scored, excluded: IntArray) -> Scored:
    keep = ~np.isin(a[0], excluded, assume_unique=True)
    return a[0][keep], a[1][keep]


class LexicalSearcher:
    def __init__(self, snapshot: Snapshot, schema: Schema, params: Bm25Params | None = None):
        self.snapshot = snapshot
        self.schema = schema
        self.params = params or Bm25Params()
        self._n = snapshot.n_docs
        self._avg_len = {
            f: max(snapshot.total_len(f) / self._n, 1e-9) if self._n else 1.0
            for f in schema.text_fields
        }
        self._idf_cache: dict[tuple[str, str], float] = {}

    def _fields(self, field: str | None) -> list[str]:
        if field is None:
            return list(self.schema.text_fields)
        if field not in self.schema.text_fields:
            raise QueryError(
                f"unknown field {field!r}; text fields: {list(self.schema.text_fields)}"
            )
        return [field]

    def idf(self, field: str, term: str) -> float:
        key = (field, term)
        cached = self._idf_cache.get(key)
        if cached is None:
            df = self.snapshot.doc_freq(field, term)
            cached = math.log(1.0 + (self._n - df + 0.5) / (df + 0.5))
            self._idf_cache[key] = cached
        return cached

    def _bm25(
        self, view: SegmentView, field: str, docs: IntArray, tf: FloatArray, idf: float
    ) -> FloatArray:
        k1, b = self.params.k1, self.params.b
        dl = view.segment.field_lengths(field)[docs].astype(np.float64)
        norm = k1 * (1.0 - b + b * dl / self._avg_len[field])
        return self.schema.text_fields[field] * idf * tf * (k1 + 1.0) / (tf + norm)

    # ---------------------------------------------------------------- evaluation

    def evaluate(self, view: SegmentView, node: Node) -> Scored:
        """Matching live local doc ids and BM25 scores for ``node`` in one segment."""
        docs, scores = self._eval(view, node)
        live = view.live[docs]
        return docs[live], scores[live]

    def _eval(self, view: SegmentView, node: Node) -> Scored:
        if isinstance(node, TermNode):
            return self._term(view, node)
        if isinstance(node, PhraseNode):
            return self._phrase(view, node)
        if isinstance(node, Bool):
            return self._bool(view, node)
        raise TypeError(f"query node {node!r} was not analysed")

    def _term(self, view: SegmentView, node: TermNode) -> Scored:
        result = _EMPTY
        for field in self._fields(node.field):
            plist = view.segment.postings(field, node.term)
            if plist is None:
                continue
            scores = self._bm25(
                view,
                field,
                plist.doc_ids,
                plist.freqs.astype(np.float64),
                self.idf(field, node.term),
            )
            result = _union(result, (plist.doc_ids, scores))
        return result

    def _phrase(self, view: SegmentView, node: PhraseNode) -> Scored:
        result = _EMPTY
        for field in self._fields(node.field):
            lists = [view.segment.postings(field, t, with_positions=True) for t in node.terms]
            if any(p is None for p in lists):
                continue
            present = [p for p in lists if p is not None]
            common = present[0].doc_ids
            for p in present[1:]:
                common = np.intersect1d(common, p.doc_ids, assume_unique=True)
            if len(common) == 0:
                continue
            counts = np.zeros(len(common), dtype=np.float64)
            lookups = [
                (dict(zip(p.doc_ids.tolist(), p.positions or [], strict=True)), off)
                for p, off in zip(present, node.offsets, strict=True)
            ]
            for row, doc in enumerate(common.tolist()):
                first, base = lookups[0][0][doc], lookups[0][1]
                starts = {p - base for p in first}
                for table, off in lookups[1:]:
                    starts &= {p - off for p in table[doc]}
                    if not starts:
                        break
                counts[row] = len(starts)
            hit = counts > 0
            if not hit.any():
                continue
            idf = sum(self.idf(field, t) for t in node.terms)
            scores = self._bm25(view, field, common[hit], counts[hit], idf)
            result = _union(result, (common[hit], scores))
        return result

    def _bool(self, view: SegmentView, node: Bool) -> Scored:
        if node.must:
            result = self._eval(view, node.must[0])
            for child in node.must[1:]:
                result = _intersect(result, self._eval(view, child))
                if len(result[0]) == 0:
                    return result
            for child in node.should:
                extra = self._eval(view, child)
                common, in_result, in_extra = np.intersect1d(
                    result[0], extra[0], assume_unique=True, return_indices=True
                )
                if len(common):
                    boosted = result[1].copy()
                    boosted[in_result] += extra[1][in_extra]
                    result = (result[0], boosted)
        else:
            result = _EMPTY
            for child in node.should:
                result = _union(result, self._eval(view, child))
        for child in node.must_not:
            result = _difference(result, self._eval(view, child)[0])
        return result
