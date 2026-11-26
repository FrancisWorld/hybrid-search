"""Relevance evaluation: nDCG@k, MRR and recall@k over a graded-judgement dataset."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from hybridsearch.index import Index
from hybridsearch.schema import Schema
from hybridsearch.search import Mode, SearchRequest

DATASETS = ("en", "pt")


@dataclass(frozen=True)
class JudgedQuery:
    id: str
    query: str
    relevance: Mapping[str, int]


@dataclass(frozen=True)
class Dataset:
    name: str
    schema: Schema
    documents: list[dict[str, Any]]
    queries: list[JudgedQuery]


@dataclass(frozen=True)
class Metrics:
    ndcg: float
    mrr: float
    recall: float
    queries: int


def dcg(gains: Sequence[int]) -> float:
    """Discounted cumulative gain with exponential gain ``2^rel - 1``."""
    return float(sum((2**g - 1) / math.log2(rank + 1) for rank, g in enumerate(gains, start=1)))


def ndcg_at_k(ranked_ids: Sequence[str], relevance: Mapping[str, int], k: int) -> float:
    ideal = dcg(sorted(relevance.values(), reverse=True)[:k])
    if ideal == 0:
        return 0.0
    return dcg([relevance.get(doc_id, 0) for doc_id in ranked_ids[:k]]) / ideal


def reciprocal_rank(ranked_ids: Sequence[str], relevance: Mapping[str, int]) -> float:
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if relevance.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked_ids: Sequence[str], relevance: Mapping[str, int], k: int) -> float:
    relevant = {d for d, g in relevance.items() if g > 0}
    if not relevant:
        return 0.0
    return len(relevant & set(ranked_ids[:k])) / len(relevant)


def load_dataset(name: str) -> Dataset:
    """Load a bundled dataset (``en`` or ``pt``)."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {list(DATASETS)}")
    root = resources.files("hybridsearch").joinpath("datasets", name)
    docs = [
        json.loads(line) for line in root.joinpath("docs.jsonl").read_text("utf-8").splitlines()
    ]
    queries = [
        JudgedQuery(q["id"], q["query"], q["relevance"])
        for q in map(json.loads, root.joinpath("queries.jsonl").read_text("utf-8").splitlines())
    ]
    schema = Schema(
        language=name,
        text_fields={"title": 2.0, "body": 1.0},
        keyword_fields=("category",),
        number_fields=("year",),
    )
    return Dataset(name, schema, docs, queries)


def evaluate(index: Index, queries: Sequence[JudgedQuery], mode: Mode, k: int = 10) -> Metrics:
    ndcgs, rrs, recalls = [], [], []
    for q in queries:
        result = index.search(
            SearchRequest(q.query, mode=mode, k=k, candidates=max(k, 50), highlight=False)
        )
        ranked = [hit.id for hit in result.hits]
        ndcgs.append(ndcg_at_k(ranked, q.relevance, k))
        rrs.append(reciprocal_rank(ranked, q.relevance))
        recalls.append(recall_at_k(ranked, q.relevance, k))
    n = len(queries)
    return Metrics(sum(ndcgs) / n, sum(rrs) / n, sum(recalls) / n, n)


def compare_modes(index: Index, queries: Sequence[JudgedQuery], k: int = 10) -> dict[str, Metrics]:
    modes: tuple[Mode, ...] = ("lexical", "vector", "hybrid")
    return {mode: evaluate(index, queries, mode, k) for mode in modes}


def load_into(index_dir: Path, dataset: Dataset) -> Index:
    """Create an index for ``dataset`` in ``index_dir`` and load its documents."""
    index = Index.open(index_dir, dataset.schema)
    index.put_many(dataset.documents)
    return index


def format_table(results: Mapping[str, Metrics], k: int) -> str:
    lines = [f"{'mode':<8} {'nDCG@' + str(k):>9} {'MRR':>7} {'recall@' + str(k):>10}"]
    for mode, m in results.items():
        lines.append(f"{mode:<8} {m.ndcg:>9.3f} {m.mrr:>7.3f} {m.recall:>10.3f}")
    return "\n".join(lines)
