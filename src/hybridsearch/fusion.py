"""Reciprocal rank fusion (Cormack, Clarke & Buettcher, 2009)."""

from __future__ import annotations

from collections.abc import Hashable, Sequence


def reciprocal_rank_fusion[K: Hashable](
    rankings: Sequence[Sequence[K]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[K, float]]:
    """Fuse ranked lists: ``score(d) = sum_i w_i / (k + rank_i(d))`` with 1-based ranks.

    Ties are broken by the best (lowest) rank any list gave the document, then
    by first appearance, which keeps the result deterministic.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    if weights is not None and len(weights) != len(rankings):
        raise ValueError("weights must match the number of rankings")
    scores: dict[K, float] = {}
    best_rank: dict[K, int] = {}
    order: dict[K, int] = {}
    for list_no, ranking in enumerate(rankings):
        weight = 1.0 if weights is None else weights[list_no]
        seen: set[K] = set()
        for rank, key in enumerate(ranking, start=1):
            if key in seen:
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
            best_rank[key] = min(best_rank.get(key, rank), rank)
            order.setdefault(key, len(order))
    return sorted(scores.items(), key=lambda kv: (-kv[1], best_rank[kv[0]], order[kv[0]]))
