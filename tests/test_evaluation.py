from __future__ import annotations

import math
from pathlib import Path

import pytest

from hybridsearch.evaluation import (
    DATASETS,
    compare_modes,
    dcg,
    format_table,
    load_dataset,
    load_into,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_ndcg_against_hand_computed_values() -> None:
    rel = {"a": 2, "b": 1}
    # ideal: gains [2, 1] -> 3/log2(2) + 1/log2(3)
    ideal = 3 / 1 + 1 / math.log2(3)
    assert dcg([2, 1]) == pytest.approx(ideal)
    assert ndcg_at_k(["a", "b"], rel, 10) == pytest.approx(1.0)
    swapped = (1 / 1 + 3 / math.log2(3)) / ideal
    assert ndcg_at_k(["b", "a"], rel, 10) == pytest.approx(swapped)
    assert ndcg_at_k(["x", "y"], rel, 10) == 0.0
    assert ndcg_at_k(["a"], {}, 10) == 0.0
    assert ndcg_at_k(["x", "a"], {"a": 2}, 1) == 0.0  # cut-off applies


def test_mrr_and_recall() -> None:
    rel = {"a": 1, "b": 2, "c": 0}
    assert reciprocal_rank(["x", "c", "b"], rel) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x"], rel) == 0.0
    assert recall_at_k(["a", "x", "b"], rel, 2) == pytest.approx(0.5)
    assert recall_at_k(["a", "b"], rel, 5) == 1.0
    assert recall_at_k(["a"], {"c": 0}, 5) == 0.0


@pytest.mark.parametrize("name", DATASETS)
def test_bundled_datasets_are_consistent(name: str) -> None:
    ds = load_dataset(name)
    ids = {d["id"] for d in ds.documents}
    assert len(ids) == len(ds.documents) >= 25
    assert len(ds.queries) >= 12
    for q in ds.queries:
        assert q.relevance
        assert set(q.relevance) <= ids
        assert max(q.relevance.values()) == 2


def test_unknown_dataset() -> None:
    with pytest.raises(ValueError, match="unknown dataset"):
        load_dataset("xx")


@pytest.mark.parametrize("name", DATASETS)
def test_hybrid_quality_floor_on_bundled_datasets(tmp_path: Path, name: str) -> None:
    """Regression guard: these floors sit below the measured values documented in the README."""
    ds = load_dataset(name)
    with load_into(tmp_path / "ix", ds) as index:
        results = compare_modes(index, ds.queries, k=10)
    print("\n" + format_table(results, 10))
    assert results["hybrid"].ndcg >= 0.80
    assert results["hybrid"].mrr >= 0.80
    assert results["lexical"].ndcg >= 0.70
    # fusing must not be worse than the weaker of its two inputs
    assert results["hybrid"].ndcg >= min(results["lexical"].ndcg, results["vector"].ndcg)
    assert "hybrid" in format_table(results, 10)
