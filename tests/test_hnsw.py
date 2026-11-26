from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from hybridsearch.hnsw import HNSWIndex, brute_force


def unit_vectors(n: int, dim: int, seed: int) -> NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n, dim)).astype(np.float32)
    return (data / np.linalg.norm(data, axis=1, keepdims=True)).astype(np.float32)


def recall(
    index: HNSWIndex,
    queries: NDArray[np.float32],
    k: int,
    ef: int,
    allowed: NDArray[np.bool_] | None = None,
    exact_below: int = 256,
) -> float:
    hits = 0
    for q in queries:
        truth, _ = brute_force(index.vectors, q, k, allowed)
        got, _ = index.search(q, k, ef=ef, allowed=allowed, exact_below=exact_below)
        hits += len(set(truth.tolist()) & set(got.tolist()))
    return hits / (k * len(queries))


@pytest.fixture(scope="module")
def built() -> HNSWIndex:
    return HNSWIndex.build(unit_vectors(1200, 32, seed=1), m=12, ef_construction=80, seed=7)


@pytest.mark.slow
def test_recall_against_brute_force(built: HNSWIndex) -> None:
    queries = unit_vectors(60, 32, seed=2)
    value = recall(built, queries, k=10, ef=64)
    print(f"\nHNSW recall@10 (n=1200, dim=32, M=12, ef=64): {value:.3f}")
    assert value >= 0.9


@pytest.mark.slow
def test_filtered_search_only_returns_allowed_and_keeps_recall(built: HNSWIndex) -> None:
    rng = np.random.default_rng(3)
    allowed = rng.random(len(built)) < 0.4
    queries = unit_vectors(40, 32, seed=4)
    for q in queries:
        ids, _ = built.search(q, 10, ef=64, allowed=allowed, exact_below=10)
        assert allowed[ids].all()
    assert recall(built, queries, 10, 64, allowed, exact_below=10) >= 0.85


def test_selective_filter_falls_back_to_exact(built: HNSWIndex) -> None:
    allowed = np.zeros(len(built), dtype=bool)
    allowed[[5, 17, 400, 999]] = True
    q = unit_vectors(1, 32, seed=5)[0]
    ids, dists = built.search(q, 3, allowed=allowed)
    truth, truth_d = brute_force(built.vectors, q, 3, allowed)
    assert ids.tolist() == truth.tolist()
    assert np.allclose(dists, truth_d)


def test_results_are_sorted_and_exact_for_tiny_graphs() -> None:
    vectors = unit_vectors(30, 8, seed=6)
    index = HNSWIndex.build(vectors, m=4, ef_construction=30)
    q = unit_vectors(1, 8, seed=7)[0]
    ids, dists = index.search(q, 5, ef=30)
    truth, _ = brute_force(vectors, q, 5)
    assert ids.tolist() == truth.tolist()
    assert list(dists) == sorted(dists)


def test_empty_and_degenerate_inputs() -> None:
    empty = HNSWIndex.build(np.empty((0, 4), dtype=np.float32))
    q = np.full(4, 0.5, dtype=np.float32)
    assert empty.search(q, 3)[0].size == 0
    one = HNSWIndex.build(unit_vectors(1, 4, 0))
    assert one.search(q, 3)[0].tolist() == [0]
    assert one.search(q, 0)[0].size == 0
    none_allowed = np.zeros(1, dtype=bool)
    assert one.search(q, 3, allowed=none_allowed)[0].size == 0
    with pytest.raises(ValueError, match="at least 2"):
        HNSWIndex(unit_vectors(1, 4, 0), m=1)


def test_serialisation_roundtrip_preserves_results(built: HNSWIndex) -> None:
    restored = HNSWIndex.from_bytes(built.to_bytes(), built.vectors)
    for q in unit_vectors(10, 32, seed=8):
        a = built.search(q, 10, ef=48)
        b = restored.search(q, 10, ef=48)
        assert a[0].tolist() == b[0].tolist()


def test_from_bytes_rejects_mismatched_vectors(built: HNSWIndex) -> None:
    with pytest.raises(ValueError, match="does not match"):
        HNSWIndex.from_bytes(built.to_bytes(), built.vectors[:10])


def test_build_is_deterministic_for_a_seed() -> None:
    vectors = unit_vectors(200, 16, seed=9)
    a = HNSWIndex.build(vectors, m=8, ef_construction=40, seed=3).to_bytes()
    b = HNSWIndex.build(vectors, m=8, ef_construction=40, seed=3).to_bytes()
    assert a == b
