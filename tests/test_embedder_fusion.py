from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray

from hybridsearch.analysis import get_analyzer
from hybridsearch.embedder import Embedder, HashingEmbedder
from hybridsearch.fusion import reciprocal_rank_fusion


def cos(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    return float(a @ b)


def test_embedder_is_deterministic_and_unit_norm() -> None:
    emb = HashingEmbedder(get_analyzer("en"), dim=128)
    a = emb.embed(["Rust makes memory safety practical"])
    b = HashingEmbedder(get_analyzer("en"), dim=128).embed(["Rust makes memory safety practical"])
    assert np.array_equal(a, b)
    assert a.shape == (1, 128)
    assert a.dtype == np.float32
    assert abs(float(np.linalg.norm(a[0])) - 1.0) < 1e-5
    assert isinstance(emb, Embedder)


def test_empty_text_gives_zero_vector() -> None:
    emb = HashingEmbedder(get_analyzer("en"))
    out = emb.embed(["", "  ,. -- "])
    assert not out.any()


def test_similarity_orders_related_texts() -> None:
    emb = HashingEmbedder(get_analyzer("en"))
    q, close, typo, far = emb.embed(
        [
            "espresso machine pressure",
            "How espresso machines build pressure",
            "espreso machin presure",
            "volcanic eruption lava flows",
        ]
    )
    assert cos(q, close) > cos(q, far) + 0.3
    assert cos(q, typo) > cos(q, far) + 0.2


def test_portuguese_morphology_and_accents() -> None:
    emb = HashingEmbedder(get_analyzer("pt"))
    a, b, c = emb.embed(["corrida de rua", "correr na rua", "vulcão ativo"])
    assert cos(a, b) > cos(a, c)
    plain, accent = emb.embed(["informacao", "informação"])
    assert cos(plain, accent) > 0.99


def test_dim_validation() -> None:
    with pytest.raises(ValueError, match="dim"):
        HashingEmbedder(get_analyzer("en"), dim=4)


def test_rrf_matches_hand_computed_scores() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]], k=60)
    scores = dict(fused)
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 62)
    assert scores["d"] == pytest.approx(1 / 63)
    assert [k for k, _ in fused] == ["b", "c", "a", "d"]


def test_rrf_weights_ties_and_validation() -> None:
    fused = reciprocal_rank_fusion([["a"], ["b"]], k=0)
    assert [k for k, _ in fused] == ["a", "b"]  # tie -> first appearance
    weighted = reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0, 3.0])
    assert weighted[0][0] == "b"
    with pytest.raises(ValueError, match="weights"):
        reciprocal_rank_fusion([["a"]], weights=[1.0, 2.0])
    with pytest.raises(ValueError, match="non-negative"):
        reciprocal_rank_fusion([["a"]], k=-1)
    assert reciprocal_rank_fusion([]) == []


def test_rrf_ignores_duplicate_entries_within_a_list() -> None:
    fused = dict(reciprocal_rank_fusion([["a", "a", "b"]], k=1))
    assert fused["a"] == pytest.approx(1 / 2)
    assert fused["b"] == pytest.approx(1 / 4)


@given(st.lists(st.lists(st.integers(0, 20), unique=True, max_size=15), max_size=4))
def test_rrf_property_union_and_monotonic(rankings: list[list[int]]) -> None:
    fused = reciprocal_rank_fusion(rankings)
    assert {k for k, _ in fused} == {x for r in rankings for x in r}
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)
