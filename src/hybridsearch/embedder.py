"""Deterministic offline text embedder.

``HashingEmbedder`` is a feature-hashing model (Weinberger et al., 2009): each
text becomes a signed bag of hashed features projected into ``dim`` buckets and
L2-normalised. Features are stemmed words, adjacent word pairs and character
trigrams, so vectors capture morphology, word order and tolerance to typos.
It does **not** know that "car" and "automobile" are related; swap in a learned
model by implementing :class:`Embedder`.
"""

from __future__ import annotations

import itertools
import math
import re
import zlib
from collections import Counter
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from hybridsearch.analysis import Analyzer, fold_accents

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


@runtime_checkable
class Embedder(Protocol):
    """Maps texts to unit-length float32 vectors (zero vector for empty text)."""

    @property
    def name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]: ...


class HashingEmbedder:
    def __init__(
        self,
        analyzer: Analyzer,
        dim: int = 256,
        *,
        ngram_weight: float = 0.4,
        bigram_weight: float = 0.6,
    ) -> None:
        if dim < 8:
            raise ValueError("dim must be at least 8")
        self.analyzer = analyzer
        self.dim = dim
        self.name = f"hashing-{analyzer.name}"
        self._ngram_weight = ngram_weight
        self._bigram_weight = bigram_weight

    def _features(self, text: str) -> Counter[tuple[str, float]]:
        features: Counter[tuple[str, float]] = Counter()
        stems = [t.term for t in self.analyzer.analyze(text)]
        for stem in stems:
            features[(f"w:{stem}", 1.0)] += 1
        for left, right in itertools.pairwise(stems):
            features[(f"b:{left} {right}", self._bigram_weight)] += 1
        for word in _WORD_RE.findall(fold_accents(text)):
            padded = f"^{word}$"
            for i in range(len(padded) - 2):
                features[(f"c:{padded[i : i + 3]}", self._ngram_weight)] += 1
        return features

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for (feature, weight), count in self._features(text).items():
                raw = feature.encode("utf-8")
                bucket = zlib.crc32(raw) % self.dim
                sign = 1.0 if zlib.crc32(raw, 0x9E3779B9) & 1 else -1.0
                out[row, bucket] += sign * weight * (1.0 + math.log(count))
            norm = float(np.linalg.norm(out[row]))
            if norm > 0:
                out[row] /= norm
        return out
