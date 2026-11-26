"""Hierarchical Navigable Small World graph (Malkov & Yashunin, 2016) in NumPy.

Vectors are expected to be L2-normalised ``float32`` rows, so cosine distance
is ``1 - dot``. The graph stores only links; the vector matrix is owned by the
caller (the segment file keeps it as one contiguous section).

Search accepts an ``allowed`` boolean mask (live documents AND filter matches).
The graph is traversed through every node, but only allowed nodes enter the
result set. When the mask is so selective that graph traversal would struggle,
the caller-visible behaviour is unchanged: we fall back to an exact scan over
the allowed rows, which is both faster and exact in that regime.
"""

from __future__ import annotations

import heapq
import io
import math
import random
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatMatrix = NDArray[np.float32]
BoolMask = NDArray[np.bool_]


def brute_force(
    vectors: FloatMatrix, query: FloatMatrix, k: int, allowed: BoolMask | None = None
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    """Exact top-``k`` by cosine distance over ``allowed`` rows."""
    if len(vectors) == 0 or k <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    candidates = np.arange(len(vectors)) if allowed is None else np.flatnonzero(allowed)
    if len(candidates) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    dists = 1.0 - vectors[candidates] @ query
    k = min(k, len(candidates))
    top = np.argpartition(dists, k - 1)[:k] if k < len(candidates) else np.arange(len(candidates))
    order = top[np.argsort(dists[top], kind="stable")]
    return candidates[order].astype(np.int64), dists[order].astype(np.float32)


class HNSWIndex:
    def __init__(
        self,
        vectors: FloatMatrix,
        *,
        m: int = 16,
        ef_construction: int = 100,
        seed: int = 0,
    ) -> None:
        if m < 2:
            raise ValueError("m must be at least 2")
        self.vectors = vectors
        self.m = m
        self.m0 = 2 * m
        self.ef_construction = ef_construction
        self._level_mult = 1.0 / math.log(m)
        self._rng = random.Random(seed)
        self._links: list[list[list[int]]] = []  # node -> layer -> neighbours
        self._entry = -1
        self._max_level = -1

    # ------------------------------------------------------------------ building

    @classmethod
    def build(
        cls, vectors: FloatMatrix, *, m: int = 16, ef_construction: int = 100, seed: int = 0
    ) -> HNSWIndex:
        index = cls(vectors, m=m, ef_construction=ef_construction, seed=seed)
        for node in range(len(vectors)):
            index._insert(node)
        return index

    def __len__(self) -> int:
        return len(self._links)

    def _random_level(self) -> int:
        return int(-math.log(1.0 - self._rng.random()) * self._level_mult)

    def _insert(self, node: int) -> None:
        level = self._random_level()
        self._links.append([[] for _ in range(level + 1)])
        if self._entry < 0:
            self._entry, self._max_level = node, level
            return
        query = self.vectors[node]
        entry = self._entry
        for layer in range(self._max_level, level, -1):
            entry = self._greedy(query, entry, layer)
        entries = [entry]
        for layer in range(min(level, self._max_level), -1, -1):
            found = self._search_layer(query, entries, self.ef_construction, layer, None)
            neighbours = self._select(found, self.m)
            self._links[node][layer] = neighbours
            cap = self.m0 if layer == 0 else self.m
            for other in neighbours:
                links = self._links[other][layer]
                links.append(node)
                if len(links) > cap:
                    self._links[other][layer] = self._shrink(other, links, cap)
            entries = [n for _, n in found]
        if level > self._max_level:
            self._entry, self._max_level = node, level

    def _select(self, candidates: list[tuple[float, int]], m: int) -> list[int]:
        """Neighbour-selection heuristic: prefer diverse directions, then fill up."""
        selected: list[int] = []
        discarded: list[int] = []
        for dist, cand in sorted(candidates):
            if len(selected) >= m:
                break
            if selected:
                to_selected = 1.0 - self.vectors[selected] @ self.vectors[cand]
                if bool(np.any(to_selected <= dist)):
                    discarded.append(cand)
                    continue
            selected.append(cand)
        for cand in discarded:
            if len(selected) >= m:
                break
            selected.append(cand)
        return selected

    def _shrink(self, node: int, links: list[int], cap: int) -> list[int]:
        dists = 1.0 - self.vectors[links] @ self.vectors[node]
        return self._select(list(zip(dists.tolist(), links, strict=True)), cap)

    # ----------------------------------------------------------------- searching

    def _greedy(self, query: FloatMatrix, entry: int, layer: int) -> int:
        best = entry
        best_dist = float(1.0 - self.vectors[entry] @ query)
        improved = True
        while improved:
            improved = False
            neighbours = self._links[best][layer] if layer < len(self._links[best]) else []
            if not neighbours:
                break
            dists = 1.0 - self.vectors[neighbours] @ query
            j = int(np.argmin(dists))
            if float(dists[j]) < best_dist:
                best, best_dist, improved = neighbours[j], float(dists[j]), True
        return best

    def _search_layer(
        self,
        query: FloatMatrix,
        entries: list[int],
        ef: int,
        layer: int,
        allowed: BoolMask | None,
    ) -> list[tuple[float, int]]:
        visited = set(entries)
        candidates: list[tuple[float, int]] = []
        results: list[tuple[float, int]] = []  # max-heap via negated distance
        entry_dists = 1.0 - self.vectors[entries] @ query
        for dist, node in zip(entry_dists.tolist(), entries, strict=True):
            heapq.heappush(candidates, (dist, node))
            if allowed is None or allowed[node]:
                heapq.heappush(results, (-dist, node))
        while candidates:
            dist, node = heapq.heappop(candidates)
            if len(results) >= ef and dist > -results[0][0]:
                break
            fresh = [n for n in self._links[node][layer] if n not in visited]
            if not fresh:
                continue
            visited.update(fresh)
            for d, n in zip((1.0 - self.vectors[fresh] @ query).tolist(), fresh, strict=True):
                if len(results) < ef or d < -results[0][0]:
                    heapq.heappush(candidates, (d, n))
                    if allowed is None or allowed[n]:
                        heapq.heappush(results, (-d, n))
                        if len(results) > ef:
                            heapq.heappop(results)
        return [(-neg, n) for neg, n in results]

    def search(
        self,
        query: FloatMatrix,
        k: int,
        *,
        ef: int = 64,
        allowed: BoolMask | None = None,
        exact_below: int = 256,
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Return ``(ids, distances)`` of the ``k`` nearest allowed nodes, nearest first."""
        if self._entry < 0 or k <= 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        if allowed is not None and int(allowed.sum()) <= max(exact_below, 2 * ef):
            return brute_force(self.vectors, query, k, allowed)
        entry = self._entry
        for layer in range(self._max_level, 0, -1):
            entry = self._greedy(query, entry, layer)
        found = sorted(self._search_layer(query, [entry], max(ef, k), 0, allowed))[:k]
        return (
            np.asarray([n for _, n in found], dtype=np.int64),
            np.asarray([d for d, _ in found], dtype=np.float32),
        )

    # ------------------------------------------------------------- serialisation

    def to_bytes(self) -> bytes:
        n = len(self._links)
        arrays: dict[str, Any] = {
            "params": np.asarray(
                [self.m, self.ef_construction, self._entry, self._max_level], dtype=np.int64
            ),
            "levels": np.asarray([len(link) - 1 for link in self._links], dtype=np.int32),
        }
        layer0 = np.full((n, self.m0), -1, dtype=np.int32)
        for node, links in enumerate(self._links):
            layer0[node, : len(links[0])] = links[0]
        arrays["layer0"] = layer0
        for layer in range(1, self._max_level + 1):
            nodes = [i for i, links in enumerate(self._links) if len(links) > layer]
            matrix = np.full((len(nodes), self.m), -1, dtype=np.int32)
            for row, node in enumerate(nodes):
                matrix[row, : len(self._links[node][layer])] = self._links[node][layer]
            arrays[f"nodes{layer}"] = np.asarray(nodes, dtype=np.int32)
            arrays[f"links{layer}"] = matrix
        buf = io.BytesIO()
        np.savez(buf, **arrays)
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, blob: bytes, vectors: FloatMatrix) -> HNSWIndex:
        with np.load(io.BytesIO(blob), allow_pickle=False) as data:
            m, efc, entry, max_level = (int(x) for x in data["params"])
            levels = data["levels"]
            if len(levels) != len(vectors):
                raise ValueError("HNSW graph does not match the vector matrix")
            index = cls(vectors, m=m, ef_construction=efc)
            index._entry, index._max_level = entry, max_level
            index._links = [[[] for _ in range(int(lv) + 1)] for lv in levels]
            for node, row in enumerate(data["layer0"].tolist()):
                index._links[node][0] = [x for x in row if x >= 0]
            for layer in range(1, max_level + 1):
                nodes = data[f"nodes{layer}"].tolist()
                for node, row in zip(nodes, data[f"links{layer}"].tolist(), strict=True):
                    index._links[node][layer] = [x for x in row if x >= 0]
        return index
