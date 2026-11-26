"""Segment merging: tiered policy plus a postings-level merge.

The merge never re-analyses text. It decodes each source segment's postings,
drops tombstoned documents, renumbers the survivors and re-encodes, which is
why a merge costs far less than re-indexing and also reclaims deleted space.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from hybridsearch.hnsw import BoolMask
from hybridsearch.segment import FloatMatrix, Segment, SegmentData


@dataclass(frozen=True)
class MergePolicy:
    """Tiered merging: segments of similar size are merged once ``factor`` of them exist.

    A segment with ``n`` live documents belongs to tier ``floor(log_factor(n / floor_docs))``
    (tier 0 below ``floor_docs``). Merging ``factor`` same-tier segments yields a segment
    about one tier up, so every document is rewritten O(log N) times in total.
    """

    factor: int = 4
    floor_docs: int = 100
    max_merge: int = 8

    def __post_init__(self) -> None:
        if self.factor < 2 or self.max_merge < self.factor:
            raise ValueError("merge factor must be >= 2 and max_merge >= factor")

    def tier(self, live_docs: int) -> int:
        tier = 0
        size = max(live_docs, 1)
        threshold = self.floor_docs
        while size >= threshold * self.factor:
            tier += 1
            threshold *= self.factor
        return tier

    def select(self, live_docs: Sequence[int]) -> list[int]:
        """Indices (into ``live_docs``) to merge together, or ``[]`` if nothing is due.

        Picks the lowest tier that has at least ``factor`` segments, smallest first.
        """
        by_tier: dict[int, list[int]] = defaultdict(list)
        for index, count in enumerate(live_docs):
            by_tier[self.tier(count)].append(index)
        for tier in sorted(by_tier):
            members = sorted(by_tier[tier], key=lambda i: (live_docs[i], i))
            if len(members) >= self.factor:
                return sorted(members[: self.max_merge])
        return []


def merge_segments(parts: Sequence[tuple[Segment, BoolMask]]) -> SegmentData:
    """Merge ``(segment, live_mask)`` pairs, in order, into one :class:`SegmentData`."""
    if not parts:
        raise ValueError("nothing to merge")
    schema = parts[0][0].schema
    if any(seg.schema != schema for seg, _ in parts):
        raise ValueError("cannot merge segments with different schemas")

    stored: list[dict[str, Any]] = []
    lengths: dict[str, list[int]] = {name: [] for name in schema.text_fields}
    postings: dict[tuple[str, str], list[tuple[int, list[int]]]] = {}
    vectors: list[FloatMatrix] = []

    for seg, live in parts:
        live_ids = np.flatnonzero(live)
        if len(live_ids) == 0:
            continue
        base = len(stored)
        remap = np.full(seg.n_docs, -1, dtype=np.int64)
        remap[live_ids] = np.arange(base, base + len(live_ids))
        stored.extend(seg.stored(int(i)) for i in live_ids)
        for name in schema.text_fields:
            lengths[name].extend(seg.field_lengths(name)[live_ids].tolist())
        vectors.append(seg.vectors[live_ids])
        for key in seg.terms():
            decoded = seg.postings(*key, with_positions=True)
            assert decoded is not None
            assert decoded.positions is not None
            new_ids = remap[decoded.doc_ids]
            merged = postings.setdefault(key, [])
            merged.extend(
                (int(new), pos)
                for new, pos in zip(new_ids.tolist(), decoded.positions, strict=True)
                if new >= 0
            )
            if not merged:
                del postings[key]

    stacked = (
        np.vstack(vectors) if vectors else np.empty((0, schema.embedding_dim), dtype=np.float32)
    )
    return SegmentData(stored, lengths, postings, stacked)
