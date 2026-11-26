"""Immutable point-in-time view of the index used by every search."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from hybridsearch.hnsw import BoolMask
from hybridsearch.segment import Segment


@dataclass(frozen=True)
class SegmentView:
    """A segment plus its tombstones. Both are immutable; deletes create a new view."""

    segment: Segment
    live: BoolMask
    live_count: int
    live_len: dict[str, int]

    @classmethod
    def of(cls, segment: Segment, deleted: Iterable[int] = ()) -> SegmentView:
        live = np.ones(segment.n_docs, dtype=bool)
        deleted = list(deleted)
        live[deleted] = False
        live.setflags(write=False)
        return cls(
            segment,
            live,
            int(live.sum()),
            {f: int(segment.field_lengths(f)[live].sum()) for f in segment.schema.text_fields},
        )

    @property
    def deleted(self) -> list[int]:
        return np.flatnonzero(~self.live).tolist()

    def without(self, locals_: Iterable[int]) -> SegmentView:
        return SegmentView.of(self.segment, [*self.deleted, *locals_])


@dataclass(frozen=True)
class Snapshot:
    views: tuple[SegmentView, ...]

    @property
    def n_docs(self) -> int:
        return sum(v.live_count for v in self.views)

    def total_len(self, field: str) -> int:
        return sum(v.live_len[field] for v in self.views)

    def doc_freq(self, field: str, term: str) -> int:
        """Document frequency summed over segments (tombstoned docs still count until merged)."""
        return sum(v.segment.doc_freq(field, term) for v in self.views)
