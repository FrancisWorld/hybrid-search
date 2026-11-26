"""The manifest is the single commit point of the index: which segment files are live."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hybridsearch.errors import CorruptIndexError
from hybridsearch.segment import fsync_dir

FORMAT_VERSION = 1


@dataclass(frozen=True)
class SegmentEntry:
    file: str
    deleted: tuple[int, ...] = ()


@dataclass(frozen=True)
class Manifest:
    """``checkpoint_seq`` is the last WAL sequence whose effects are inside ``segments``."""

    checkpoint_seq: int = 0
    next_segment_id: int = 1
    segments: tuple[SegmentEntry, ...] = field(default_factory=tuple)

    def to_json(self) -> dict[str, Any]:
        return {
            "format": FORMAT_VERSION,
            "checkpoint_seq": self.checkpoint_seq,
            "next_segment_id": self.next_segment_id,
            "segments": [{"file": s.file, "deleted": list(s.deleted)} for s in self.segments],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Manifest:
        try:
            if data["format"] != FORMAT_VERSION:
                raise CorruptIndexError(f"unsupported manifest format {data['format']!r}")
            return cls(
                int(data["checkpoint_seq"]),
                int(data["next_segment_id"]),
                tuple(
                    SegmentEntry(str(s["file"]), tuple(int(i) for i in s["deleted"]))
                    for s in data["segments"]
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CorruptIndexError(f"malformed manifest: {exc}") from exc


def read_manifest(path: Path) -> Manifest:
    if not path.exists():
        return Manifest()
    try:
        return Manifest.from_json(json.loads(path.read_text()))
    except json.JSONDecodeError as exc:
        raise CorruptIndexError(f"manifest is not valid JSON: {exc}") from exc


def write_manifest(path: Path, manifest: Manifest) -> None:
    """Atomically replace the manifest (temp file, fsync, rename, fsync directory)."""
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as fh:
        json.dump(manifest.to_json(), fh)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)
    fsync_dir(path.parent)
