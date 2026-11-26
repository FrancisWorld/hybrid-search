"""The index: memtable + WAL + immutable segments + manifest, with crash recovery.

Write path
    1. validate and embed the documents (no locks held)
    2. append one WAL batch and fsync (the durability point)
    3. apply to memory: the memtable, and tombstones on older segments
    4. when the memtable is large: *flush* it into a new segment, commit a new
       manifest (checkpoint), reset the WAL, then run the merge policy

Read path
    Searches take an immutable :class:`Snapshot` (segments, tombstone masks and a
    searchable build of the memtable) and never block on writers, flushes or merges.

Recovery
    Open the manifest, load its segments, delete segment files it does not list,
    truncate any torn WAL tail and replay records newer than ``checkpoint_seq``.
    All operations are state-based (put replaces, delete removes), so replaying a
    record twice is harmless.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from hybridsearch.analysis import LanguageAnalyzer, get_analyzer
from hybridsearch.embedder import Embedder, HashingEmbedder
from hybridsearch.errors import (
    CorruptIndexError,
    IndexClosedError,
    IndexLockedError,
    ValidationError,
)
from hybridsearch.manifest import Manifest, SegmentEntry, read_manifest, write_manifest
from hybridsearch.merge import MergePolicy, merge_segments
from hybridsearch.schema import Document, Schema, parse_document
from hybridsearch.search import SearchRequest, SearchResult, run_search
from hybridsearch.searcher import Bm25Params
from hybridsearch.segment import (
    AnalyzedDoc,
    Segment,
    SegmentOptions,
    analyze_document,
    assemble_segment_data,
    serialize_segment,
    write_segment_file,
)
from hybridsearch.snapshot import SegmentView, Snapshot
from hybridsearch.wal import OP_DELETE, OP_PUT, RecoveryReport, WalRecord, WriteAheadLog

log = logging.getLogger("hybridsearch.index")

CONFIG_FILE = "index.json"
MANIFEST_FILE = "manifest.json"
WAL_FILE = "wal.log"
LOCK_FILE = "LOCK"
SEGMENT_DIR = "segments"
_OP_NAMES = {"put": OP_PUT, "delete": OP_DELETE}
_MEMTABLE_SEGMENT = SegmentOptions(hnsw_min_docs=1 << 60)  # memtable is scanned exactly


@dataclass(frozen=True)
class IndexOptions:
    fsync: bool = True
    memtable_max_docs: int = 1000
    auto_merge: bool = True
    merge_policy: MergePolicy = field(default_factory=MergePolicy)
    segment: SegmentOptions = field(default_factory=SegmentOptions)
    bm25: Bm25Params = field(default_factory=Bm25Params)


@dataclass
class _MemDoc:
    analyzed: AnalyzedDoc
    vector: Any  # float32 row


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Index:
    """A single-writer, many-reader hybrid search index stored in one directory."""

    def __init__(
        self,
        path: Path,
        schema: Schema,
        embedder: Embedder,
        options: IndexOptions,
        lock_fd: int,
    ) -> None:
        self.path = path
        self.schema = schema
        self.embedder = embedder
        self.options = options
        self.analyzer: LanguageAnalyzer = get_analyzer(schema.language)
        self._lock_fd: int | None = lock_fd
        self._write_lock = threading.RLock()
        self._state_lock = threading.Lock()
        self._wal = WriteAheadLog(path / WAL_FILE, fsync=options.fsync)
        self._views: list[SegmentView] = []
        self._files: list[str] = []
        self._mem: dict[str, _MemDoc] = {}
        self._mem_view: SegmentView | None = None
        self._pending: dict[int, set[int]] = {}  # tombstones of the batch being applied
        self._dirty = False  # tombstones or documents not yet covered by the manifest
        self._manifest = Manifest()
        self._seq = 0
        self._closed = False
        self.recovery = RecoveryReport(0, 0)

    # ------------------------------------------------------------------- opening

    @classmethod
    def open(
        cls,
        path: str | Path,
        schema: Schema | None = None,
        *,
        embedder: Embedder | None = None,
        options: IndexOptions | None = None,
    ) -> Index:
        """Open an index directory, creating it from ``schema`` if it does not exist."""
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        (root / SEGMENT_DIR).mkdir(exist_ok=True)
        lock_fd = cls._acquire_lock(root)
        try:
            schema = cls._load_or_create_config(root, schema)
            if embedder is None:
                embedder = HashingEmbedder(get_analyzer(schema.language), schema.embedding_dim)
            if embedder.dim != schema.embedding_dim:
                raise ValidationError(
                    f"embedder dim {embedder.dim} does not match schema embedding_dim "
                    f"{schema.embedding_dim}"
                )
            index = cls(root, schema, embedder, options or IndexOptions(), lock_fd)
            index._recover()
            return index
        except BaseException:
            os.close(lock_fd)
            raise

    @staticmethod
    def _acquire_lock(root: Path) -> int:
        fd = os.open(root / LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            raise IndexLockedError(f"{root} is already open by another writer") from exc
        return fd

    @staticmethod
    def _load_or_create_config(root: Path, schema: Schema | None) -> Schema:
        config = root / CONFIG_FILE
        if config.exists():
            try:
                stored = Schema.from_dict(json.loads(config.read_text())["schema"])
            except (ValueError, KeyError) as exc:
                raise CorruptIndexError(f"unreadable {CONFIG_FILE}: {exc}") from exc
            if schema is not None and schema != stored:
                raise ValidationError(f"schema differs from the one stored in {config}")
            return stored
        schema = schema or Schema()
        tmp = config.with_suffix(".tmp")
        tmp.write_text(json.dumps({"format": 1, "schema": schema.to_dict()}, indent=2))
        tmp.replace(config)
        _fsync_dir(root)
        return schema

    def _recover(self) -> None:
        seg_dir = self.path / SEGMENT_DIR
        self._manifest = read_manifest(self.path / MANIFEST_FILE)
        listed = {entry.file for entry in self._manifest.segments}
        for stray in seg_dir.iterdir():
            if stray.name not in listed:
                log.warning("removing segment file not in manifest: %s", stray.name)
                stray.unlink()
        for entry in self._manifest.segments:
            segment = Segment.open(seg_dir / entry.file)
            self._views.append(SegmentView.of(segment, entry.deleted))
            self._files.append(entry.file)
        self._seq = self._manifest.checkpoint_seq

        records, self.recovery = self._wal.recover()
        if self.recovery.was_torn:
            log.warning("truncated %d torn bytes from the WAL", self.recovery.dropped_bytes)
        pending = [r for r in records if r.seq > self._manifest.checkpoint_seq]
        self._replay(pending)
        if pending:
            self._seq = max(self._seq, pending[-1].seq)
            self._dirty = True
            log.info("replayed %d WAL records", len(pending))
        if len(self._mem) >= self.options.memtable_max_docs:
            self._checkpoint()

    def _replay(self, records: Sequence[WalRecord]) -> None:
        puts = [parse_document(r.data["doc"], self.schema) for r in records if r.op == "put"]
        vectors = iter(self.embedder.embed([d.text for d in puts])) if puts else iter(())
        docs = iter(puts)
        for record in records:
            if record.op == "put":
                self._apply_put(next(docs), next(vectors))
            else:
                self._apply_delete([str(record.data["id"])])
        self._flush_tombstones()

    # --------------------------------------------------------------------- writes

    def put(self, document: Mapping[str, Any]) -> None:
        self.put_many([document])

    def put_many(self, documents: Iterable[Mapping[str, Any]]) -> int:
        """Insert or replace documents (by ``id``). All-or-nothing: one bad document rejects all."""
        docs = [parse_document(d, self.schema) for d in documents]
        if not docs:
            return 0
        latest: dict[str, Document] = {}
        for doc in docs:  # last write wins inside one batch
            latest[doc.id] = doc
        docs = list(latest.values())
        vectors = self.embedder.embed([d.text for d in docs])
        with self._write_lock:
            self._ensure_open()
            self._log([("put", {"doc": d.to_dict()}) for d in docs])
            with self._state_lock:
                for doc, vector in zip(docs, vectors, strict=True):
                    self._apply_put(doc, vector)
                self._flush_tombstones()
            self._after_write()
        return len(docs)

    def delete(self, doc_id: str) -> bool:
        return self.delete_many([doc_id]) == 1

    def delete_many(self, ids: Iterable[str]) -> int:
        """Delete documents by id; returns how many existed."""
        unique = list(dict.fromkeys(ids))
        if not unique:
            return 0
        with self._write_lock:
            self._ensure_open()
            existing = [i for i in unique if self._exists(i)]
            if not existing:
                return 0
            self._log([("delete", {"id": i}) for i in existing])
            with self._state_lock:
                removed = self._apply_delete(existing)
                self._flush_tombstones()
            self._after_write()
        return removed

    def _log(self, ops: list[tuple[str, dict[str, Any]]]) -> None:
        records = [WalRecord(self._seq + i + 1, op, data) for i, (op, data) in enumerate(ops)]
        self._wal.append(records)
        self._seq = records[-1].seq
        self._dirty = True

    def _after_write(self) -> None:
        if len(self._mem) >= self.options.memtable_max_docs:
            self._checkpoint()
            if self.options.auto_merge:
                self._merge()

    # In-memory state changes. Callers hold the state lock (or run during recovery).

    def _locate(self, doc_id: str) -> tuple[int, int] | None:
        pending = self._pending
        for i, view in enumerate(self._views):
            local = view.segment.id_to_local.get(doc_id)
            if local is not None and view.live[local] and local not in pending.get(i, ()):
                return i, local
        return None

    def _exists(self, doc_id: str) -> bool:
        return doc_id in self._mem or self._locate(doc_id) is not None

    def _apply_put(self, doc: Document, vector: Any) -> None:
        location = self._locate(doc.id)
        if location is not None:
            self._pending.setdefault(location[0], set()).add(location[1])
        self._mem[doc.id] = _MemDoc(analyze_document(doc, self.schema, self.analyzer), vector)
        self._mem_view = None

    def _apply_delete(self, ids: Sequence[str]) -> int:
        removed = 0
        for doc_id in ids:
            if self._mem.pop(doc_id, None) is not None:
                self._mem_view = None
                removed += 1
            elif (location := self._locate(doc_id)) is not None:
                self._pending.setdefault(location[0], set()).add(location[1])
                removed += 1
        return removed

    def _flush_tombstones(self) -> None:
        pending = self._pending
        for i, locals_ in pending.items():
            self._views[i] = self._views[i].without(locals_)
        pending.clear()

    # ------------------------------------------------------------ flush and merge

    def flush(self) -> None:
        """Write the memtable to a new segment and checkpoint the WAL."""
        with self._write_lock:
            self._ensure_open()
            self._checkpoint()

    def merge(self, *, force: bool = False) -> int:
        """Run the merge policy (or merge everything with ``force``); returns merges done."""
        with self._write_lock:
            self._ensure_open()
            return self._merge(force=force)

    def _segment_path(self, name: str) -> Path:
        return self.path / SEGMENT_DIR / name

    def _entries(
        self, views: Sequence[SegmentView], files: Sequence[str]
    ) -> tuple[SegmentEntry, ...]:
        return tuple(SegmentEntry(f, tuple(v.deleted)) for v, f in zip(views, files, strict=True))

    def _checkpoint(self) -> None:
        if not self._mem and not self._dirty:
            return
        views, files = list(self._views), list(self._files)
        new_path: Path | None = None
        next_id = self._manifest.next_segment_id
        if self._mem:
            docs = list(self._mem.values())
            data = assemble_segment_data(
                [d.analyzed for d in docs], np.vstack([d.vector for d in docs]), self.schema
            )
            name = f"seg-{next_id:06d}.seg"
            new_path = self._segment_path(name)
            write_segment_file(new_path, serialize_segment(data, self.schema, self.options.segment))
            views.append(SegmentView.of(Segment.open(new_path)))
            files.append(name)
            next_id += 1
        dropped = [f for v, f in zip(views, files, strict=True) if v.live_count == 0]
        kept = [(v, f) for v, f in zip(views, files, strict=True) if v.live_count > 0]
        views, files = [v for v, _ in kept], [f for _, f in kept]
        manifest = Manifest(self._seq, next_id, self._entries(views, files))
        try:
            write_manifest(self.path / MANIFEST_FILE, manifest)  # commit point
        except BaseException:
            if new_path is not None:
                new_path.unlink(missing_ok=True)
            raise
        with self._state_lock:
            self._views, self._files, self._manifest = views, files, manifest
            self._mem.clear()
            self._mem_view = None
            self._dirty = False
        self._wal.reset()
        for name in dropped:
            self._segment_path(name).unlink(missing_ok=True)
        log.info("checkpoint at seq %d: %d segments", self._seq, len(files))

    def _merge(self, *, force: bool = False) -> int:
        merges = 0
        while True:
            live = [v.live_count for v in self._views]
            if force and len(live) > 1:
                chosen = list(range(len(live)))
            elif (
                force
                and len(live) == 1
                and self._views[0].live_count < self._views[0].segment.n_docs
            ):
                chosen = [0]
            else:
                chosen = [] if force else self.options.merge_policy.select(live)
            if not chosen:
                return merges
            self._merge_into_one(chosen)
            merges += 1

    def _merge_into_one(self, chosen: list[int]) -> None:
        parts = [(self._views[i].segment, self._views[i].live) for i in chosen]
        data = merge_segments(parts)
        next_id = self._manifest.next_segment_id
        views = [v for i, v in enumerate(self._views) if i not in chosen]
        files = [f for i, f in enumerate(self._files) if i not in chosen]
        new_path: Path | None = None
        if data.stored:
            name = f"seg-{next_id:06d}.seg"
            new_path = self._segment_path(name)
            write_segment_file(new_path, serialize_segment(data, self.schema, self.options.segment))
            views.append(SegmentView.of(Segment.open(new_path)))
            files.append(name)
            next_id += 1
        manifest = Manifest(self._manifest.checkpoint_seq, next_id, self._entries(views, files))
        try:
            write_manifest(self.path / MANIFEST_FILE, manifest)
        except BaseException:
            if new_path is not None:
                new_path.unlink(missing_ok=True)
            raise
        old_files = [self._files[i] for i in chosen]
        with self._state_lock:
            self._views, self._files, self._manifest = views, files, manifest
        for name in old_files:
            self._segment_path(name).unlink(missing_ok=True)
        log.info("merged %d segments into %s", len(chosen), files[-1] if data.stored else "nothing")

    # ---------------------------------------------------------------------- reads

    def snapshot(self) -> Snapshot:
        """Immutable view of everything committed so far, including the memtable."""
        with self._state_lock:
            self._ensure_open()
            if self._mem and self._mem_view is None:
                docs = list(self._mem.values())
                data = assemble_segment_data(
                    [d.analyzed for d in docs], np.vstack([d.vector for d in docs]), self.schema
                )
                blob = serialize_segment(data, self.schema, _MEMTABLE_SEGMENT)
                self._mem_view = SegmentView.of(Segment(blob))
            views = list(self._views)
            if self._mem and self._mem_view is not None:
                views.append(self._mem_view)
            return Snapshot(tuple(views))

    def search(self, request: SearchRequest) -> SearchResult:
        return run_search(
            self.snapshot(), self.schema, self.analyzer, self.embedder, request, self.options.bm25
        )

    def get(self, doc_id: str) -> dict[str, Any] | None:
        with self._state_lock:
            self._ensure_open()
            mem = self._mem.get(doc_id)
            if mem is not None:
                return mem.analyzed.stored
            for view in self._views:
                local = view.segment.id_to_local.get(doc_id)
                if local is not None and view.live[local]:
                    return view.segment.stored(local)
        return None

    def count(self) -> int:
        with self._state_lock:
            return sum(v.live_count for v in self._views) + len(self._mem)

    def stats(self) -> dict[str, Any]:
        with self._state_lock:
            self._ensure_open()
            wal_bytes = (self.path / WAL_FILE).stat().st_size
            return {
                "documents": sum(v.live_count for v in self._views) + len(self._mem),
                "memtable_documents": len(self._mem),
                "segments": [
                    {
                        "file": f,
                        "documents": v.segment.n_docs,
                        "live": v.live_count,
                        "hnsw": v.segment.hnsw is not None,
                    }
                    for v, f in zip(self._views, self._files, strict=True)
                ],
                "wal_bytes": wal_bytes,
                "last_seq": self._seq,
                "checkpoint_seq": self._manifest.checkpoint_seq,
                "language": self.schema.language,
                "embedder": self.embedder.name,
            }

    # ------------------------------------------------------------------ lifecycle

    def _ensure_open(self) -> None:
        if self._closed:
            raise IndexClosedError("index is closed")

    def close(self) -> None:
        """Flush, release files and the directory lock. Safe to call twice."""
        with self._write_lock:
            if self._closed:
                return
            try:
                self._checkpoint()
            finally:
                self._closed = True
                self._wal.close()
                for view in self._views:
                    view.segment.close()
                if self._lock_fd is not None:
                    os.close(self._lock_fd)
                    self._lock_fd = None

    def __enter__(self) -> Index:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
