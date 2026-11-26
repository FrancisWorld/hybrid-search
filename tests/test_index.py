"""Index lifecycle: durability, crash recovery, merging and concurrency on a real directory."""

from __future__ import annotations

import os
import shutil
import stat
import threading
from pathlib import Path
from typing import Any

import pytest

import hybridsearch.index as index_module
from hybridsearch.analysis import get_analyzer
from hybridsearch.embedder import HashingEmbedder
from hybridsearch.errors import (
    CorruptIndexError,
    IndexClosedError,
    IndexLockedError,
    ValidationError,
)
from hybridsearch.index import WAL_FILE, Index, IndexOptions
from hybridsearch.merge import MergePolicy
from hybridsearch.schema import Schema
from hybridsearch.search import Mode, SearchRequest
from hybridsearch.segment import SegmentOptions
from hybridsearch.wal import WriteAheadLog
from tests.helpers import VOCAB, dir_of

SMALL = IndexOptions(
    fsync=False,
    memtable_max_docs=20,
    merge_policy=MergePolicy(factor=3, floor_docs=10, max_merge=6),
    segment=SegmentOptions(hnsw_min_docs=15),
)


def doc(i: int, version: int = 1) -> dict[str, Any]:
    words = [VOCAB[(i * 7 + j * 3) % len(VOCAB)] for j in range(8)]
    return {
        "id": f"doc-{i}",
        "fields": {"title": f"title {i} v{version}", "body": " ".join(words) + f" marker{i}"},
        "meta": {"category": f"c{i % 3}", "year": 2000 + i % 25},
    }


def ids_of(index: Index, query: str, mode: Mode = "lexical", k: int = 100) -> list[str]:
    result = index.search(SearchRequest(query, mode=mode, k=k, candidates=max(k, 100)))
    return [h.id for h in result.hits]


@pytest.fixture
def schema() -> Schema:
    return Schema(
        language="en",
        text_fields={"title": 2.0, "body": 1.0},
        keyword_fields=("category",),
        number_fields=("year",),
        embedding_dim=64,
    )


def snapshot_dir(src: Path, dst: Path) -> Path:
    """What a crashed process would leave behind: the files, with no orderly shutdown."""
    shutil.copytree(src, dst)
    (dst / "LOCK").unlink(missing_ok=True)
    return dst


def test_put_search_get_delete(tmp_path: Path, schema: Schema) -> None:
    with Index.open(tmp_path / "ix", schema, options=SMALL) as ix:
        assert ix.put_many([doc(1), doc(2), doc(3)]) == 3
        assert ix.count() == 3
        assert ix.get("doc-2") is not None
        assert ix.get("nope") is None
        assert "doc-2" in ids_of(ix, "marker2")
        assert ix.delete("doc-2") is True
        assert ix.delete("doc-2") is False
        assert ix.count() == 2
        assert ids_of(ix, "marker2") == []
        assert ix.get("doc-2") is None


def test_upsert_replaces_in_memtable_and_in_segments(tmp_path: Path, schema: Schema) -> None:
    with Index.open(tmp_path / "ix", schema, options=SMALL) as ix:
        ix.put(doc(1))
        ix.put(doc(1, version=2))
        assert ix.count() == 1
        assert ix.get("doc-1")["fields"]["title"] == "title 1 v2"  # type: ignore[index]
        ix.flush()
        ix.put(doc(1, version=3))  # now the old copy lives in a segment
        assert ix.count() == 1
        assert ix.get("doc-1")["fields"]["title"] == "title 1 v3"  # type: ignore[index]
        assert ids_of(ix, "marker1") == ["doc-1"]
        assert ids_of(ix, "v1") == []
        assert ids_of(ix, "v3") == ["doc-1"]
        assert ix.put_many([doc(5, 1), doc(5, 2)]) == 1  # duplicates inside a batch collapse
        assert ix.get("doc-5")["fields"]["title"] == "title 5 v2"  # type: ignore[index]


def test_batch_is_all_or_nothing(tmp_path: Path, schema: Schema) -> None:
    with Index.open(tmp_path / "ix", schema, options=SMALL) as ix:
        with pytest.raises(ValidationError):
            ix.put_many([doc(1), {"id": "bad", "fields": {"nope": "x"}}])
        assert ix.count() == 0
        assert ix.stats()["last_seq"] == 0
        assert ix.put_many([]) == 0
        assert ix.delete_many([]) == 0


def test_flush_reopen_and_segments_persist(tmp_path: Path, schema: Schema) -> None:
    path = tmp_path / "ix"
    with Index.open(path, schema, options=SMALL) as ix:
        ix.put_many([doc(i) for i in range(50)])
        ix.delete("doc-3")
        before = ids_of(ix, "marker7 OR marker3 OR marker40")
    with Index.open(path, options=SMALL) as ix:
        assert ix.count() == 49
        assert ix.stats()["memtable_documents"] == 0
        assert ids_of(ix, "marker7 OR marker3 OR marker40") == before
        assert "doc-3" not in before


def test_second_writer_is_refused_and_close_releases(tmp_path: Path, schema: Schema) -> None:
    path = tmp_path / "ix"
    ix = Index.open(path, schema, options=SMALL)
    with pytest.raises(IndexLockedError):
        Index.open(path, options=SMALL)
    ix.close()
    ix.close()
    Index.open(path, options=SMALL).close()
    with pytest.raises(IndexClosedError):
        ix.search(SearchRequest("x"))
    with pytest.raises(IndexClosedError):
        ix.put(doc(1))


def test_schema_and_embedder_mismatch_rejected(tmp_path: Path, schema: Schema) -> None:
    path = tmp_path / "ix"
    Index.open(path, schema, options=SMALL).close()
    other = Schema(language="pt", embedding_dim=64)
    with pytest.raises(ValidationError, match="schema differs"):
        Index.open(path, other, options=SMALL)
    Index.open(path, options=SMALL).close()  # lock was released after the failed open
    with pytest.raises(ValidationError, match="embedder dim"):
        Index.open(path, embedder=HashingEmbedder(get_analyzer("en"), dim=32), options=SMALL)


def test_corrupt_config_and_manifest_are_reported(tmp_path: Path, schema: Schema) -> None:
    path = tmp_path / "ix"
    with Index.open(path, schema, options=SMALL) as ix:
        ix.put_many([doc(i) for i in range(5)])
    (path / "manifest.json").write_text("{not json")
    with pytest.raises(CorruptIndexError, match="manifest"):
        Index.open(path, options=SMALL)
    (path / "manifest.json").write_text('{"format": 1}')
    with pytest.raises(CorruptIndexError, match="malformed"):
        Index.open(path, options=SMALL)
    (path / "index.json").write_text("{}")
    with pytest.raises(CorruptIndexError, match=r"index\.json"):
        Index.open(path, options=SMALL)


def test_flipped_byte_in_segment_is_detected_on_open(tmp_path: Path, schema: Schema) -> None:
    path = tmp_path / "ix"
    with Index.open(path, schema, options=SMALL) as ix:
        ix.put_many([doc(i) for i in range(5)])
    seg = next((path / "segments").glob("*.seg"))
    good = seg.read_bytes()
    data = bytearray(good)
    data[len(data) // 3] ^= 0x55
    seg.write_bytes(bytes(data))
    with pytest.raises(CorruptIndexError):
        Index.open(path, options=SMALL)
    seg.write_bytes(good)  # also proves the failed open released the directory lock
    with Index.open(path, options=SMALL) as ix:
        assert ix.count() == 5


# ------------------------------------------------------------------ crash recovery


def test_crash_before_any_flush_recovers_everything_from_wal(
    tmp_path: Path, schema: Schema
) -> None:
    live = Index.open(tmp_path / "ix", schema, options=SMALL)
    live.put_many([doc(i) for i in range(12)])
    live.delete("doc-4")
    live.put(doc(5, version=2))
    expected = ids_of(live, "marker5 OR marker4 OR marker11")
    crashed = snapshot_dir(tmp_path / "ix", tmp_path / "crashed")

    with Index.open(crashed, options=SMALL) as ix:
        assert ix.recovery.records == 14
        assert ix.count() == 11
        assert ix.get("doc-4") is None
        assert ix.get("doc-5")["fields"]["title"] == "title 5 v2"  # type: ignore[index]
        assert ids_of(ix, "marker5 OR marker4 OR marker11") == expected
    live.close()


def test_torn_wal_tail_loses_only_the_unacknowledged_write(tmp_path: Path, schema: Schema) -> None:
    live = Index.open(tmp_path / "ix", schema, options=SMALL)
    for i in range(6):
        live.put(doc(i))
    crashed = snapshot_dir(tmp_path / "ix", tmp_path / "crashed")
    wal = crashed / WAL_FILE
    wal.write_bytes(wal.read_bytes()[:-9])  # cut inside the last record
    with Index.open(crashed, options=SMALL) as ix:
        assert ix.recovery.was_torn
        assert ix.count() == 5
        assert ix.get("doc-5") is None
        ix.put(doc(5))  # appends continue cleanly after truncation
    with Index.open(crashed, options=SMALL) as ix:
        assert ix.count() == 6
    live.close()


def test_crash_after_segment_write_before_manifest_commit(
    tmp_path: Path, schema: Schema, monkeypatch: pytest.MonkeyPatch
) -> None:
    ix = Index.open(tmp_path / "ix", schema, options=SMALL)
    ix.put_many([doc(i) for i in range(8)])

    def boom(*_: object) -> None:
        raise OSError("disk fell over")

    monkeypatch.setattr(index_module, "write_manifest", boom)
    with pytest.raises(OSError, match="fell over"):
        ix.flush()
    monkeypatch.undo()
    # The failed flush rolled back cleanly: nothing lost, no orphan segment, still serving.
    assert ix.count() == 8
    assert not list((tmp_path / "ix" / "segments").iterdir())
    assert ids_of(ix, "marker3") == ["doc-3"]
    # A crash at the same point (segment file present, manifest old) is also recoverable.
    crashed = snapshot_dir(tmp_path / "ix", tmp_path / "crashed")
    (crashed / "segments" / "seg-000001.seg").write_bytes(b"half written segment")
    with Index.open(crashed, options=SMALL) as recovered:
        assert recovered.count() == 8
        assert (
            not (crashed / "segments" / "seg-000001.seg").exists() or recovered.stats()["segments"]
        )
    ix.flush()
    assert len(ix.stats()["segments"]) == 1
    ix.close()


def test_crash_after_manifest_before_wal_reset_does_not_duplicate(
    tmp_path: Path, schema: Schema, monkeypatch: pytest.MonkeyPatch
) -> None:
    ix = Index.open(tmp_path / "ix", schema, options=SMALL)
    ix.put_many([doc(i) for i in range(8)])
    ix.delete("doc-2")

    def boom(self: WriteAheadLog) -> None:
        raise OSError("crashed before truncating the WAL")

    monkeypatch.setattr(WriteAheadLog, "reset", boom)
    with pytest.raises(OSError, match="crashed"):
        ix.flush()
    monkeypatch.undo()
    crashed = snapshot_dir(tmp_path / "ix", tmp_path / "crashed")
    with Index.open(crashed, options=SMALL) as recovered:
        assert recovered.stats()["checkpoint_seq"] == 9
        assert recovered.recovery.records == 9  # still in the log, but already checkpointed
        assert recovered.count() == 7
        assert ids_of(recovered, "marker3") == ["doc-3"]
        assert recovered.stats()["memtable_documents"] == 0
    ix.close()


def test_replay_is_idempotent_across_repeated_recoveries(tmp_path: Path, schema: Schema) -> None:
    live = Index.open(tmp_path / "ix", schema, options=SMALL)
    live.put_many([doc(i) for i in range(10)])
    live.delete_many(["doc-1", "doc-2"])
    live.put(doc(3, version=9))
    crashed = snapshot_dir(tmp_path / "ix", tmp_path / "crashed")
    for attempt in range(3):
        copy = snapshot_dir(crashed, tmp_path / f"attempt-{attempt}")
        for _ in range(2):  # recover, close (checkpoints), recover again from the checkpoint
            with Index.open(copy, options=SMALL) as ix:
                assert ix.count() == 8
                assert ix.get("doc-3")["fields"]["title"] == "title 3 v9"  # type: ignore[index]
                assert ix.get("doc-1") is None
    live.close()


def test_recovery_after_flush_replays_only_newer_records(tmp_path: Path, schema: Schema) -> None:
    live = Index.open(tmp_path / "ix", schema, options=SMALL)
    live.put_many([doc(i) for i in range(10)])
    live.flush()
    live.delete("doc-0")  # tombstone on a segment doc, lives only in WAL
    live.put(doc(50))
    crashed = snapshot_dir(tmp_path / "ix", tmp_path / "crashed")
    with Index.open(crashed, options=SMALL) as ix:
        assert ix.recovery.records == 2
        assert ix.count() == 10
        assert ix.get("doc-0") is None
        assert ids_of(ix, "marker50") == ["doc-50"]
    live.close()


# ---------------------------------------------------------------------- merging


def test_auto_flush_and_tiered_merge_keep_results_identical(tmp_path: Path, schema: Schema) -> None:
    with Index.open(tmp_path / "ix", schema, options=SMALL) as ix:
        ranking_before: dict[str, list[tuple[str, float]]] = {}
        for i in range(0, 120):
            ix.put(doc(i))
        stats = ix.stats()
        assert stats["checkpoint_seq"] > 0
        assert len(stats["segments"]) < 120 // 20  # merges happened
        queries = ["marker7", "espresso machine", "lava OR volcano", '"espresso machine"']
        for q in queries:
            res = ix.search(SearchRequest(q, mode="lexical", k=20, candidates=100))
            ranking_before[q] = [(h.id, round(h.score, 9)) for h in res.hits]
        ix.merge(force=True)
        assert len(ix.stats()["segments"]) <= 2
        for q in queries:
            res = ix.search(SearchRequest(q, mode="lexical", k=20, candidates=100))
            assert [(h.id, round(h.score, 9)) for h in res.hits] == ranking_before[q], q
        assert ix.count() == 120


def test_merge_reclaims_deleted_documents_and_files(tmp_path: Path, schema: Schema) -> None:
    path = tmp_path / "ix"
    with Index.open(path, schema, options=SMALL) as ix:
        ix.put_many([doc(i) for i in range(10)])
        ix.flush()
        ix.put_many([doc(i) for i in range(10, 20)])
        ix.flush()
        ix.delete_many([f"doc-{i}" for i in range(0, 20, 2)])
        assert sum(s["documents"] for s in ix.stats()["segments"]) == 20
        ix.merge(force=True)
        segs = ix.stats()["segments"]
        assert [s["documents"] for s in segs] == [10]
        assert [s["live"] for s in segs] == [10]
        assert len(list((path / "segments").iterdir())) == 1
        ix.delete_many([f"doc-{i}" for i in range(1, 20, 2)])
        ix.flush()
        assert ix.stats()["segments"] == []  # fully dead segments are dropped
        assert list((path / "segments").iterdir()) == []
        assert ix.count() == 0


def test_crash_after_merge_commit_still_consistent(tmp_path: Path, schema: Schema) -> None:
    live = Index.open(tmp_path / "ix", schema, options=SMALL)
    live.put_many([doc(i) for i in range(10)])
    live.flush()
    live.put_many([doc(i) for i in range(10, 20)])
    live.flush()
    live.delete("doc-3")  # WAL-only tombstone
    live.put(doc(4, version=2))
    live.merge(force=True)  # manifest now has the merged segment, WAL still holds both ops
    crashed = snapshot_dir(tmp_path / "ix", tmp_path / "crashed")
    with Index.open(crashed, options=SMALL) as ix:
        assert ix.count() == 19
        assert ix.get("doc-3") is None
        assert ix.get("doc-4")["fields"]["title"] == "title 4 v2"  # type: ignore[index]
    live.close()


# ------------------------------------------------------------------ concurrency


def test_concurrent_writers_readers_flush_and_merge(tmp_path: Path, schema: Schema) -> None:
    ix = Index.open(
        tmp_path / "ix",
        schema,
        options=IndexOptions(
            fsync=False,
            memtable_max_docs=15,
            merge_policy=MergePolicy(factor=3, floor_docs=8, max_merge=6),
            segment=SegmentOptions(hnsw_min_docs=10),
        ),
    )
    errors: list[BaseException] = []
    stop = threading.Event()
    n_writers, per_writer = 4, 40

    def writer(w: int) -> None:
        try:
            for i in range(per_writer):
                n = w * 1000 + i
                ix.put(doc(n))
                if i % 5 == 0:
                    ix.put(doc(n, version=2))  # upsert
                if i % 7 == 0:
                    ix.delete(f"doc-{n}")
        except BaseException as exc:
            errors.append(exc)

    def reader() -> None:
        try:
            while not stop.is_set():
                modes: tuple[Mode, ...] = ("lexical", "vector", "hybrid")
                for mode in modes:
                    res = ix.search(
                        SearchRequest("espresso machine lava", mode=mode, k=10, facets=["category"])
                    )
                    for hit in res.hits:
                        number = int(hit.id.split("-")[1])
                        assert hit.document["id"] == hit.id
                        assert hit.document["fields"]["title"].startswith(f"title {number} ")
        except BaseException as exc:
            errors.append(exc)

    def maintenance() -> None:
        try:
            while not stop.is_set():
                ix.flush()
                ix.merge()
                stop.wait(0.02)
        except BaseException as exc:
            errors.append(exc)

    writers = [threading.Thread(target=writer, args=(w,)) for w in range(n_writers)]
    others = [threading.Thread(target=reader) for _ in range(2)] + [
        threading.Thread(target=maintenance)
    ]
    for t in [*writers, *others]:
        t.start()
    for t in writers:
        t.join(timeout=120)
    stop.set()
    for t in others:
        t.join(timeout=60)
    assert not errors, errors[0]

    expected = {
        f"doc-{w * 1000 + i}" for w in range(n_writers) for i in range(per_writer) if i % 7 != 0
    }
    ix.flush()
    assert ix.count() == len(expected)
    found = {
        h.id for h in ix.search(SearchRequest("title", mode="lexical", k=100, candidates=1000)).hits
    }
    assert found <= expected
    all_ids = set()
    for s in ix.snapshot().views:
        all_ids |= {i for i, loc in s.segment.id_to_local.items() if s.live[loc]}
    assert all_ids == expected
    ix.close()
    with Index.open(tmp_path / "ix", options=SMALL) as again:
        assert again.count() == len(expected)


def test_every_durable_file_change_fsyncs_its_directory(
    tmp_path: Path, schema: Schema, monkeypatch: pytest.MonkeyPatch
) -> None:
    synced: list[str] = []
    real = os.fsync

    def spy(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            synced.append(os.path.realpath(dir_of(fd)))
        real(fd)

    monkeypatch.setattr(os, "fsync", spy)
    root = os.path.realpath(tmp_path / "ix")
    with Index.open(tmp_path / "ix", schema, options=SMALL) as ix:
        assert synced.count(root) >= 2  # index.json and wal.log creation
        synced.clear()
        ix.put_many([doc(i) for i in range(3)])
        ix.flush()
        # new segment file -> segments/, manifest rename -> root, WAL reset -> root
        assert os.path.join(root, "segments") in synced
        assert synced.count(root) >= 2
