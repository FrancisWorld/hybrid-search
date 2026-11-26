from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hybridsearch.wal import (
    MAGIC,
    WalError,
    WalRecord,
    WriteAheadLog,
    encode_record,
    parse_records,
)
from tests.helpers import dir_of


def make_records(n: int, start: int = 1) -> list[WalRecord]:
    return [
        WalRecord(start + i, "put" if i % 3 else "delete", {"id": f"doc-{i}", "text": "héllo ☃"})
        for i in range(n)
    ]


def test_roundtrip_and_reopen(tmp_path: Path) -> None:
    path = tmp_path / "wal.log"
    wal = WriteAheadLog(path)
    assert wal.recover()[0] == []
    batch = make_records(5)
    wal.append(batch)
    wal.close()

    reopened = WriteAheadLog(path)
    records, report = reopened.recover()
    assert records == batch
    assert not report.was_torn
    reopened.append(make_records(1, start=6))
    reopened.close()
    assert len(WriteAheadLog(path).recover()[0]) == 6


def test_torn_tail_is_truncated_and_appends_continue(tmp_path: Path) -> None:
    path = tmp_path / "wal.log"
    wal = WriteAheadLog(path)
    wal.recover()
    batch = make_records(4)
    wal.append(batch)
    wal.close()
    full = path.read_bytes()
    path.write_bytes(full[:-5])

    wal2 = WriteAheadLog(path)
    records, report = wal2.recover()
    assert records == batch[:3]
    assert report.was_torn
    assert path.stat().st_size == len(full) - len(encode_record(batch[3]))
    wal2.append([batch[3]])
    wal2.close()
    assert WriteAheadLog(path).recover()[0] == batch


def test_corrupted_byte_stops_replay_at_bad_record(tmp_path: Path) -> None:
    path = tmp_path / "wal.log"
    wal = WriteAheadLog(path)
    wal.recover()
    batch = make_records(5)
    wal.append(batch)
    wal.close()
    data = bytearray(path.read_bytes())
    offset = len(MAGIC) + sum(len(encode_record(r)) for r in batch[:2]) + 12
    data[offset] ^= 0xFF
    path.write_bytes(bytes(data))
    records, report = WriteAheadLog(path).recover()
    assert records == batch[:2]
    assert report.dropped_bytes > 0


@given(st.integers(1, 12), st.data())
@settings(max_examples=60, deadline=None)
def test_any_truncation_recovers_a_record_prefix(n: int, data: st.DataObject) -> None:
    """Whatever byte the crash cuts at, recovery yields exactly a prefix of the writes."""
    batch = make_records(n)
    blob = MAGIC + b"".join(encode_record(r) for r in batch)
    cut = data.draw(st.integers(0, len(blob)))
    records, valid = parse_records(blob[:cut])
    assert records == batch[: len(records)]
    assert valid <= cut
    boundaries = [len(MAGIC)]
    for r in batch:
        boundaries.append(boundaries[-1] + len(encode_record(r)))
    expected = max(i for i, b in enumerate(boundaries) if b <= cut) if cut >= len(MAGIC) else 0
    assert len(records) == expected


def test_foreign_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "wal.log"
    path.write_bytes(b"something else entirely")
    with pytest.raises(WalError, match="not a hybrid-search WAL"):
        WriteAheadLog(path).recover()


def test_partial_magic_is_treated_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "wal.log"
    path.write_bytes(MAGIC[:3])
    wal = WriteAheadLog(path)
    assert wal.recover()[0] == []
    wal.append(make_records(1))
    wal.close()
    assert len(WriteAheadLog(path).recover()[0]) == 1


def test_reset_discards_records_and_guards(tmp_path: Path) -> None:
    path = tmp_path / "wal.log"
    wal = WriteAheadLog(path, fsync=False)
    with pytest.raises(WalError, match="not open"):
        wal.append(make_records(1))
    wal.recover()
    with pytest.raises(WalError, match="already open"):
        wal.recover()
    wal.append(make_records(3))
    wal.reset()
    wal.append(make_records(1, start=4))
    wal.close()
    records, _ = WriteAheadLog(path).recover()
    assert [r.seq for r in records] == [4]


def _spy_dir_fsyncs(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the path of every directory whose file descriptor is fsynced."""
    synced: list[str] = []
    real = os.fsync

    def spy(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            synced.append(dir_of(fd))
        real(fd)

    monkeypatch.setattr(os, "fsync", spy)
    return synced


def test_creating_and_resetting_the_wal_fsyncs_the_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synced = _spy_dir_fsyncs(monkeypatch)
    wal = WriteAheadLog(tmp_path / "wal.log")
    wal.recover()
    assert [os.path.realpath(p) for p in synced] == [os.path.realpath(tmp_path)]
    wal.append(make_records(2))
    wal.reset()
    assert len(synced) == 2
    wal.close()
    synced.clear()
    WriteAheadLog(tmp_path / "wal.log").recover()  # existing, valid log: nothing to create
    assert synced == []
