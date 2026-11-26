"""Write-ahead log with CRC-framed records and torn-tail recovery.

File layout::

    "HSWAL001"                       8-byte magic
    record*                          u32 payload_len | u32 crc32(payload) | payload
    payload = u64 seq | u8 op | json

A crash can leave the last record half written. :meth:`WriteAheadLog.recover`
reads records until the first one whose frame is incomplete or whose checksum
fails, and truncates the file there, so the log always ends on a record
boundary before new appends.
"""

from __future__ import annotations

import json
import os
import struct
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from hybridsearch.fsutil import fsync_dir

MAGIC = b"HSWAL001"
_FRAME = struct.Struct("<II")
_HEAD = struct.Struct("<QB")
MAX_RECORD_BYTES = 64 * 1024 * 1024

OP_PUT = 1
OP_DELETE = 2
_OPS = {OP_PUT: "put", OP_DELETE: "delete"}
_OP_CODES = {name: code for code, name in _OPS.items()}


class WalError(RuntimeError):
    """The log cannot be used (for example an unknown file magic)."""


@dataclass(frozen=True, slots=True)
class WalRecord:
    seq: int
    op: str
    data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    records: int
    dropped_bytes: int

    @property
    def was_torn(self) -> bool:
        return self.dropped_bytes > 0


def encode_record(record: WalRecord) -> bytes:
    payload = _HEAD.pack(record.seq, _OP_CODES[record.op]) + json.dumps(
        record.data, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return _FRAME.pack(len(payload), zlib.crc32(payload)) + payload


def parse_records(blob: bytes) -> tuple[list[WalRecord], int]:
    """Parse as many valid records as possible; return them and the valid length.

    ``blob`` is the file content including the magic. The returned length is
    the offset one past the last good record.
    """
    if not blob.startswith(MAGIC):
        if len(blob) < len(MAGIC) and MAGIC.startswith(blob):
            return [], 0
        raise WalError("not a hybrid-search WAL file")
    pos = len(MAGIC)
    records: list[WalRecord] = []
    while pos + _FRAME.size <= len(blob):
        length, crc = _FRAME.unpack_from(blob, pos)
        start = pos + _FRAME.size
        end = start + length
        if length < _HEAD.size or length > MAX_RECORD_BYTES or end > len(blob):
            break
        payload = blob[start:end]
        if zlib.crc32(payload) != crc:
            break
        seq, op_code = _HEAD.unpack_from(payload)
        op = _OPS.get(op_code)
        if op is None:
            break
        try:
            data = json.loads(payload[_HEAD.size :])
        except ValueError:
            break
        if not isinstance(data, dict):
            break
        records.append(WalRecord(seq, op, data))
        pos = end
    return records, pos


class WriteAheadLog:
    """Append-only log. Not thread-safe: callers serialise writers."""

    def __init__(self, path: Path, *, fsync: bool = True) -> None:
        self.path = path
        self._fsync = fsync
        self._fh: BinaryIO | None = None

    def recover(self) -> tuple[list[WalRecord], RecoveryReport]:
        """Read every valid record, truncate any torn tail and open for append."""
        if self._fh is not None:
            raise WalError("log already open")
        blob = self.path.read_bytes() if self.path.exists() else b""
        records, valid = parse_records(blob)
        dropped = len(blob) - valid if blob else 0
        if valid == 0:
            with self.path.open("wb") as out:
                out.write(MAGIC)
                out.flush()
                os.fsync(out.fileno())
            fsync_dir(self.path.parent)  # the new directory entry must survive power loss too
            dropped = len(blob) - len(MAGIC) if len(blob) > len(MAGIC) else 0
        elif dropped:
            with self.path.open("r+b") as fh:
                fh.truncate(valid)
                os.fsync(fh.fileno())
        self._fh = self.path.open("ab")
        return records, RecoveryReport(len(records), max(dropped, 0))

    def append(self, records: Sequence[WalRecord]) -> None:
        """Append a batch with a single write and a single fsync (group commit)."""
        fh = self._require_open()
        if not records:
            return
        fh.write(b"".join(encode_record(r) for r in records))
        fh.flush()
        if self._fsync:
            os.fsync(fh.fileno())

    def reset(self) -> None:
        """Discard all records (called once their effects are durable elsewhere)."""
        fh = self._require_open()
        fh.close()
        self._fh = None
        with self.path.open("wb") as out:
            out.write(MAGIC)
            out.flush()
            os.fsync(out.fileno())
        fsync_dir(self.path.parent)
        self._fh = self.path.open("ab")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def _require_open(self) -> BinaryIO:
        if self._fh is None:
            raise WalError("log is not open")
        return self._fh
