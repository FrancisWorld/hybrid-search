"""Integer codecs for postings: LEB128 varints, delta coding and postings blocks.

A postings list for one term is stored as::

    varint  n_docs
    varint  len(doc_bytes)   doc_bytes   = delta-coded doc ids, varint each
    varint  len(tf_bytes)    tf_bytes    = term frequencies, varint each
    ...                      pos_bytes   = per doc, ``tf`` delta-coded positions

Frequencies are kept apart from positions so that plain BM25 scoring never has
to decode positions; only phrase queries pay for them.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


class CodecError(ValueError):
    """Raised when a byte stream is truncated or malformed."""


def encode_varint(value: int) -> bytes:
    """Encode a non-negative integer as an unsigned LEB128 varint."""
    if value < 0:
        raise ValueError("varint cannot encode negative values")
    out = bytearray()
    while value > 0x7F:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def encode_varints(values: Iterable[int]) -> bytes:
    return b"".join(encode_varint(value) for value in values)


def decode_varint(buf: bytes | memoryview, pos: int = 0) -> tuple[int, int]:
    """Decode one varint at ``pos``; return ``(value, next_pos)``."""
    result = 0
    shift = 0
    n = len(buf)
    while True:
        if pos >= n:
            raise CodecError("truncated varint")
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7
        if shift > 63:
            raise CodecError("varint too long")


def decode_varints(buf: bytes | memoryview, count: int, pos: int = 0) -> tuple[list[int], int]:
    """Decode ``count`` varints starting at ``pos``."""
    out: list[int] = []
    for _ in range(count):
        value, pos = decode_varint(buf, pos)
        out.append(value)
    return out, pos


def delta_encode(values: Sequence[int]) -> list[int]:
    """Turn a non-decreasing sequence into gaps (first value is kept as is)."""
    out: list[int] = []
    prev = 0
    for value in values:
        if value < prev:
            raise ValueError("delta coding needs a non-decreasing sequence")
        out.append(value - prev)
        prev = value
    return out


def delta_decode(gaps: Iterable[int]) -> list[int]:
    out: list[int] = []
    total = 0
    for gap in gaps:
        total += gap
        out.append(total)
    return out


@dataclass(frozen=True)
class PostingsList:
    """Decoded postings: parallel arrays, plus positions when requested."""

    doc_ids: NDArray[np.int64]
    freqs: NDArray[np.int64]
    positions: list[list[int]] | None = None


def encode_postings(doc_ids: Sequence[int], positions: Sequence[Sequence[int]]) -> bytes:
    """Encode a postings list. ``positions[i]`` are the sorted positions in ``doc_ids[i]``.

    The term frequency of a document is ``len(positions[i])``, so it is never
    possible to store a frequency that disagrees with the positions.
    """
    if len(doc_ids) != len(positions):
        raise ValueError("doc_ids and positions must be the same length")
    if any(not pos for pos in positions):
        raise ValueError("every posting needs at least one position")
    doc_bytes = encode_varints(delta_encode(doc_ids))
    tf_bytes = encode_varints(len(p) for p in positions)
    pos_bytes = bytearray()
    for plist in positions:
        pos_bytes += encode_varints(delta_encode(plist))
    return b"".join(
        (
            encode_varint(len(doc_ids)),
            encode_varint(len(doc_bytes)),
            doc_bytes,
            encode_varint(len(tf_bytes)),
            tf_bytes,
            bytes(pos_bytes),
        )
    )


def decode_postings(buf: bytes | memoryview, with_positions: bool = False) -> PostingsList:
    n_docs, pos = decode_varint(buf, 0)
    doc_len, pos = decode_varint(buf, pos)
    gaps, end = decode_varints(buf, n_docs, pos)
    if end - pos != doc_len:
        raise CodecError("doc id block length mismatch")
    pos = end
    tf_len, pos = decode_varint(buf, pos)
    freqs, end = decode_varints(buf, n_docs, pos)
    if end - pos != tf_len:
        raise CodecError("frequency block length mismatch")
    doc_ids = np.cumsum(np.asarray(gaps, dtype=np.int64))
    freq_arr = np.asarray(freqs, dtype=np.int64)
    positions: list[list[int]] | None = None
    if with_positions:
        positions = []
        pos = end
        for tf in freqs:
            plist, pos = decode_varints(buf, tf, pos)
            positions.append(delta_decode(plist))
    return PostingsList(doc_ids, freq_arr, positions)
