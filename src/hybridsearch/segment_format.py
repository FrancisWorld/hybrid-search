"""Container format shared by segment files: checksummed, 8-byte aligned sections.

::

    MAGIC | section* | toc_json | u32 toc_len | u32 toc_crc | END_MAGIC

The table of contents sits at the end so a writer can stream sections out
without knowing offsets in advance, and a reader can reject a truncated file
by looking at the last bytes only.
"""

from __future__ import annotations

import json
import struct
import zlib

from hybridsearch.errors import CorruptIndexError

MAGIC = b"HSSEG\x00\x01\x00"
END_MAGIC = b"HSEND\x00\x01\x00"
_TAIL = struct.Struct("<II")
_ALIGN = 8


def pack_sections(sections: dict[str, bytes]) -> bytes:
    out = bytearray(MAGIC)
    toc: dict[str, list[int]] = {}
    for name, blob in sections.items():
        out += b"\x00" * (-len(out) % _ALIGN)
        toc[name] = [len(out), len(blob), zlib.crc32(blob)]
        out += blob
    toc_blob = json.dumps(toc, separators=(",", ":")).encode()
    out += toc_blob
    out += _TAIL.pack(len(toc_blob), zlib.crc32(toc_blob))
    out += END_MAGIC
    return bytes(out)


def unpack_sections(buf: bytes | memoryview, *, verify: bool = True) -> dict[str, memoryview]:
    view = memoryview(buf)
    min_size = len(MAGIC) + _TAIL.size + len(END_MAGIC)
    if len(view) < min_size or bytes(view[: len(MAGIC)]) != MAGIC:
        raise CorruptIndexError("not a segment file (bad magic)")
    if bytes(view[-len(END_MAGIC) :]) != END_MAGIC:
        raise CorruptIndexError("segment file is truncated (missing end marker)")
    tail_at = len(view) - len(END_MAGIC) - _TAIL.size
    toc_len, toc_crc = _TAIL.unpack_from(view, tail_at)
    toc_blob = bytes(view[tail_at - toc_len : tail_at]) if toc_len <= tail_at else b""
    if zlib.crc32(toc_blob) != toc_crc:
        raise CorruptIndexError("segment table of contents failed its checksum")
    try:
        toc = json.loads(toc_blob)
    except ValueError as exc:
        raise CorruptIndexError("segment table of contents is not valid JSON") from exc
    sections: dict[str, memoryview] = {}
    for name, (offset, length, crc) in toc.items():
        if offset < len(MAGIC) or offset + length > tail_at - toc_len:
            raise CorruptIndexError(f"section {name!r} points outside the file")
        section = view[offset : offset + length]
        if verify and zlib.crc32(section) != crc:
            raise CorruptIndexError(f"section {name!r} failed its checksum")
        sections[name] = section
    return sections
