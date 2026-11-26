"""Immutable on-disk segments: writer, reader and in-memory build data.

Sections of a segment file (see :mod:`hybridsearch.segment_format`):

``meta``      JSON: doc count, schema, per-field token totals, vector dim
``ids``       JSON list of external ids, local doc id = index
``doc_off``   uint64[n+1] offsets into ``doc_blob``
``doc_blob``  zlib-compressed JSON of each stored document
``lengths``   uint32[n_text_fields, n] token counts (for BM25 length norm)
``terms``     term dictionary: ``key | df | offset | length`` per term, sorted
``postings``  concatenated postings blocks (see :mod:`hybridsearch.codec`)
``keywords``  JSON: field -> value -> local doc ids
``numbers``   float64[n_number_fields, n], NaN where missing
``vectors``   float32[n, dim]
``hnsw``      HNSW graph (only for segments above ``hnsw_min_docs``)
"""

from __future__ import annotations

import json
import mmap
import os
import zlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hybridsearch.analysis import Analyzer
from hybridsearch.codec import (
    PostingsList,
    decode_postings,
    decode_varint,
    encode_postings,
    encode_varint,
)
from hybridsearch.errors import CorruptIndexError
from hybridsearch.hnsw import BoolMask, HNSWIndex, brute_force
from hybridsearch.schema import Document, Schema
from hybridsearch.segment_format import pack_sections, unpack_sections

_KEY_SEP = "\x00"
FloatMatrix = NDArray[np.float32]


@dataclass(frozen=True)
class SegmentOptions:
    hnsw_min_docs: int = 64
    hnsw_m: int = 16
    hnsw_ef_construction: int = 100
    hnsw_seed: int = 0


@dataclass
class SegmentData:
    """Everything needed to write a segment, independent of where it came from."""

    stored: list[dict[str, Any]]
    field_lengths: dict[str, list[int]]
    postings: dict[tuple[str, str], list[tuple[int, list[int]]]]
    vectors: FloatMatrix
    ids: list[str] = field(init=False)

    def __post_init__(self) -> None:
        self.ids = [doc["id"] for doc in self.stored]


@dataclass(frozen=True)
class AnalyzedDoc:
    """One document after analysis: cheap to keep in the memtable and to assemble later."""

    stored: dict[str, Any]
    lengths: dict[str, int]
    terms: dict[tuple[str, str], list[int]]


def analyze_document(doc: Document, schema: Schema, analyzer: Analyzer) -> AnalyzedDoc:
    lengths: dict[str, int] = {}
    terms: dict[tuple[str, str], list[int]] = {}
    for name in schema.text_fields:
        tokens = analyzer.analyze(doc.fields.get(name, ""))
        lengths[name] = len(tokens)
        for tok in tokens:
            terms.setdefault((name, tok.term), []).append(tok.position)
    return AnalyzedDoc(doc.to_dict(), lengths, terms)


def assemble_segment_data(
    docs: Sequence[AnalyzedDoc], vectors: FloatMatrix, schema: Schema
) -> SegmentData:
    """Collect analysed documents into postings (local doc id = list index)."""
    lengths: dict[str, list[int]] = {name: [] for name in schema.text_fields}
    postings: dict[tuple[str, str], list[tuple[int, list[int]]]] = {}
    for local, doc in enumerate(docs):
        for name in schema.text_fields:
            lengths[name].append(doc.lengths[name])
        for key, positions in doc.terms.items():
            postings.setdefault(key, []).append((local, positions))
    return SegmentData([d.stored for d in docs], lengths, postings, vectors)


def analyze_documents(
    docs: Sequence[Document], vectors: FloatMatrix, schema: Schema, analyzer: Analyzer
) -> SegmentData:
    return assemble_segment_data(
        [analyze_document(d, schema, analyzer) for d in docs], vectors, schema
    )


def _columns(
    stored: Sequence[Mapping[str, Any]], schema: Schema
) -> tuple[dict[str, dict[str, list[int]]], NDArray[np.float64]]:
    keywords: dict[str, dict[str, list[int]]] = {name: {} for name in schema.keyword_fields}
    numbers = np.full((len(schema.number_fields), len(stored)), np.nan, dtype=np.float64)
    for local, doc in enumerate(stored):
        meta = doc["meta"]
        for name in schema.keyword_fields:
            for value in meta.get(name, ()):
                keywords[name].setdefault(value, []).append(local)
        for row, name in enumerate(schema.number_fields):
            if name in meta:
                numbers[row, local] = meta[name]
    return keywords, numbers


def serialize_segment(data: SegmentData, schema: Schema, options: SegmentOptions) -> bytes:
    n = len(data.stored)
    if n == 0:
        raise ValueError("cannot write an empty segment")
    if data.vectors.shape != (n, schema.embedding_dim):
        raise ValueError("vector matrix does not match documents and embedding_dim")

    blobs = [zlib.compress(json.dumps(d, separators=(",", ":")).encode()) for d in data.stored]
    offsets = np.zeros(n + 1, dtype=np.uint64)
    np.cumsum([len(b) for b in blobs], out=offsets[1:])

    postings = bytearray()
    entries = bytearray()
    for key in sorted(data.postings):
        plist = data.postings[key]
        block = encode_postings([d for d, _ in plist], [p for _, p in plist])
        raw_key = f"{key[0]}{_KEY_SEP}{key[1]}".encode()
        entries += encode_varint(len(raw_key)) + raw_key
        entries += encode_varint(len(plist)) + encode_varint(len(postings))
        entries += encode_varint(len(block))
        postings += block

    keywords, numbers = _columns(data.stored, schema)
    lengths = np.asarray([data.field_lengths[f] for f in schema.text_fields], dtype=np.uint32)
    vectors = np.ascontiguousarray(data.vectors, dtype=np.float32)
    meta = {
        "n_docs": n,
        "schema": schema.to_dict(),
        "total_len": {f: int(sum(data.field_lengths[f])) for f in schema.text_fields},
        "n_terms": len(data.postings),
    }
    sections: dict[str, bytes] = {
        "meta": json.dumps(meta).encode(),
        "ids": json.dumps(data.ids).encode(),
        "doc_off": offsets.tobytes(),
        "doc_blob": b"".join(blobs),
        "lengths": lengths.tobytes(),
        "terms": bytes(entries),
        "postings": bytes(postings),
        "keywords": json.dumps(keywords).encode(),
        "numbers": numbers.tobytes(),
        "vectors": vectors.tobytes(),
    }
    if n >= options.hnsw_min_docs:
        graph = HNSWIndex.build(
            vectors,
            m=options.hnsw_m,
            ef_construction=options.hnsw_ef_construction,
            seed=options.hnsw_seed,
        )
        sections["hnsw"] = graph.to_bytes()
    return pack_sections(sections)


def write_segment_file(path: Path, blob: bytes) -> None:
    """Durably create ``path``: write a temp file, fsync it, rename, fsync the directory."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        fh.write(blob)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)
    fsync_dir(path.parent)


def fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Segment:
    """Read-only view of one segment. Safe to share between threads."""

    def __init__(self, buf: bytes | memoryview, *, verify: bool = True, closer: Any = None) -> None:
        self._closer = closer
        self._buf = buf
        sections = unpack_sections(buf, verify=verify)
        try:
            meta = json.loads(bytes(sections["meta"]))
            self.schema = Schema.from_dict(meta["schema"])
            self.n_docs: int = int(meta["n_docs"])
            self.total_len: dict[str, int] = {k: int(v) for k, v in meta["total_len"].items()}
            self.ids: list[str] = json.loads(bytes(sections["ids"]))
            self._doc_off = np.frombuffer(sections["doc_off"], dtype=np.uint64).copy()
            self._doc_blob = sections["doc_blob"]
            self._lengths: NDArray[np.uint32] = np.frombuffer(
                sections["lengths"], dtype=np.uint32
            ).reshape(len(self.schema.text_fields), self.n_docs)
            self._postings = sections["postings"]
            self._terms = self._read_dictionary(sections["terms"], int(meta["n_terms"]))
            self._keywords: dict[str, dict[str, NDArray[np.int64]]] = {
                f: {v: np.asarray(ids, dtype=np.int64) for v, ids in values.items()}
                for f, values in json.loads(bytes(sections["keywords"])).items()
            }
            self._numbers: NDArray[np.float64] = np.frombuffer(
                sections["numbers"], dtype=np.float64
            ).reshape(len(self.schema.number_fields), self.n_docs)
            self.vectors: FloatMatrix = np.frombuffer(
                sections["vectors"], dtype=np.float32
            ).reshape(self.n_docs, self.schema.embedding_dim)
            self.hnsw: HNSWIndex | None = (
                HNSWIndex.from_bytes(bytes(sections["hnsw"]), self.vectors)
                if "hnsw" in sections
                else None
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise CorruptIndexError(f"segment is inconsistent: {exc}") from exc
        if len(self.ids) != self.n_docs or len(self._doc_off) != self.n_docs + 1:
            raise CorruptIndexError("segment doc tables disagree with its doc count")
        self.id_to_local = {doc_id: i for i, doc_id in enumerate(self.ids)}
        self._text_index = {f: i for i, f in enumerate(self.schema.text_fields)}
        self._number_index = {f: i for i, f in enumerate(self.schema.number_fields)}

    @classmethod
    def open(cls, path: Path, *, verify: bool = True) -> Segment:
        with path.open("rb") as fh:
            mapped = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        view = memoryview(mapped)
        return cls(view, verify=verify, closer=(view, mapped))

    def close(self) -> None:
        """Release the file mapping. The segment must not be used afterwards."""
        if self._closer is not None:
            view, mapped = self._closer
            self._closer = None
            for name in ("_doc_blob", "_postings", "_lengths", "_numbers", "vectors", "hnsw"):
                setattr(self, name, None)
            self._buf = b""
            view.release()
            mapped.close()

    @staticmethod
    def _read_dictionary(
        buf: memoryview, n_terms: int
    ) -> dict[tuple[str, str], tuple[int, int, int]]:
        terms: dict[tuple[str, str], tuple[int, int, int]] = {}
        pos = 0
        for _ in range(n_terms):
            key_len, pos = decode_varint(buf, pos)
            field_name, _, term = bytes(buf[pos : pos + key_len]).decode().partition(_KEY_SEP)
            pos += key_len
            df, pos = decode_varint(buf, pos)
            offset, pos = decode_varint(buf, pos)
            length, pos = decode_varint(buf, pos)
            terms[(field_name, term)] = (df, offset, length)
        return terms

    # ------------------------------------------------------------------- lookups

    def doc_freq(self, field_name: str, term: str) -> int:
        entry = self._terms.get((field_name, term))
        return entry[0] if entry else 0

    def postings(
        self, field_name: str, term: str, *, with_positions: bool = False
    ) -> PostingsList | None:
        entry = self._terms.get((field_name, term))
        if entry is None:
            return None
        _, offset, length = entry
        return decode_postings(self._postings[offset : offset + length], with_positions)

    def terms(self) -> Iterator[tuple[str, str]]:
        return iter(self._terms)

    def field_lengths(self, field_name: str) -> NDArray[np.uint32]:
        row: NDArray[np.uint32] = self._lengths[self._text_index[field_name]]
        return row

    def stored(self, local: int) -> dict[str, Any]:
        start, end = int(self._doc_off[local]), int(self._doc_off[local + 1])
        doc: dict[str, Any] = json.loads(zlib.decompress(self._doc_blob[start:end]))
        return doc

    def keyword_docs(self, field_name: str, value: str) -> NDArray[np.int64]:
        return self._keywords.get(field_name, {}).get(value, np.empty(0, dtype=np.int64))

    def keyword_values(self, field_name: str) -> Mapping[str, NDArray[np.int64]]:
        return self._keywords.get(field_name, {})

    def number_column(self, field_name: str) -> NDArray[np.float64]:
        row: NDArray[np.float64] = self._numbers[self._number_index[field_name]]
        return row

    def vector_search(
        self, query: FloatMatrix, k: int, allowed: BoolMask | None, ef: int
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        if self.hnsw is not None:
            return self.hnsw.search(query, k, ef=ef, allowed=allowed)
        return brute_force(self.vectors, query, k, allowed)
