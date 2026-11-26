from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from hybridsearch.codec import (
    CodecError,
    decode_postings,
    decode_varint,
    decode_varints,
    delta_decode,
    delta_encode,
    encode_postings,
    encode_varint,
    encode_varints,
)

U63 = st.integers(min_value=0, max_value=2**63 - 1)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, b"\x00"), (127, b"\x7f"), (128, b"\x80\x01"), (300, b"\xac\x02")],
)
def test_varint_known_vectors(value: int, expected: bytes) -> None:
    assert encode_varint(value) == expected
    assert decode_varint(expected) == (value, len(expected))


@given(U63)
def test_varint_roundtrip(value: int) -> None:
    encoded = encode_varint(value)
    assert decode_varint(encoded) == (value, len(encoded))


@given(st.lists(U63, max_size=200))
def test_varints_roundtrip_stream(values: list[int]) -> None:
    buf = encode_varints(values)
    decoded, end = decode_varints(buf, len(values))
    assert decoded == values
    assert end == len(buf)


def test_negative_rejected() -> None:
    with pytest.raises(ValueError, match="negative"):
        encode_varint(-1)
    with pytest.raises(ValueError, match="negative"):
        encode_varints([1, -2])


def test_truncated_and_overlong_varints() -> None:
    with pytest.raises(CodecError, match="truncated"):
        decode_varint(b"\x80")
    with pytest.raises(CodecError, match="too long"):
        decode_varint(b"\xff" * 12)


@given(st.lists(st.integers(0, 10**9), max_size=100))
def test_delta_roundtrip(values: list[int]) -> None:
    ordered = sorted(values)
    assert delta_decode(delta_encode(ordered)) == ordered


def test_delta_rejects_decreasing() -> None:
    with pytest.raises(ValueError, match="non-decreasing"):
        delta_encode([3, 2])


@st.composite
def postings_lists(draw: st.DrawFn) -> tuple[list[int], list[list[int]]]:
    docs = sorted(draw(st.sets(st.integers(0, 100_000), min_size=0, max_size=40)))
    positions = [
        sorted(draw(st.sets(st.integers(0, 5_000), min_size=1, max_size=12))) for _ in docs
    ]
    return docs, positions


@given(postings_lists())
def test_postings_roundtrip(data: tuple[list[int], list[list[int]]]) -> None:
    docs, positions = data
    blob = encode_postings(docs, positions)
    plain = decode_postings(blob)
    assert plain.doc_ids.tolist() == docs
    assert plain.freqs.tolist() == [len(p) for p in positions]
    assert plain.positions is None
    full = decode_postings(blob, with_positions=True)
    assert full.positions == positions


def test_delta_coding_compresses_dense_lists() -> None:
    docs = list(range(10_000, 11_000))
    blob = encode_postings(docs, [[0]] * len(docs))
    # 1000 docs: ~1 byte gap + 1 byte tf + 1 byte position, versus 8 bytes per raw int64 id.
    assert len(blob) < 3.2 * len(docs)
    assert len(blob) < 8 * len(docs) / 2


def test_postings_validation_and_corruption() -> None:
    with pytest.raises(ValueError, match="same length"):
        encode_postings([1], [])
    with pytest.raises(ValueError, match="at least one position"):
        encode_postings([1], [[]])
    blob = encode_postings([1, 5], [[0], [2, 3]])
    with pytest.raises(CodecError):
        decode_postings(blob[:-2], with_positions=True)
    bad = bytes([blob[0], blob[1] + 1]) + blob[2:]
    with pytest.raises(CodecError, match="block length"):
        decode_postings(bad)
