"""Structured filters over keyword and number fields (all conditions are ANDed).

Wire format, per field: a scalar means equality, a list means "any of", and an
object may use ``eq``, ``in``, ``gt``, ``gte``, ``lt``, ``lte`` (numbers only
for the range operators)::

    {"category": ["coffee", "tea"], "year": {"gte": 2020, "lt": 2025}}
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from hybridsearch.errors import ValidationError
from hybridsearch.schema import Schema

_RANGE_OPS = {"gt", "gte", "lt", "lte"}
_ALL_OPS = _RANGE_OPS | {"eq", "in"}


@dataclass(frozen=True)
class Condition:
    field: str
    op: str
    value: str | float | tuple[str, ...] | tuple[float, ...]


class ColumnSource(Protocol):
    """What a filter needs from a segment."""

    @property
    def n_docs(self) -> int: ...

    def keyword_docs(self, field: str, value: str) -> NDArray[np.int64]: ...

    def number_column(self, field: str) -> NDArray[np.float64]: ...


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def parse_filters(raw: Mapping[str, object] | None, schema: Schema) -> tuple[Condition, ...]:
    if not raw:
        return ()
    conditions: list[Condition] = []
    for field, spec in raw.items():
        is_keyword = field in schema.keyword_fields
        if not is_keyword and field not in schema.number_fields:
            raise ValidationError(f"cannot filter on unknown field {field!r}")
        ops: dict[str, object]
        if isinstance(spec, Mapping):
            ops = dict(spec)
        elif isinstance(spec, list):
            ops = {"in": spec}
        else:
            ops = {"eq": spec}
        if not ops:
            raise ValidationError(f"empty filter for {field!r}")
        for op, value in ops.items():
            conditions.append(_condition(field, op, value, is_keyword))
    return tuple(conditions)


def _condition(field: str, op: str, value: object, is_keyword: bool) -> Condition:
    if op not in _ALL_OPS:
        raise ValidationError(f"unknown filter operator {op!r} for {field!r}")
    if is_keyword and op in _RANGE_OPS:
        raise ValidationError(f"range operator {op!r} needs a number field, {field!r} is keyword")
    if op == "in":
        if not isinstance(value, list) or not value:
            raise ValidationError(f"'in' for {field!r} needs a non-empty list")
        items = value
    else:
        items = [value]
    for item in items:
        ok = isinstance(item, str) if is_keyword else _is_number(item)
        if not ok:
            kind = "string" if is_keyword else "number"
            raise ValidationError(f"filter value for {field!r} must be a {kind}")
    if op == "in":
        return Condition(field, op, tuple(items))
    return Condition(field, op, items[0])


def evaluate(conditions: tuple[Condition, ...], source: ColumnSource) -> NDArray[np.bool_]:
    """Boolean mask over the segment's local doc ids."""
    mask = np.ones(source.n_docs, dtype=bool)
    for cond in conditions:
        mask &= _evaluate_one(cond, source)
    return mask


def _evaluate_one(cond: Condition, source: ColumnSource) -> NDArray[np.bool_]:
    out = np.zeros(source.n_docs, dtype=bool)
    if isinstance(cond.value, str):
        out[source.keyword_docs(cond.field, cond.value)] = True
        return out
    if isinstance(cond.value, tuple) and cond.value and isinstance(cond.value[0], str):
        for item in cond.value:
            out[source.keyword_docs(cond.field, str(item))] = True
        return out
    column = source.number_column(cond.field)
    if cond.op == "in":
        assert isinstance(cond.value, tuple)
        return np.isin(column, np.asarray(cond.value, dtype=np.float64))
    number = float(cond.value)  # type: ignore[arg-type]
    with np.errstate(invalid="ignore"):  # NaN (missing) compares False
        result: NDArray[np.bool_] = {
            "eq": column == number,
            "gt": column > number,
            "gte": column >= number,
            "lt": column < number,
            "lte": column <= number,
        }[cond.op]
        return result
