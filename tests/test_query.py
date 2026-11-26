from __future__ import annotations

import contextlib

import pytest
from hypothesis import given
from hypothesis import strategies as st

from hybridsearch.analysis import get_analyzer
from hybridsearch.errors import QueryError
from hybridsearch.query import (
    MAX_DEPTH,
    Bool,
    PhraseNode,
    TermNode,
    Text,
    analyze_query,
    parse_query,
)


def w(text: str, field: str | None = None) -> Text:
    return Text(text, False, field)


def test_adjacent_words_are_should_clauses() -> None:
    assert parse_query("espresso machine") == Bool(should=(w("espresso"), w("machine")))
    assert parse_query("single") == w("single")


def test_plus_minus_and_not() -> None:
    assert parse_query("+a b -c") == Bool(must=(w("a"),), should=(w("b"),), must_not=(w("c"),))
    assert parse_query("a NOT c") == Bool(should=(w("a"),), must_not=(w("c"),))


def test_and_binds_tighter_than_adjacency_and_or() -> None:
    assert parse_query("a b AND c") == Bool(should=(w("a"), Bool(must=(w("b"), w("c")))))
    assert parse_query("a OR b") == Bool(should=(w("a"), w("b")))
    assert parse_query("a AND NOT b") == Bool(must=(w("a"),), must_not=(w("b"),))


def test_phrases_and_fields() -> None:
    assert parse_query('"pressure profile"') == Text("pressure profile", True)
    assert parse_query('title:"pressure profile"') == Text("pressure profile", True, "title")
    assert parse_query("title:espresso body:machine") == Bool(
        should=(w("espresso", "title"), w("machine", "body"))
    )


def test_field_group_applies_to_children_but_not_overrides() -> None:
    node = parse_query("title:(a OR body:b) c")
    assert node == Bool(should=(Bool(should=(w("a", "title"), w("b", "body"))), w("c")))


def test_colon_inside_words_is_not_a_field() -> None:
    assert parse_query("10:30 meeting") == Bool(should=(w("10:30"), w("meeting")))
    assert parse_query("trailing: x") == Bool(should=(w("trailing:"), w("x")))


def test_hyphen_inside_word_and_lone_minus() -> None:
    assert parse_query("state-of-art") == w("state-of-art")
    assert parse_query("a - b") == Bool(should=(w("a"), w("-"), w("b")))


@pytest.mark.parametrize(
    ("query", "fragment"),
    [
        ("", "empty"),
        ("   ", "empty"),
        ('"open phrase', "unterminated"),
        ("(a b", r"missing '\)'"),
        ("a b)", "unbalanced"),
        ("-a", "only prohibited"),
        ("NOT a", "only prohibited"),
        ("a AND", "unexpected"),
        ("AND a", "unexpected"),
        ("()", "empty"),
        ("x" * 3000, "longer than"),
    ],
)
def test_syntax_errors(query: str, fragment: str) -> None:
    with pytest.raises(QueryError, match=fragment):
        parse_query(query)


def test_errors_are_structured() -> None:
    with pytest.raises(QueryError) as info:
        parse_query('a "b')
    assert info.value.code == "query_syntax_error"
    assert info.value.position == 2
    assert info.value.to_dict()["code"] == "query_syntax_error"


def test_nesting_and_clause_limits() -> None:
    parse_query("(" * MAX_DEPTH + "a" + ")" * MAX_DEPTH)
    with pytest.raises(QueryError, match="nested too deeply"):
        parse_query("(" * (MAX_DEPTH + 2) + "a" + ")" * (MAX_DEPTH + 2))
    with pytest.raises(QueryError, match="more than"):
        parse_query(" ".join(["a"] * 300))


def test_analysis_stems_drops_stopwords_and_builds_phrases() -> None:
    an = get_analyzer("en")
    assert analyze_query(parse_query("Running"), an) == TermNode(None, "run")
    assert analyze_query(parse_query("the"), an) is None
    node = analyze_query(parse_query('"king of spain"'), an)
    assert node == PhraseNode(None, ("king", "spain"), (0, 2))
    assert analyze_query(parse_query("state-of-art"), an) == PhraseNode(
        None, ("state", "art"), (0, 2)
    )
    mixed = analyze_query(parse_query("+the cats -dogs"), an)
    assert mixed == Bool(should=(TermNode(None, "cat"),), must_not=(TermNode(None, "dog"),))
    assert analyze_query(parse_query("the AND of"), an) is None
    assert analyze_query(parse_query("cats -the"), an) == TermNode(None, "cat")


@given(st.text(alphabet='ab +-"():AND ORNOT', max_size=40))
def test_parser_never_raises_anything_but_query_error(query: str) -> None:
    with contextlib.suppress(QueryError):
        parse_query(query)
