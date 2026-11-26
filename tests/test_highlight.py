from __future__ import annotations

import pytest

from hybridsearch.analysis import get_analyzer
from hybridsearch.highlight import (
    collect_targets,
    find_spans,
    render_fragment,
    terms_only_targets,
)
from hybridsearch.query import analyze_query, parse_query

EN = get_analyzer("en")
PT = get_analyzer("pt")


def spans_for(text: str, query: str, analyzer=EN, field: str = "body") -> list[tuple[int, int]]:  # type: ignore[no-untyped-def]
    targets = collect_targets(analyze_query(parse_query(query), analyzer))
    terms, phrases = targets.for_field(field)
    return find_spans(text, analyzer, terms, phrases)


def words(text: str, spans: list[tuple[int, int]]) -> list[str]:
    return [text[s:e] for s, e in spans]


def test_stemmed_and_case_insensitive_matches() -> None:
    text = "Running shoes for runners who ran yesterday"
    assert words(text, spans_for(text, "run")) == ["Running"]
    assert words(text, spans_for(text, "RUNNERS shoe")) == ["shoes", "runners"]


def test_accent_insensitive_portuguese() -> None:
    text = "A informação é importante; informacoes úteis"
    assert words(text, spans_for(text, "informação", PT)) == ["informação", "informacoes"]


def test_phrase_highlights_only_full_phrase_with_gaps() -> None:
    text = "king of spain met king spain and the king of spain"
    spans = spans_for(text, '"king of spain"')
    assert words(text, spans) == ["king", "spain", "king", "spain"]
    assert [s for s, _ in spans] == [0, 8, 37, 45]


def test_prohibited_terms_are_not_highlighted_and_fields_respected() -> None:
    text = "espresso and decaf"
    assert words(text, spans_for(text, "espresso -decaf")) == ["espresso"]
    assert spans_for(text, "title:espresso", field="body") == []
    assert words(text, spans_for(text, "title:espresso", field="title")) == ["espresso"]


def test_vector_query_targets() -> None:
    targets = terms_only_targets({"espresso"})
    terms, phrases = targets.for_field("body")
    assert words("An Espresso", find_spans("An Espresso", EN, terms, phrases)) == ["Espresso"]


def test_render_marks_and_escapes_html() -> None:
    text = "Use <script> with espresso & friends"
    spans = spans_for(text, "espresso")
    html, marks = render_fragment(text, spans)
    assert html == "Use &lt;script&gt; with <mark>espresso</mark> &amp; friends"
    assert marks == 1


def test_fragment_centres_on_densest_window_and_adds_ellipses() -> None:
    filler = "lorem ipsum dolor sit amet " * 20
    text = f"espresso {filler} pressure grinder espresso pressure {filler}"
    spans = spans_for(text, "espresso pressure grinder")
    html, marks = render_fragment(text, spans, size=100)
    assert marks == 4
    assert html.startswith("… ")
    assert html.endswith(" …")
    assert "<mark>grinder</mark>" in html
    assert len(html) < 220


def test_fragment_without_matches_returns_leading_text() -> None:
    text = "word " * 100
    html, marks = render_fragment(text, [], size=50)
    assert marks == 0
    assert html.endswith(" …")
    assert "<mark>" not in html
    assert render_fragment("short", [])[0] == "short"


@pytest.mark.parametrize("size", [40, 80, 160])
def test_fragment_never_splits_marks(size: int) -> None:
    text = " ".join(["espresso", "machine", "pressure"] * 40)
    html, marks = render_fragment(text, spans_for(text, "espresso machine"), size=size)
    assert html.count("<mark>") == html.count("</mark>") == marks
