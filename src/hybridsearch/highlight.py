"""Highlighting and snippet extraction.

Matches are found by re-analysing the stored text with the same analyzer used
at index time, so ``Running`` highlights for the query ``run``. Phrase queries
only highlight where the whole phrase occurs, with stop-word gaps respected.
"""

from __future__ import annotations

import html
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from hybridsearch.analysis import Analyzer
from hybridsearch.query import Bool, Node, PhraseNode, TermNode

Span = tuple[int, int]


@dataclass
class Targets:
    """Terms and phrases to highlight, per text field (``None`` key = every field)."""

    terms: dict[str | None, set[str]] = field(default_factory=dict)
    phrases: dict[str | None, list[PhraseNode]] = field(default_factory=dict)

    def for_field(self, name: str) -> tuple[set[str], list[PhraseNode]]:
        terms = self.terms.get(None, set()) | self.terms.get(name, set())
        phrases = self.phrases.get(None, []) + self.phrases.get(name, [])
        return terms, phrases


def collect_targets(node: Node | None) -> Targets:
    """Gather positive terms/phrases; prohibited clauses are never highlighted."""
    targets = Targets()

    def walk(n: Node) -> None:
        if isinstance(n, TermNode):
            targets.terms.setdefault(n.field, set()).add(n.term)
        elif isinstance(n, PhraseNode):
            targets.phrases.setdefault(n.field, []).append(n)
        elif isinstance(n, Bool):
            for child in (*n.must, *n.should):
                walk(child)

    if node is not None:
        walk(node)
    return targets


def terms_only_targets(terms: Iterable[str]) -> Targets:
    """Highlight targets for free-text (vector) queries: every analysed word."""
    return Targets(terms={None: set(terms)})


def find_spans(
    text: str, analyzer: Analyzer, terms: set[str], phrases: Sequence[PhraseNode]
) -> list[Span]:
    """Character spans in ``text`` matching ``terms`` or any whole ``phrases``."""
    tokens = analyzer.analyze(text)
    spans: set[Span] = {(t.start, t.end) for t in tokens if t.term in terms}
    if phrases:
        by_position = {t.position: t for t in tokens}
        for phrase in phrases:
            for tok in tokens:
                if tok.term != phrase.terms[0]:
                    continue
                matched = [
                    by_position.get(tok.position + off - phrase.offsets[0])
                    for off in phrase.offsets
                ]
                if all(
                    m is not None and m.term == t
                    for m, t in zip(matched, phrase.terms, strict=True)
                ):
                    spans.update((m.start, m.end) for m in matched if m is not None)
    return sorted(spans)


def _best_anchor(spans: Sequence[Span], size: int) -> int:
    """Start offset of the span that has the most spans inside the following window."""
    reach = size - 20
    counts = [
        sum(1 for s in spans if first[0] <= s[0] and s[1] <= first[0] + reach) for first in spans
    ]
    return spans[counts.index(max(counts))][0]


def render_fragment(text: str, spans: Sequence[Span], size: int = 160) -> tuple[str, int]:
    """Return ``(html, marks)``: the window of ``text`` with most matches, HTML-escaped.

    The window starts a little before its densest cluster of matches, snaps to
    word boundaries and is wrapped in ellipses when text was cut off. ``marks``
    is the number of highlighted spans inside the fragment.
    """
    anchor = _best_anchor(spans, size) if spans else 0
    start = max(0, anchor - 20)
    end = min(len(text), start + size)
    inside = [s for s in spans if s[0] >= start and s[1] <= end]
    if start > 0:
        space = text.find(" ", start, anchor)
        start = space + 1 if space >= 0 else start
    if end < len(text):
        space = text.rfind(" ", max(start, inside[-1][1] if inside else start), end)
        end = space if space >= 0 else end
    inside = [s for s in inside if s[0] >= start and s[1] <= end]

    parts: list[str] = []
    cursor = start
    for s, e in inside:
        parts.append(html.escape(text[cursor:s]))
        parts.append(f"<mark>{html.escape(text[s:e])}</mark>")
        cursor = e
    parts.append(html.escape(text[cursor:end]))
    return f"{'… ' if start > 0 else ''}{''.join(parts)}{' …' if end < len(text) else ''}", len(
        inside
    )
