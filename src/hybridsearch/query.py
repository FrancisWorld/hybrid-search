"""Query language parser and analysis.

Syntax (a small Lucene-like subset)::

    espresso machine          either word may match (scores add up)
    +espresso -decaf          required / prohibited
    espresso AND machine      both required; AND binds tighter than adjacency/OR
    "pressure profile"        exact phrase (stop words keep their gaps)
    title:espresso            restrict to one field
    title:(espresso OR moka)  restrict a group
    NOT decaf                 prohibited (same as -decaf)

A query made only of prohibited clauses is rejected: there is nothing to rank.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from hybridsearch.analysis import Analyzer
from hybridsearch.errors import QueryError

MAX_DEPTH = 16
MAX_CLAUSES = 256
MAX_QUERY_CHARS = 2048
_FIELD_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*):")


@dataclass(frozen=True)
class Text:
    """Unanalysed word or phrase as typed by the user."""

    text: str
    phrase: bool = False
    field: str | None = None


@dataclass(frozen=True)
class Bool:
    must: tuple[Node, ...] = ()
    should: tuple[Node, ...] = ()
    must_not: tuple[Node, ...] = ()


@dataclass(frozen=True)
class TermNode:
    field: str | None
    term: str


@dataclass(frozen=True)
class PhraseNode:
    field: str | None
    terms: tuple[str, ...]
    offsets: tuple[int, ...]


Node = Text | Bool | TermNode | PhraseNode


class _Kind(Enum):
    LPAREN = "("
    RPAREN = ")"
    PHRASE = "phrase"
    WORD = "word"
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    PLUS = "+"
    MINUS = "-"
    FIELD = "field"
    END = "end"


@dataclass(frozen=True)
class _Tok:
    kind: _Kind
    value: str
    pos: int


def _lex(query: str) -> list[_Tok]:
    toks: list[_Tok] = []
    i, n = 0, len(query)
    while i < n:
        ch = query[i]
        if ch.isspace():
            i += 1
        elif ch in "()":
            toks.append(_Tok(_Kind(ch), ch, i))
            i += 1
        elif ch == '"':
            end = query.find('"', i + 1)
            if end < 0:
                raise QueryError("unterminated phrase", i)
            toks.append(_Tok(_Kind.PHRASE, query[i + 1 : end], i))
            i = end + 1
        elif ch in "+-" and i + 1 < n and not query[i + 1].isspace():
            toks.append(_Tok(_Kind(ch), ch, i))
            i += 1
        else:
            m = _FIELD_RE.match(query, i)
            if m and m.end() < n and not query[m.end()].isspace():
                toks.append(_Tok(_Kind.FIELD, m.group(1), i))
                i = m.end()
                continue
            j = i
            while j < n and not query[j].isspace() and query[j] not in '()"':
                j += 1
            word = query[i:j]
            kind = _Kind(word) if word in ("AND", "OR", "NOT") else _Kind.WORD
            toks.append(_Tok(kind, word, i))
            i = j
    toks.append(_Tok(_Kind.END, "", n))
    return toks


_MUST, _SHOULD, _MUST_NOT = "must", "should", "must_not"


class _Parser:
    def __init__(self, query: str) -> None:
        self.toks = _lex(query)
        self.i = 0
        self.clauses = 0

    def peek(self) -> _Tok:
        return self.toks[self.i]

    def take(self) -> _Tok:
        tok = self.toks[self.i]
        self.i += 1
        return tok

    def parse(self) -> Node:
        node = self.parse_clauses(depth=0)
        if self.peek().kind is not _Kind.END:
            raise QueryError("unbalanced ')'", self.peek().pos)
        return node

    def parse_clauses(self, depth: int) -> Node:
        if depth > MAX_DEPTH:
            raise QueryError("query nested too deeply")
        items: list[tuple[str, Node]] = []
        while self.peek().kind not in (_Kind.END, _Kind.RPAREN):
            if self.peek().kind is _Kind.OR:
                self.take()
                continue
            items.append(self.parse_and_chain(depth))
        return self._build(items)

    def parse_and_chain(self, depth: int) -> tuple[str, Node]:
        first = self.parse_unary(depth)
        if self.peek().kind is not _Kind.AND:
            return first
        chain = [first]
        while self.peek().kind is _Kind.AND:
            self.take()
            chain.append(self.parse_unary(depth))
        return _SHOULD, self._build([(o if o == _MUST_NOT else _MUST, n) for o, n in chain])

    def parse_unary(self, depth: int) -> tuple[str, Node]:
        tok = self.take()
        occur = _SHOULD
        if tok.kind in (_Kind.PLUS, _Kind.MINUS, _Kind.NOT):
            occur = {_Kind.PLUS: _MUST}.get(tok.kind, _MUST_NOT)
            tok = self.take()
        return occur, self.parse_atom(tok, depth, None)

    def parse_atom(self, tok: _Tok, depth: int, field: str | None) -> Node:
        self.clauses += 1
        if self.clauses > MAX_CLAUSES:
            raise QueryError(f"query has more than {MAX_CLAUSES} clauses")
        if tok.kind is _Kind.WORD:
            return Text(tok.value, False, field)
        if tok.kind is _Kind.PHRASE:
            return Text(tok.value, True, field)
        if tok.kind is _Kind.FIELD:
            return self.parse_atom(self.take(), depth, tok.value)
        if tok.kind is _Kind.LPAREN:
            inner = self.parse_clauses(depth + 1)
            if self.peek().kind is not _Kind.RPAREN:
                raise QueryError("missing ')'", tok.pos)
            self.take()
            return _with_field(inner, field) if field else inner
        raise QueryError(f"unexpected {tok.kind.value!r}", tok.pos)

    @staticmethod
    def _build(items: list[tuple[str, Node]]) -> Node:
        if not items:
            raise QueryError("empty query")
        must = tuple(n for o, n in items if o == _MUST)
        should = tuple(n for o, n in items if o == _SHOULD)
        must_not = tuple(n for o, n in items if o == _MUST_NOT)
        if not must and not should:
            raise QueryError("query has only prohibited clauses")
        if len(items) == 1:
            return items[0][1]
        return Bool(must, should, must_not)


def _with_field(node: Node, field: str) -> Node:
    if isinstance(node, Text):
        return node if node.field else Text(node.text, node.phrase, field)
    if isinstance(node, Bool):
        return Bool(
            tuple(_with_field(n, field) for n in node.must),
            tuple(_with_field(n, field) for n in node.should),
            tuple(_with_field(n, field) for n in node.must_not),
        )
    return node


def parse_query(query: str) -> Node:
    if len(query) > MAX_QUERY_CHARS:
        raise QueryError(f"query longer than {MAX_QUERY_CHARS} characters")
    if not query.strip():
        raise QueryError("empty query")
    return _Parser(query).parse()


def _analyze_text(node: Text, analyzer: Analyzer) -> Node | None:
    tokens = analyzer.analyze(node.text)
    if not tokens:
        return None
    if len(tokens) == 1 and not node.phrase:
        return TermNode(node.field, tokens[0].term)
    base = tokens[0].position
    return PhraseNode(
        node.field,
        tuple(t.term for t in tokens),
        tuple(t.position - base for t in tokens),
    )


def analyze_query(node: Node, analyzer: Analyzer) -> Node | None:
    """Replace raw text by analysed term/phrase nodes; prune what analysis removed.

    Words that vanish (stop words, punctuation) are dropped. A word that
    analyses to several tokens (``state-of-the-art``) becomes a phrase.
    """
    if isinstance(node, Text):
        return _analyze_text(node, analyzer)
    if not isinstance(node, Bool):
        return node
    must = _prune(node.must, analyzer)
    should = _prune(node.should, analyzer)
    must_not = _prune(node.must_not, analyzer)
    if not must and not should:
        return None
    if not must_not and len(must) + len(should) == 1:
        return (must or should)[0]
    return Bool(must, should, must_not)


def _prune(nodes: tuple[Node, ...], analyzer: Analyzer) -> tuple[Node, ...]:
    out = (analyze_query(n, analyzer) for n in nodes)
    return tuple(n for n in out if n is not None)
