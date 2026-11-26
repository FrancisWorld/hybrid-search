"""Text analysis: tokenisation, accent folding, stop words and stemming.

Token positions are the index of the token in the *original* stream. Stop
words are dropped but leave a gap, so a phrase such as ``"king of spain"``
keeps its exact distances at index and query time.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from hybridsearch.stemmers import porter_stem, portuguese_stem

_TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)

ENGLISH_STOPWORDS = frozenset(
    "a an and are as at be but by for if in into is it no not of on or such that the their "
    "then there these they this to was will with".split()
)
PORTUGUESE_STOPWORDS = frozenset(
    "a o as os um uma uns umas de do da dos das em no na nos nas por para com sem e ou mas "
    "que se ao aos ja foi ser sao era ele ela eles elas isso este esta esse essa".split()
)


def fold_accents(text: str) -> str:
    """Lowercase and strip diacritics (``Ação`` -> ``acao``)."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return unicodedata.normalize(
        "NFC", "".join(c for c in decomposed if not unicodedata.combining(c))
    )


@dataclass(frozen=True, slots=True)
class Token:
    term: str
    position: int
    start: int
    end: int


class Analyzer(Protocol):
    @property
    def name(self) -> str: ...

    def analyze(self, text: str) -> list[Token]: ...


@dataclass(frozen=True)
class LanguageAnalyzer:
    """Tokenise -> fold accents -> drop stop words -> stem."""

    name: str
    stopwords: frozenset[str]
    stem: Callable[[str], str]

    def analyze(self, text: str) -> list[Token]:
        tokens: list[Token] = []
        for position, match in enumerate(_TOKEN_RE.finditer(text)):
            folded = fold_accents(match.group().replace("’", "'"))
            if folded.endswith("'s"):
                folded = folded[:-2]
            folded = folded.replace("'", "")
            if not folded or folded in self.stopwords:
                continue
            tokens.append(Token(self.stem(folded), position, match.start(), match.end()))
        return tokens


_ANALYZERS: dict[str, LanguageAnalyzer] = {
    "en": LanguageAnalyzer(
        "en", frozenset(fold_accents(w) for w in ENGLISH_STOPWORDS), porter_stem
    ),
    "pt": LanguageAnalyzer(
        "pt", frozenset(fold_accents(w) for w in PORTUGUESE_STOPWORDS), portuguese_stem
    ),
}


def get_analyzer(language: str) -> LanguageAnalyzer:
    try:
        return _ANALYZERS[language]
    except KeyError:
        raise ValueError(
            f"unsupported language {language!r}; choose from {sorted(_ANALYZERS)}"
        ) from None
