from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from hybridsearch.analysis import fold_accents, get_analyzer
from hybridsearch.stemmers import porter_stem, portuguese_stem

# Reference pairs from Porter's published vocabulary / paper examples.
PORTER_CASES = {
    "caresses": "caress", "ponies": "poni", "ties": "ti", "caress": "caress", "cats": "cat",
    "feed": "feed", "agreed": "agre", "plastered": "plaster", "bled": "bled",
    "motoring": "motor", "sing": "sing", "conflated": "conflat", "troubled": "troubl",
    "sized": "size", "hopping": "hop", "tanned": "tan", "falling": "fall", "hissing": "hiss",
    "fizzed": "fizz", "failing": "fail", "filing": "file", "happy": "happi", "sky": "sky",
    "relational": "relat", "conditional": "condit", "rational": "ration", "valenci": "valenc",
    "hesitanci": "hesit", "digitizer": "digit", "conformabli": "conform", "radicalli": "radic",
    "differentli": "differ", "vileli": "vile", "analogousli": "analog",
    "vietnamization": "vietnam", "predication": "predic", "operator": "oper",
    "feudalism": "feudal", "decisiveness": "decis", "hopefulness": "hope",
    "callousness": "callous", "formaliti": "formal", "sensitiviti": "sensit",
    "sensibiliti": "sensibl", "triplicate": "triplic", "formative": "form", "formalize": "formal",
    "electriciti": "electr", "electrical": "electr", "hopeful": "hope", "goodness": "good",
    "revival": "reviv", "allowance": "allow", "inference": "infer", "airliner": "airlin",
    "adjustable": "adjust", "defensible": "defens", "irritant": "irrit",
    "replacement": "replac", "adjustment": "adjust", "dependent": "depend",
    "adoption": "adopt", "homologou": "homolog", "communism": "commun", "activate": "activ",
    "angulariti": "angular", "homologous": "homolog", "effective": "effect",
    "bowdlerize": "bowdler",
    "probate": "probat", "rate": "rate", "cease": "ceas", "controll": "control", "roll": "roll",
    "generalizations": "gener", "running": "run", "searching": "search", "searches": "search",
}  # fmt: skip


@pytest.mark.parametrize(("word", "stem"), sorted(PORTER_CASES.items()))
def test_porter_reference_cases(word: str, stem: str) -> None:
    assert porter_stem(word) == stem


@pytest.mark.parametrize(
    "family",
    [
        ["connect", "connected", "connecting", "connection", "connections"],
        ["index", "indexes", "indexing", "indexed"],
    ],
)
def test_porter_conflates_families(family: list[str]) -> None:
    assert len({porter_stem(w) for w in family}) == 1


@pytest.mark.parametrize(
    "family",
    [
        ["casa", "casas"],
        ["livro", "livros"],
        ["cidade", "cidades"],
        ["informacao", "informacoes"],
        ["corredor", "corredores", "corredora"],
        ["rapido", "rapida", "rapidos", "rapidamente"],
        ["cantar", "cantando", "cantado"],
        ["gato", "gatinho", "gatos", "gata"],
        ["homem", "homens"],
    ],
)
def test_portuguese_conflates_families(family: list[str]) -> None:
    assert len({portuguese_stem(fold_accents(w)) for w in family}) == 1, [
        portuguese_stem(fold_accents(w)) for w in family
    ]


def test_portuguese_keeps_distinct_words_apart() -> None:
    assert portuguese_stem("casa") != portuguese_stem("carro")
    assert portuguese_stem("lapis") == "lapis"


def test_fold_accents() -> None:
    assert fold_accents("Ação Coração ÀÉÎÕÜ Çedilha") == "acao coracao aeiou cedilha"


def test_analyzer_english_positions_keep_stopword_gaps() -> None:
    tokens = get_analyzer("en").analyze("The King of Spain's running")
    assert [(t.term, t.position) for t in tokens] == [("king", 1), ("spain", 3), ("run", 4)]
    assert "The King of Spain's running"[tokens[1].start : tokens[1].end] == "Spain's"


def test_analyzer_portuguese_folds_and_stems() -> None:
    en_pt = get_analyzer("pt")
    a = [t.term for t in en_pt.analyze("As informações")]
    b = [t.term for t in en_pt.analyze("a informação")]
    assert a == b
    assert [t.term for t in en_pt.analyze("AÇÃO")] == [t.term for t in en_pt.analyze("acao")]


def test_unknown_language_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported language"):
        get_analyzer("xx")


@given(st.text(max_size=200))
def test_analyzer_offsets_always_valid(text: str) -> None:
    for lang in ("en", "pt"):
        previous = -1
        for tok in get_analyzer(lang).analyze(text):
            assert 0 <= tok.start < tok.end <= len(text)
            assert tok.position > previous
            previous = tok.position
            assert tok.term
