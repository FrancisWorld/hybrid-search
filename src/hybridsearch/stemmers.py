"""Stemmers: a faithful Porter (1980) stemmer for English and a compact
suffix-stripping stemmer for Brazilian Portuguese.

The Portuguese stemmer works on accent-folded input (the analyzer folds first)
and follows the structure of RSLP (plural, feminine, augmentative/diminutive,
adverb, noun, verb and vowel steps) with a deliberately smaller rule table.
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------- English


def _is_consonant(word: str, i: int) -> bool:
    ch = word[i]
    if ch in "aeiou":
        return False
    if ch == "y":
        return i == 0 or not _is_consonant(word, i - 1)
    return True


def _measure(stem: str) -> int:
    """Porter's m: the number of VC sequences in ``[C](VC){m}[V]``."""
    pattern = "".join("c" if _is_consonant(stem, i) else "v" for i in range(len(stem)))
    return len(re.findall(r"vc", pattern))


def _has_vowel(stem: str) -> bool:
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def _ends_double_consonant(word: str) -> bool:
    return len(word) >= 2 and word[-1] == word[-2] and _is_consonant(word, len(word) - 1)


def _ends_cvc(word: str) -> bool:
    n = len(word)
    return (
        n >= 3
        and _is_consonant(word, n - 3)
        and not _is_consonant(word, n - 2)
        and _is_consonant(word, n - 1)
        and word[-1] not in "wxy"
    )


_STEP2 = {
    "ational": "ate", "tional": "tion", "enci": "ence", "anci": "ance", "izer": "ize",
    "bli": "ble", "alli": "al", "entli": "ent", "eli": "e", "ousli": "ous",
    "ization": "ize", "ation": "ate", "ator": "ate", "alism": "al", "iveness": "ive",
    "fulness": "ful", "ousness": "ous", "aliti": "al", "iviti": "ive", "biliti": "ble",
    "logi": "log",
}  # fmt: skip
_STEP3 = {
    "icate": "ic", "ative": "", "alize": "al", "iciti": "ic", "ical": "ic",
    "ful": "", "ness": "",
}  # fmt: skip
_STEP4 = (
    "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent",
    "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
)  # fmt: skip


def _replace_suffix(word: str, table: dict[str, str]) -> str:
    for suffix in sorted(table, key=len, reverse=True):
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            return stem + table[suffix] if _measure(stem) > 0 else word
    return word


def porter_stem(word: str) -> str:
    """Stem a lowercase English word with the original Porter algorithm."""
    if len(word) <= 2:
        return word
    # Step 1a
    if word.endswith(("sses", "ies")):
        word = word[:-2]
    elif word.endswith("ss"):
        pass
    elif word.endswith("s"):
        word = word[:-1]
    # Step 1b
    flag = False
    if word.endswith("eed"):
        if _measure(word[:-3]) > 0:
            word = word[:-1]
    elif word.endswith("ed") and _has_vowel(word[:-2]):
        word, flag = word[:-2], True
    elif word.endswith("ing") and _has_vowel(word[:-3]):
        word, flag = word[:-3], True
    if flag:
        if word.endswith(("at", "bl", "iz")):
            word += "e"
        elif _ends_double_consonant(word) and word[-1] not in "lsz":
            word = word[:-1]
        elif _measure(word) == 1 and _ends_cvc(word):
            word += "e"
    # Step 1c
    if word.endswith("y") and _has_vowel(word[:-1]):
        word = word[:-1] + "i"
    # Steps 2 and 3
    word = _replace_suffix(word, _STEP2)
    word = _replace_suffix(word, _STEP3)
    # Step 4
    for suffix in sorted(_STEP4, key=len, reverse=True):
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            if _measure(stem) > 1 and (suffix != "ion" or stem.endswith(("s", "t"))):
                word = stem
            break
    # Step 5
    if word.endswith("e"):
        stem = word[:-1]
        m = _measure(stem)
        if m > 1 or (m == 1 and not _ends_cvc(stem)):
            word = stem
    if _measure(word) > 1 and _ends_double_consonant(word) and word.endswith("l"):
        word = word[:-1]
    return word


# ------------------------------------------------------------------------ Portuguese

# (suffix, replacement, minimum stem length); longest suffix wins inside a step.
_PT_PLURAL = (
    ("oes", "ao", 1), ("aes", "ao", 1), ("ais", "al", 1), ("eis", "el", 2),
    ("ois", "ol", 2), ("res", "r", 2), ("ns", "m", 1), ("is", "il", 2), ("s", "", 2),
)  # fmt: skip
_PT_FEMININE = (("ona", "ao", 3), ("ora", "or", 3), ("esa", "es", 3), ("a", "o", 3))
_PT_AUGMENTATIVE = (
    ("zinho", "", 3), ("zinha", "", 3), ("inho", "", 3), ("inha", "", 3),
    ("issimo", "", 3), ("issima", "", 3), ("zao", "", 3), ("ao", "", 4),
)  # fmt: skip
_PT_ADVERB = (("mente", "", 4),)
_PT_NOUN = (
    ("amentos", "", 4), ("amento", "", 4), ("imentos", "", 4), ("imento", "", 4),
    ("adoras", "", 3), ("adores", "", 3), ("adora", "", 3), ("ador", "", 3),
    ("acoes", "", 3), ("acao", "", 3), ("icacao", "", 3), ("idade", "", 3), ("idades", "", 3),
    ("ezas", "", 3), ("eza", "", 3), ("ismo", "", 3), ("ista", "", 3), ("istas", "", 3),
    ("avel", "", 3), ("ivel", "", 3), ("osos", "", 3), ("osa", "", 3), ("oso", "", 3),
    ("ico", "", 3), ("ica", "", 3), ("ncia", "", 3), ("ante", "", 3), ("encia", "", 3),
)  # fmt: skip
_PT_VERB = (
    ("ariamos", "", 2), ("ariam", "", 2), ("aremos", "", 2), ("assem", "", 2),
    ("ando", "", 2), ("endo", "", 2), ("indo", "", 2), ("armos", "", 2),
    ("aram", "", 2), ("avam", "", 2), ("ava", "", 2), ("ara", "", 2), ("amos", "", 2),
    ("ado", "", 2), ("ido", "", 2), ("ada", "", 2), ("ida", "", 2), ("ar", "", 2),
    ("er", "", 2), ("ir", "", 2), ("am", "", 2), ("em", "", 2), ("ou", "", 2),
    ("ei", "", 2), ("ia", "", 2),
)  # fmt: skip
_PT_VOWEL = (("a", "", 3), ("e", "", 3), ("o", "", 3))

_PT_PLURAL_EXCEPTIONS = frozenset({"lapis", "pires", "onibus", "campus", "bonus", "gas", "pais"})


def _apply_step(word: str, rules: tuple[tuple[str, str, int], ...]) -> tuple[str, bool]:
    for suffix, repl, min_stem in sorted(rules, key=lambda r: len(r[0]), reverse=True):
        if word.endswith(suffix) and len(word) - len(suffix) >= min_stem:
            return word[: len(word) - len(suffix)] + repl, True
    return word, False


def portuguese_stem(word: str) -> str:
    """Stem an accent-folded, lowercase Portuguese word."""
    if len(word) <= 3 or word in _PT_PLURAL_EXCEPTIONS:
        return word
    word, _ = _apply_step(word, _PT_PLURAL)
    word, _ = _apply_step(word, _PT_FEMININE)
    word, _ = _apply_step(word, _PT_ADVERB)
    word, changed = _apply_step(word, _PT_AUGMENTATIVE)
    if not changed:
        word, changed = _apply_step(word, _PT_NOUN)
    if not changed:
        word, changed = _apply_step(word, _PT_VERB)
    if not changed:
        word, _ = _apply_step(word, _PT_VOWEL)
    return word
