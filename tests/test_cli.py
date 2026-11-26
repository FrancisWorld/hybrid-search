from __future__ import annotations

import json
from pathlib import Path

import pytest

from hybridsearch.cli import _parse_filter, main


def run(*argv: str) -> int:
    return main(list(argv))


def test_full_workflow(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ix = str(tmp_path / "ix")
    docs = tmp_path / "docs.jsonl"
    docs.write_text(
        "\n".join(
            json.dumps(d)
            for d in [
                {
                    "id": "a",
                    "fields": {"title": "Espresso machine", "body": "nine bars"},
                    "meta": {"category": "coffee", "year": 2021},
                },
                {
                    "id": "b",
                    "fields": {"title": "Volcano", "body": "lava and ash"},
                    "meta": {"category": "earth", "year": 2019},
                },
            ]
        )
        + "\n\n"
    )
    assert run("init", ix, "--keyword", "category", "--number", "year", "--dim", "64") == 0
    assert run("index", ix, str(docs), "--batch", "1") == 0
    assert "indexed 2 documents" in capsys.readouterr().out.splitlines()[-1]

    assert run("search", ix, "espresso", "--mode", "lexical", "--facet", "category") == 0
    out = capsys.readouterr().out
    assert "1 matches" in out
    assert "[Espresso]" in out
    assert "coffee (1)" in out

    assert run("search", ix, "lava", "--mode", "lexical", "--json", "--filter", "year>=2000") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["hits"][0]["id"] == "b"

    assert run("get", ix, "a") == 0
    assert json.loads(capsys.readouterr().out)["id"] == "a"
    assert run("get", ix, "zzz") == 1
    capsys.readouterr()

    assert run("delete", ix, "a", "nope") == 0
    assert "deleted 1 of 2" in capsys.readouterr().out
    assert run("stats", ix) == 0
    assert json.loads(capsys.readouterr().out)["documents"] == 1
    assert run("merge", ix, "--force") == 0
    assert "merges" in capsys.readouterr().out


def test_errors_are_reported_with_exit_code_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ix = str(tmp_path / "ix")
    assert run("init", ix, "--dim", "64") == 0
    capsys.readouterr()
    assert run("search", ix, '"broken', "--mode", "lexical") == 2
    assert "query_syntax_error" in capsys.readouterr().err
    assert run("search", ix, "x", "--filter", "garbage") == 2
    assert "cannot parse filter" in capsys.readouterr().err
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n")
    assert run("index", ix, str(bad)) == 2
    assert "invalid JSON" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        run("nonsense")


def test_sample_and_eval(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ix = str(tmp_path / "ix")
    assert run("sample", ix, "--dataset", "pt") == 0
    assert "loaded 25" in capsys.readouterr().out
    assert run("search", ix, "eclipse do sol", "--mode", "lexical", "-k", "1") == 0
    assert "astronomia-eclipse" in capsys.readouterr().out
    assert run("eval", "--dataset", "en") == 0
    out = capsys.readouterr().out
    assert "documents=38" in out
    assert "hybrid" in out


def test_filter_parsing() -> None:
    assert _parse_filter(["category=coffee", "year>=2020", "year<2024", "score<=1.5"]) == {
        "category": {"eq": "coffee"},
        "year": {"gte": 2020, "lt": 2024},
        "score": {"lte": 1.5},
    }
