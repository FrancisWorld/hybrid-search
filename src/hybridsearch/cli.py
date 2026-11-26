"""Command line interface: ``hybrid-search --help``."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import logging
import sys
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import uvicorn

from hybridsearch.api import create_app
from hybridsearch.config import Settings
from hybridsearch.errors import SearchError
from hybridsearch.evaluation import (
    DATASETS,
    compare_modes,
    format_table,
    load_dataset,
    load_into,
)
from hybridsearch.index import Index
from hybridsearch.schema import Schema
from hybridsearch.search import SearchRequest, result_to_dict


def _parse_filter(items: Sequence[str]) -> dict[str, Any]:
    """``field=value`` (equality), ``field>=3``, ``field<3`` and so on."""
    filters: dict[str, Any] = {}
    ops = (("<=", "lte"), (">=", "gte"), ("<", "lt"), (">", "gt"), ("=", "eq"))
    for item in items:
        for token, op in ops:
            field, sep, raw = item.partition(token)
            if sep:
                value: Any = raw
                with contextlib.suppress(ValueError):
                    value = float(raw) if "." in raw else int(raw)
                filters.setdefault(field, {})[op] = value
                break
        else:
            raise SearchError(f"cannot parse filter {item!r}; use field=value or field>=number")
    return filters


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SearchError(f"{path}:{number}: invalid JSON ({exc.msg})") from exc


def _cmd_init(args: argparse.Namespace) -> int:
    schema = Schema(
        language=args.language,
        keyword_fields=tuple(args.keyword),
        number_fields=tuple(args.number),
        embedding_dim=args.dim,
    )
    with Index.open(args.dir, schema):
        pass
    print(f"initialised {args.dir} ({args.language})")
    return 0


def _cmd_index(args: argparse.Namespace) -> int:
    total, batch = 0, []
    with Index.open(args.dir) as index:
        for doc in _read_jsonl(args.file):
            batch.append(doc)
            if len(batch) >= args.batch:
                total += index.put_many(batch)
                batch = []
        total += index.put_many(batch)
        print(f"indexed {total} documents; {index.count()} in index")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    request = SearchRequest(
        args.query,
        mode=args.mode,
        k=args.k,
        filters=_parse_filter(args.filter) or None,
        facets=args.facet,
        highlight=True,
        candidates=max(100, args.k),
    )
    with Index.open(args.dir) as index:
        result = index.search(request)
    if args.json:
        print(json.dumps(result_to_dict(result), indent=2, ensure_ascii=False))
        return 0
    print(f"{result.total} matches ({result.mode}, {result.took_ms:.1f} ms)")
    for rank, hit in enumerate(result.hits, start=1):
        ranks = f"lex#{hit.lexical_rank or '-'} vec#{hit.vector_rank or '-'}"
        title = hit.document["fields"].get("title", "")
        print(f"{rank:>2}. {hit.id}  {hit.score:.4f}  [{ranks}]  {title}")
        print(f"    {hit.snippet.replace('<mark>', '[').replace('</mark>', ']')}")
    for name, buckets in result.facets.items():
        print(f"{name}: " + ", ".join(f"{b['value']} ({b['count']})" for b in buckets))
    return 0


def _cmd_get(args: argparse.Namespace) -> int:
    with Index.open(args.dir) as index:
        doc = index.get(args.id)
    if doc is None:
        print(f"document {args.id!r} not found", file=sys.stderr)
        return 1
    print(json.dumps(doc, indent=2, ensure_ascii=False))
    return 0


def _cmd_delete(args: argparse.Namespace) -> int:
    with Index.open(args.dir) as index:
        removed = index.delete_many(args.ids)
    print(f"deleted {removed} of {len(args.ids)}")
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    with Index.open(args.dir) as index:
        print(json.dumps(index.stats(), indent=2))
    return 0


def _cmd_merge(args: argparse.Namespace) -> int:
    with Index.open(args.dir) as index:
        index.flush()
        merges = index.merge(force=args.force)
        print(f"{merges} merges; segments: {len(index.stats()['segments'])}")
    return 0


def _cmd_sample(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset)
    with load_into(args.dir, dataset) as index:
        print(f"loaded {index.count()} sample documents ({args.dataset}) into {args.dir}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset)
    with (
        tempfile.TemporaryDirectory(prefix="hybrid-search-eval-") as tmp,
        load_into(Path(tmp) / "ix", dataset) as index,
    ):
        results = compare_modes(index, dataset.queries, k=args.k)
    print(
        f"dataset={args.dataset} documents={len(dataset.documents)} queries={len(dataset.queries)}"
    )
    print(format_table(results, args.k))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    settings = dataclasses.replace(Settings.from_env(), index_dir=args.dir)
    uvicorn.run(create_app(settings=settings), host=args.host, port=args.port, log_level="info")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hybrid-search", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(
        name: str, handler: Any, help_: str, *, directory: bool = True
    ) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_)
        if directory:
            p.add_argument("dir", type=Path, help="index directory")
        p.set_defaults(handler=handler)
        return p

    p = add("init", _cmd_init, "create an empty index")
    p.add_argument("--language", choices=["en", "pt"], default="en")
    p.add_argument("--keyword", action="append", default=[], help="keyword (facet/filter) field")
    p.add_argument("--number", action="append", default=[], help="numeric (range filter) field")
    p.add_argument("--dim", type=int, default=256, help="embedding dimension")

    p = add("index", _cmd_index, "index documents from a JSONL file")
    p.add_argument("file", type=Path)
    p.add_argument("--batch", type=int, default=500)

    p = add("search", _cmd_search, "run a query")
    p.add_argument("query")
    p.add_argument("--mode", choices=["lexical", "vector", "hybrid"], default="hybrid")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--filter", action="append", default=[], help="field=value or field>=number")
    p.add_argument("--facet", action="append", default=[])
    p.add_argument("--json", action="store_true")

    p = add("get", _cmd_get, "print one document")
    p.add_argument("id")
    p = add("delete", _cmd_delete, "delete documents by id")
    p.add_argument("ids", nargs="+")
    add("stats", _cmd_stats, "show index statistics")
    p = add("merge", _cmd_merge, "flush the memtable and run the merge policy")
    p.add_argument("--force", action="store_true", help="merge every segment into one")

    p = add("sample", _cmd_sample, "load a bundled sample dataset (creates the index)")
    p.add_argument("--dataset", choices=DATASETS, default="en")

    p = add(
        "eval",
        _cmd_eval,
        "nDCG/MRR of lexical, vector and hybrid on a bundled dataset",
        directory=False,
    )
    p.add_argument("--dataset", choices=DATASETS, default="en")
    p.add_argument("-k", type=int, default=10)

    p = add("serve", _cmd_serve, "run the HTTP API and demo page")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        code: int = args.handler(args)
    except SearchError as exc:
        print(f"error [{exc.code}]: {exc}", file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":
    sys.exit(main())
