"""Measure HNSW recall against brute force, indexing throughput and query latency.

Usage: ``uv run python benchmarks/measure.py``. Numbers depend on the machine; the
README quotes the values from the machine class named there.
"""

from __future__ import annotations

import random
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

from hybridsearch.hnsw import HNSWIndex, brute_force
from hybridsearch.index import Index, IndexOptions
from hybridsearch.schema import Schema
from hybridsearch.search import SearchRequest

WORDS = [f"term{i}" for i in range(3000)]


def unit(n: int, dim: int, seed: int) -> np.ndarray:
    data = np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float32)
    return (data / np.linalg.norm(data, axis=1, keepdims=True)).astype(np.float32)


def hnsw_recall(n: int = 5000, dim: int = 64, k: int = 10) -> None:
    vectors, queries = unit(n, dim, 1), unit(100, dim, 2)
    started = time.perf_counter()
    graph = HNSWIndex.build(vectors, m=16, ef_construction=100, seed=0)
    build = time.perf_counter() - started
    print(f"HNSW n={n} dim={dim} M=16 efC=100: build {build:.1f}s ({n / build:.0f} vectors/s)")
    for ef in (16, 32, 64, 128):
        hits = 0
        started = time.perf_counter()
        for q in queries:
            got, _ = graph.search(q, k, ef=ef)
            hits += len(set(got.tolist()) & set(brute_force(vectors, q, k)[0].tolist()))
        per_query = (time.perf_counter() - started) / len(queries) * 1000
        print(f"  ef={ef:<4} recall@{k}={hits / (k * len(queries)):.3f}  {per_query:.2f} ms/query")
    started = time.perf_counter()
    for q in queries:
        brute_force(vectors, q, k)
    print(f"  brute force: {(time.perf_counter() - started) / len(queries) * 1000:.2f} ms/query")


def engine(n_docs: int = 5000) -> None:
    rng = random.Random(0)
    schema = Schema(keyword_fields=("category",), number_fields=("year",), embedding_dim=128)
    docs = [
        {
            "id": f"d{i}",
            "fields": {
                "title": " ".join(rng.choices(WORDS, k=4)),
                "body": " ".join(rng.choices(WORDS, k=60)),
            },
            "meta": {"category": f"c{i % 8}", "year": 2000 + i % 25},
        }
        for i in range(n_docs)
    ]
    with tempfile.TemporaryDirectory() as tmp:
        options = IndexOptions(fsync=False, memtable_max_docs=1000)
        with Index.open(Path(tmp) / "ix", schema, options=options) as index:
            started = time.perf_counter()
            for start in range(0, n_docs, 250):
                index.put_many(docs[start : start + 250])
            index.flush()
            took = time.perf_counter() - started
            print(
                f"engine: indexed {n_docs} docs (60-word bodies) in {took:.1f}s "
                f"= {n_docs / took:.0f} docs/s"
            )
            print(f"  segments after auto-merge: {len(index.stats()['segments'])}")
            for mode in ("lexical", "vector", "hybrid"):
                times = []
                for _ in range(50):
                    query = " ".join(rng.choices(WORDS, k=3))
                    started = time.perf_counter()
                    index.search(SearchRequest(query, mode=mode, k=10))  # type: ignore[arg-type]
                    times.append((time.perf_counter() - started) * 1000)
                times.sort()
                print(
                    f"  {mode:<8} p50={statistics.median(times):.1f} ms  "
                    f"p95={times[int(len(times) * 0.95)]:.1f} ms"
                )


if __name__ == "__main__":
    hnsw_recall()
    engine()
