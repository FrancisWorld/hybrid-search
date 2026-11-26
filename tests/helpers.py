"""Shared test helpers."""

from __future__ import annotations

import random

from hybridsearch.analysis import LanguageAnalyzer
from hybridsearch.embedder import HashingEmbedder
from hybridsearch.schema import Document, Schema, parse_document
from hybridsearch.segment import Segment, SegmentOptions, analyze_documents, serialize_segment

VOCAB = (
    "espresso machine pressure grinder bean roast kettle pour filter water boiler steam "
    "volcano lava crater magma ash eruption island ocean reef marathon training pace "
    "solar panel battery inverter grid sourdough starter flour oven crust yeast"
).split()


def random_docs(schema: Schema, n: int, seed: int = 0, prefix: str = "d") -> list[Document]:
    rng = random.Random(seed)
    docs = []
    for i in range(n):
        raw = {
            "id": f"{prefix}{i}",
            "fields": {
                "title": " ".join(rng.choices(VOCAB, k=rng.randint(1, 4))),
                "body": " ".join(rng.choices(VOCAB, k=rng.randint(5, 30))),
            },
            "meta": {
                "category": rng.choice(["coffee", "earth", "sport", "energy"]),
                "tags": rng.sample(["a", "b", "c", "d"], k=rng.randint(0, 3)),
                "year": rng.randint(2015, 2025),
            },
        }
        docs.append(parse_document(raw, schema))
    return docs


def build_segment(
    docs: list[Document],
    schema: Schema,
    analyzer: LanguageAnalyzer,
    embedder: HashingEmbedder,
    options: SegmentOptions | None = None,
) -> Segment:
    data = analyze_documents(docs, embedder.embed([d.text for d in docs]), schema, analyzer)
    return Segment(serialize_segment(data, schema, options or SegmentOptions()))
