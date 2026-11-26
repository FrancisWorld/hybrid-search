from __future__ import annotations

import pytest

from hybridsearch.analysis import LanguageAnalyzer, get_analyzer
from hybridsearch.embedder import HashingEmbedder
from hybridsearch.schema import Schema


@pytest.fixture
def schema() -> Schema:
    return Schema(
        language="en",
        text_fields={"title": 2.0, "body": 1.0},
        keyword_fields=("category", "tags"),
        number_fields=("year",),
        embedding_dim=64,
    )


@pytest.fixture
def analyzer() -> LanguageAnalyzer:
    return get_analyzer("en")


@pytest.fixture
def embedder(analyzer: LanguageAnalyzer) -> HashingEmbedder:
    return HashingEmbedder(analyzer, dim=64)
