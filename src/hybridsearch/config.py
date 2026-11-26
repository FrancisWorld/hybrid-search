"""Runtime settings read from environment variables (see ``.env.example``)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from hybridsearch.index import IndexOptions
from hybridsearch.schema import Schema


def _csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


@dataclass(frozen=True)
class Settings:
    index_dir: Path = Path("./data/index")
    fsync: bool = True
    language: str = "en"
    keyword_fields: tuple[str, ...] = ("category", "tags")
    number_fields: tuple[str, ...] = ("year",)
    embedding_dim: int = 256

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        defaults = cls()
        return cls(
            index_dir=Path(env.get("HYBRID_SEARCH_DIR", str(defaults.index_dir))),
            fsync=env.get("HYBRID_SEARCH_FSYNC", "1") not in ("0", "false", "no"),
            language=env.get("HYBRID_SEARCH_LANGUAGE", defaults.language),
            keyword_fields=_csv(
                env.get("HYBRID_SEARCH_KEYWORDS", ",".join(defaults.keyword_fields))
            ),
            number_fields=_csv(env.get("HYBRID_SEARCH_NUMBERS", ",".join(defaults.number_fields))),
            embedding_dim=int(env.get("HYBRID_SEARCH_DIM", str(defaults.embedding_dim))),
        )

    def schema(self) -> Schema:
        """Schema used only when the index directory does not exist yet."""
        return Schema(
            language=self.language,
            keyword_fields=self.keyword_fields,
            number_fields=self.number_fields,
            embedding_dim=self.embedding_dim,
        )

    def index_options(self) -> IndexOptions:
        return IndexOptions(fsync=self.fsync)
