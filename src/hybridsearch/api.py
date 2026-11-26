"""FastAPI application: JSON API plus a small server-rendered search page."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field

from hybridsearch.config import Settings
from hybridsearch.errors import (
    CorruptIndexError,
    IndexClosedError,
    IndexLockedError,
    QueryError,
    SearchError,
    ValidationError,
)
from hybridsearch.index import Index
from hybridsearch.search import Mode, SearchRequest, SearchResult, result_to_dict

log = logging.getLogger("hybridsearch.api")
TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
MAX_BULK = 1000

_STATUS: dict[type[SearchError], int] = {
    ValidationError: 400,
    QueryError: 400,
    IndexLockedError: 409,
    IndexClosedError: 503,
    CorruptIndexError: 500,
}


class DocumentIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    fields: dict[str, str]
    meta: dict[str, Any] = Field(default_factory=dict)


class BulkIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    documents: list[DocumentIn] = Field(min_length=1, max_length=MAX_BULK)


class SearchIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    mode: Mode = "hybrid"
    k: int = 10
    offset: int = 0
    filters: dict[str, Any] | None = None
    facets: list[str] = Field(default_factory=list)
    facet_size: int = 10
    highlight: bool = True
    candidates: int = 100
    rrf_k: int = 60
    min_similarity: float = 0.1

    def to_request(self) -> SearchRequest:
        return SearchRequest(**self.model_dump())


def create_app(index: Index | None = None, settings: Settings | None = None) -> FastAPI:
    """Build the app. Without ``index`` one is opened from ``settings`` on startup."""
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = index is None
        app.state.index = index or Index.open(
            settings.index_dir, settings.schema(), options=settings.index_options()
        )
        log.info("index ready: %s", app.state.index.stats()["documents"])
        try:
            yield
        finally:
            if owned:  # graceful shutdown: flush the memtable and release the lock
                app.state.index.close()
                log.info("index closed")

    app = FastAPI(title="hybrid-search", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(SearchError)
    async def search_error_handler(_: Request, exc: SearchError) -> JSONResponse:
        status = next((s for t, s in _STATUS.items() if isinstance(exc, t)), 500)
        return JSONResponse({"error": exc.to_dict()}, status_code=status)

    def ix(request: Request) -> Index:
        index_: Index = request.app.state.index
        return index_

    @app.get("/healthz")
    def healthz(request: Request) -> dict[str, Any]:
        return {"status": "ok", "documents": ix(request).count()}

    @app.get("/stats")
    def stats(request: Request) -> dict[str, Any]:
        return ix(request).stats()

    @app.post("/documents")
    def put_documents(body: BulkIn, request: Request) -> dict[str, int]:
        return {"indexed": ix(request).put_many(d.model_dump() for d in body.documents)}

    @app.get("/documents/{doc_id}")
    def get_document(doc_id: str, request: Request) -> dict[str, Any]:
        doc = ix(request).get(doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail=f"document {doc_id!r} not found")
        return doc

    @app.delete("/documents/{doc_id}")
    def delete_document(doc_id: str, request: Request) -> dict[str, bool]:
        if not ix(request).delete(doc_id):
            raise HTTPException(status_code=404, detail=f"document {doc_id!r} not found")
        return {"deleted": True}

    @app.post("/search")
    def search(body: SearchIn, request: Request) -> dict[str, Any]:
        return result_to_dict(ix(request).search(body.to_request()))

    @app.post("/flush")
    def flush(request: Request) -> dict[str, Any]:
        ix(request).flush()
        return ix(request).stats()

    @app.post("/merge")
    def merge(request: Request, force: bool = False) -> dict[str, Any]:
        merges = ix(request).merge(force=force)
        return {"merges": merges, **ix(request).stats()}

    @app.get("/", response_class=HTMLResponse)
    def home(
        request: Request,
        q: Annotated[str, Query(max_length=500)] = "",
        mode: Mode = "hybrid",
        category: str = "",
    ) -> HTMLResponse:
        index_ = ix(request)
        result: SearchResult | None = None
        error: str | None = None
        facets = list(index_.schema.keyword_fields)[:1]
        if q.strip():
            filters = {facets[0]: category} if category and facets else None
            try:
                result = index_.search(
                    SearchRequest(q, mode=mode, k=10, facets=facets, filters=filters)
                )
            except (QueryError, ValidationError) as exc:
                error = str(exc)
        return TEMPLATES.TemplateResponse(
            request,
            "index.html",
            {
                "q": q,
                "mode": mode,
                "category": category,
                "result": result,
                "error": error,
                "facet_field": facets[0] if facets else None,
                "documents": index_.count(),
            },
        )

    return app


def app_factory() -> FastAPI:
    """Entry point for ``uvicorn --factory``; configuration comes from the environment."""
    return create_app()
