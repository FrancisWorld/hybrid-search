# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm AS build
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 \
    HYBRID_SEARCH_DIR=/data/index
RUN useradd --system --uid 10001 --home-dir /nonexistent --shell /usr/sbin/nologin search \
    && mkdir -p /data && chown search /data
COPY --from=build /install /usr/local
USER search
VOLUME /data
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"]
# One worker: the index directory is single-writer (flock), so scale reads with replicas of the data, not workers.
CMD ["uvicorn", "hybridsearch.api:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000"]
