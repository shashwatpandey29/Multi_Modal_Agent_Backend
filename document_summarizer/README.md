Document Summarizer — enhancements

This service now supports an optional Redis-backed cache and an async precompute worker.

Environment variables:
- `DATABASE_URL` — SQLAlchemy DB URL (defaults to sqlite:///./brain.db)
- `CACHE_URL` — Redis URL (defaults to redis://localhost:6379/0)
- `CACHE_TTL_SECONDS` — TTL for cached query results (default 86400)
- `DOCSUM_ASYNC_PRECOMPUTE` — set to `true` to enqueue background precompute jobs on ingest
- `SIMILARITY_THRESHOLD` — float similarity threshold for QA fuzzy-matching (default 0.8)

Install additional deps:

```bash
pip install -r backend/document_summarizer/requirements.txt
```

Run API server (example):

```bash
uvicorn backend.document_summarizer.server:app --reload --port 8000
```

Run worker (if using async precompute):

```bash
# start an RQ worker listening to default queue
rq worker --url redis://localhost:6379/0
```

Notes:
- `/ask`, `/search`, and `/summary` now check Redis first for cached responses.
- When `DOCSUM_ASYNC_PRECOMPUTE=true`, ingestion will enqueue a `precompute_analysis` job instead of blocking.
- Make sure Redis is available when enabling async precompute or caching.
