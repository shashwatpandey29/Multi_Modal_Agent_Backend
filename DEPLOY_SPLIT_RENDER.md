# Split Deploy on Render (Fast Setup)

Deploy two Render Web Services from the same repository.

## 1) Core Backend Service

- Name: `multi-core-backend`
- Root Directory: `backend`
- Build Command:

```bash
pip install -r requirements.txt
```

- Start Command:

```bash
uvicorn server:app --host 0.0.0.0 --port $PORT
```

- Required environment variables:

```env
NVIDIA_API_KEY=...
NVIDIA_API_BASE_URL=https://integrate.api.nvidia.com/v1
OPENAI_CHAT_MODEL=...
OPENAI_CODER_MODEL=...
OPENAI_ANALYSIS_MODEL=...
OPENAI_EMBEDDING_MODEL=nvidia/nv-embed-v1
DOCSUM_API_BASE_URL=https://<your-docsum-service>.onrender.com
DOCSUM_INTERNAL_TOKEN=<same-secret-as-docsum-service>
DOCSUM_TIMEOUT_SEC=180
DOCSUM_PROXY_ONLY=true
```

## 2) Document Summarizer Service

- Name: `multi-docsum-backend`
- Root Directory: `backend`
- Build Command:

```bash
pip install -r document_summarizer/requirements.txt
```

- Start Command:

```bash
uvicorn document_summarizer.server:app --host 0.0.0.0 --port $PORT
```

- Required environment variables:

```env
NVIDIA_API_KEY=...
NVIDIA_API_BASE_URL=https://integrate.api.nvidia.com/v1
OPENAI_CHAT_MODEL=...
OPENAI_CODER_MODEL=...
OPENAI_ANALYSIS_MODEL=...
OPENAI_EMBEDDING_MODEL=nvidia/nv-embed-v1
DOCSUM_INTERNAL_TOKEN=<same-secret-as-core-service>
EMBEDDING_PROVIDER=openai
OPENAI_EMBEDDING_MODEL=nvidia/nv-embed-v1
```

## 3) Frontend

Frontend should call only the core backend URL.
Do not call docsum service directly from frontend.

## 4) Health Checks

- Core health: `/health`
- Docsum health: `/health`

## 5) Notes

- If `DOCSUM_API_BASE_URL` is empty in core backend, it falls back to local in-process summarizer behavior.
- Set `DOCSUM_PROXY_ONLY=true` in production to disable local fallback and avoid accidental memory-heavy summarizer loading in core service.
- If `DOCSUM_INTERNAL_TOKEN` is set in docsum service, all routes except `/` and `/health` require header:

```http
X-Internal-Token: <token>
```

- Free-tier memory tip: keep `EMBEDDING_PROVIDER=openai` for doc service to avoid loading torch/sentence-transformers at runtime, and point `OPENAI_EMBEDDING_MODEL` at a NVIDIA embedding model such as `nvidia/nv-embed-v1`.

- Rotate API keys before production deployment.
