import base64
import binascii
import re
from uuid import uuid4

from fastapi import APIRouter, HTTPException, UploadFile, File, Header, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import json
import os
import shutil
import requests

# ----------------------------
# AI Agents
# ----------------------------
from agents.image_generator import generate_image_and_video
from agents.coder import generate_code
from agents.chat import generate_text, stream_text
from agents.llm_provider import get_llm_provider, list_openrouter_free_models
from agents.memory import (
    clear_memory_session,
    export_knowledge_bridge,
    get_memory_snapshot,
    get_session_mode,
    import_knowledge_bridge,
    set_session_mode,
)

# ----------------------------
# Document Summarizer Proxy Config
# ----------------------------
DOCSUM_API_BASE_URL = os.getenv("DOCSUM_API_BASE_URL", "").strip().rstrip("/")
DOCSUM_INTERNAL_TOKEN = os.getenv("DOCSUM_INTERNAL_TOKEN", "").strip()
DOCSUM_TIMEOUT_SEC = int(os.getenv("DOCSUM_TIMEOUT_SEC", "180"))
DOCSUM_PROXY_ONLY = os.getenv("DOCSUM_PROXY_ONLY", "false").strip().lower() in {"1", "true", "yes", "on"}
JUDGE0_URL = os.getenv("JUDGE0_URL", "https://ce.judge0.com/submissions?base64_encoded=true&wait=true").strip()
JUDGE0_RAPIDAPI_KEY = os.getenv("JUDGE0_RAPIDAPI_KEY", "").strip()
JUDGE0_RAPIDAPI_HOST = os.getenv("JUDGE0_RAPIDAPI_HOST", "").strip()
PISTON_URL = os.getenv("PISTON_URL", "https://emkc.org/api/v2/piston/execute").strip()
EXECUTION_TIMEOUT_SEC = int(os.getenv("EXECUTION_TIMEOUT_SEC", "25"))

PISTON_RUNTIME_BY_LANGUAGE_ID: Dict[int, Dict[str, str]] = {
    50: {"language": "c", "version": "10.2.0"},
    54: {"language": "cpp", "version": "10.2.0"},
    60: {"language": "go", "version": "1.16.2"},
    62: {"language": "java", "version": "15.0.2"},
    63: {"language": "javascript", "version": "18.15.0"},
    71: {"language": "python", "version": "3.10.0"},
    73: {"language": "rust", "version": "1.68.2"},
}

_REQUEST_ID_SANITIZER = re.compile(r"[^A-Za-z0-9._:-]+")


router = APIRouter(
    prefix="/ai",
    tags=["AI Services"]
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ==========================================================
# 🔹 SCHEMAS
# ==========================================================

class PromptRequest(BaseModel):
    prompt: str


class TextRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    session_id: Optional[str] = None
    session_mode: Optional[str] = None
    response_length: Optional[str] = None


class SessionModeRequest(BaseModel):
    mode: str


class MemoryBridgeExportRequest(BaseModel):
    source_session_id: str
    source_mode: Optional[str] = None
    fact_ids: Optional[List[int]] = None
    node_ids: Optional[List[int]] = None
    edge_ids: Optional[List[int]] = None
    message_ids: Optional[List[int]] = None


class MemoryBridgeImportRequest(BaseModel):
    target_session_id: str
    target_mode: Optional[str] = None
    payload: Dict[str, Any]
    include_messages: bool = True
    include_facts: bool = True
    include_graph: bool = True


class AskRequest(BaseModel):
    paper_id: int
    question: str


class SearchRequest(BaseModel):
    paper_id: int
    query: str


class ExecuteRequest(BaseModel):
    language_id: int = Field(gt=0)
    source_code: str = Field(min_length=1, max_length=120_000)
    stdin: str = Field(default="", max_length=20_000)
    cpu_time_limit: float = Field(default=10, gt=0, le=20)
    memory_limit: int = Field(default=128_000, ge=16_000, le=256_000)


def _use_docsum_proxy() -> bool:
    return bool(DOCSUM_API_BASE_URL)


def _ensure_docsum_mode():
    if DOCSUM_PROXY_ONLY and not _use_docsum_proxy():
        raise HTTPException(
            status_code=503,
            detail=(
                "DOCSUM proxy-only mode is enabled, but DOCSUM_API_BASE_URL is not set. "
                "Set DOCSUM_API_BASE_URL to the deployed document summarizer service URL."
            ),
        )


def _docsum_headers() -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if DOCSUM_INTERNAL_TOKEN:
        headers["X-Internal-Token"] = DOCSUM_INTERNAL_TOKEN
    return headers


def _docsum_url(path: str) -> str:
    return f"{DOCSUM_API_BASE_URL}{path}"


def _raise_proxy_error(response: requests.Response):
    detail: Any = response.text or "Document summarizer service error"
    try:
        payload = response.json()
        detail = payload.get("detail", payload)
    except ValueError:
        pass

    raise HTTPException(status_code=response.status_code, detail=detail)


def _proxy_get(path: str):
    try:
        response = requests.get(
            _docsum_url(path),
            headers=_docsum_headers(),
            timeout=DOCSUM_TIMEOUT_SEC,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Doc summarizer unavailable: {exc}")

    if response.status_code >= 400:
        _raise_proxy_error(response)

    return response.json()


def _proxy_post_json(path: str, data: Dict[str, Any]):
    try:
        response = requests.post(
            _docsum_url(path),
            json=data,
            headers=_docsum_headers(),
            timeout=DOCSUM_TIMEOUT_SEC,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Doc summarizer unavailable: {exc}")

    if response.status_code >= 400:
        _raise_proxy_error(response)

    return response.json()


def _proxy_upload(file: UploadFile):
    try:
        file.file.seek(0)
        response = requests.post(
            _docsum_url("/upload"),
            files={
                "file": (
                    file.filename,
                    file.file,
                    file.content_type or "application/octet-stream",
                )
            },
            headers=_docsum_headers(),
            timeout=DOCSUM_TIMEOUT_SEC,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Doc summarizer unavailable: {exc}")

    if response.status_code >= 400:
        _raise_proxy_error(response)

    return response.json()


def _decode_base64_utf8(value: str) -> str:
    if not value:
        return ""
    normalized = value.strip()
    missing_padding = len(normalized) % 4
    if missing_padding:
        normalized += "=" * (4 - missing_padding)

    try:
        return base64.b64decode(normalized, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        # Some providers return plain UTF-8 text already.
        return value


def _encode_base64_utf8(value: str) -> str:
    if not value:
        return ""
    return base64.b64encode(value.encode("utf-8")).decode("utf-8")


def _resolve_request_id(incoming_request_id: Optional[str]) -> str:
    cleaned = _REQUEST_ID_SANITIZER.sub("-", (incoming_request_id or "").strip())
    cleaned = cleaned.strip("-")[:64]
    if cleaned:
        return cleaned
    return uuid4().hex


def _judge0_headers() -> Dict[str, str]:
    headers: Dict[str, str] = {"Content-Type": "application/json"}

    if JUDGE0_RAPIDAPI_KEY:
        headers["X-RapidAPI-Key"] = JUDGE0_RAPIDAPI_KEY
    if JUDGE0_RAPIDAPI_HOST:
        headers["X-RapidAPI-Host"] = JUDGE0_RAPIDAPI_HOST

    return headers


def _normalize_judge0_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    stdout = _decode_base64_utf8(str(payload.get("stdout") or ""))
    stderr = _decode_base64_utf8(str(payload.get("stderr") or ""))
    compile_output = _decode_base64_utf8(str(payload.get("compile_output") or ""))
    message = _decode_base64_utf8(str(payload.get("message") or ""))

    status_payload = payload.get("status") if isinstance(payload.get("status"), dict) else {}
    status_id_raw = status_payload.get("id") if isinstance(status_payload, dict) else None

    try:
        status_id = int(status_id_raw) if status_id_raw is not None else 0
    except (TypeError, ValueError):
        status_id = 0

    exit_code_raw = payload.get("exit_code")
    try:
        exit_code = int(exit_code_raw) if exit_code_raw is not None else (1 if status_id >= 6 else 0)
    except (TypeError, ValueError):
        exit_code = 1 if status_id >= 6 else 0

    time_raw = payload.get("time")
    try:
        time_ms = float(time_raw) * 1000
    except (TypeError, ValueError):
        time_ms = 0.0

    stderr_parts = [part for part in [stderr, compile_output, message] if part]

    normalized: Dict[str, Any] = {
        "provider": "judge0",
        "stdout": stdout,
        "stderr": "\n".join(stderr_parts),
        "exitCode": exit_code,
        "timeMs": round(time_ms, 2),
    }

    if isinstance(payload.get("memory"), (int, float)):
        normalized["memoryKb"] = int(payload["memory"])

    return normalized


def _run_with_piston(payload: ExecuteRequest) -> Dict[str, Any]:
    runtime = PISTON_RUNTIME_BY_LANGUAGE_ID.get(payload.language_id)
    if not runtime:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "execution_not_configured",
                "message": "No fallback runtime is configured for the selected language.",
            },
        )

    source_code = _decode_base64_utf8(payload.source_code)

    try:
        response = requests.post(
            PISTON_URL,
            json={
                "language": runtime["language"],
                "version": runtime["version"],
                "files": [{"name": "main", "content": source_code}],
                "stdin": payload.stdin,
                "run_timeout": int(payload.cpu_time_limit * 1000),
            },
            timeout=EXECUTION_TIMEOUT_SEC,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Piston execution failed: {exc}")

    if response.status_code >= 400:
        detail = response.text or "Piston returned an error"
        raise HTTPException(status_code=502, detail=f"Piston execution failed: {detail}")

    data = response.json()
    compile_stdout = str(data.get("compile", {}).get("stdout") or "")
    compile_stderr = str(data.get("compile", {}).get("stderr") or "")
    run_stdout = str(data.get("run", {}).get("stdout") or "")
    run_stderr = str(data.get("run", {}).get("stderr") or "")
    code_raw = data.get("run", {}).get("code")

    try:
        exit_code = int(code_raw) if code_raw is not None else (1 if (compile_stderr or run_stderr) else 0)
    except (TypeError, ValueError):
        exit_code = 1 if (compile_stderr or run_stderr) else 0

    return {
        "provider": "piston",
        "stdout": "\n".join(part for part in [compile_stdout, run_stdout] if part),
        "stderr": "\n".join(part for part in [compile_stderr, run_stderr] if part),
        "exitCode": exit_code,
        "timeMs": 0,
    }


def _get_local_brain():
    from document_summarizer.brain.brain import ResearchBrain

    return ResearchBrain()


# ==========================================================
# 🔹 AI GENERATION ROUTES
# ==========================================================

@router.post("/generate-image")
async def image_route(request: PromptRequest):

    result = await generate_image_and_video(request.prompt)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return FileResponse(
        path=result["file_path"],
        media_type="image/png",
        filename="generated_image.png"
    )


@router.post("/generate-code")
async def code_route(request: PromptRequest):

    result = generate_code(request.prompt)

    if isinstance(result, str) and result.startswith("Error"):
        raise HTTPException(status_code=500, detail=result)

    return {
        "status": "success",
        "generated_code": result
    }


@router.post("/execute")
def execute_code(request: ExecuteRequest):
    if "rapidapi.com" in JUDGE0_URL.lower() and not JUDGE0_RAPIDAPI_KEY:
        return _run_with_piston(request)

    payload = request.model_dump()

    # Judge0 expects stdin to be base64 when base64_encoded=true is set.
    if "base64_encoded=true" in JUDGE0_URL.lower():
        payload["stdin"] = _encode_base64_utf8(request.stdin)

    try:
        response = requests.post(
            JUDGE0_URL,
            json=payload,
            headers=_judge0_headers(),
            timeout=EXECUTION_TIMEOUT_SEC,
        )
    except requests.RequestException:
        return _run_with_piston(request)

    if response.status_code >= 400:
        try:
            return _run_with_piston(request)
        except HTTPException as fallback_error:
            detail = response.text or "Judge0 request failed"
            raise HTTPException(status_code=response.status_code, detail={"judge0": detail, "fallback": fallback_error.detail})

    try:
        result = response.json()
    except ValueError:
        return _run_with_piston(request)

    return _normalize_judge0_result(result)


@router.post("/generate-text")
async def text_route(
    request: TextRequest,
    response: Response,
    x_request_id: Optional[str] = Header(default=None, alias="X-Request-ID"),
):
    request_id = _resolve_request_id(x_request_id)

    result = await generate_text(
        request.prompt,
        model=request.model,
        session_id=request.session_id,
        session_mode=request.session_mode,
        response_length=request.response_length,
        request_id=request_id,
    )

    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result["message"])

    response.headers["X-Request-ID"] = request_id
    result["request_id"] = request_id
    return result


@router.post("/generate-text/stream")
async def text_stream_route(
    request: TextRequest,
    x_request_id: Optional[str] = Header(default=None, alias="X-Request-ID"),
):
    request_id = _resolve_request_id(x_request_id)

    def event_stream():
        try:
            for chunk in stream_text(
                request.prompt,
                model=request.model,
                session_id=request.session_id,
                session_mode=request.session_mode,
                response_length=request.response_length,
                request_id=request_id,
            ):
                if not chunk:
                    continue

                payload = json.dumps({"type": "chunk", "content": chunk}, ensure_ascii=False)
                yield f"data: {payload}\n\n"

            effective_mode = request.session_mode
            if not effective_mode and request.session_id:
                effective_mode = get_session_mode(request.session_id).get("mode")

            done_payload = json.dumps(
                {
                    "type": "done",
                    "session_mode": effective_mode or "persistent",
                    "request_id": request_id,
                }
            )
            yield f"data: {done_payload}\n\n"
        except Exception as exc:
            error_payload = json.dumps(
                {"type": "error", "message": str(exc), "request_id": request_id},
                ensure_ascii=False,
            )
            yield f"data: {error_payload}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Request-ID": request_id,
        },
    )


@router.get("/models/openrouter")
def get_openrouter_free_models():
    provider = get_llm_provider()
    default_model = os.getenv(
        "OPENROUTER_CHAT_MODEL",
        os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
    )

    try:
        models = list_openrouter_free_models()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch OpenRouter models: {exc}")

    if default_model and all(model.get("id") != default_model for model in models):
        models.insert(
            0,
            {
                "id": default_model,
                "name": f"{default_model} (default)",
            },
        )

    return {
        "provider": provider,
        "enabled": provider == "openrouter",
        "default_model": default_model,
        "models": models,
    }


@router.get("/memory/{session_id}")
def get_nova_memory(session_id: str, mode: Optional[str] = None):
    return get_memory_snapshot(session_id=session_id, session_mode=mode)


@router.delete("/memory/{session_id}")
def reset_nova_memory(session_id: str, mode: Optional[str] = None):
    return clear_memory_session(session_id=session_id, session_mode=mode)


@router.get("/memory/{session_id}/mode")
def get_nova_memory_mode(session_id: str):
    return get_session_mode(session_id=session_id)


@router.post("/memory/{session_id}/mode")
def set_nova_memory_mode(session_id: str, request: SessionModeRequest):
    return set_session_mode(session_id=session_id, session_mode=request.mode)


@router.post("/memory/bridge/export")
def export_nova_bridge(request: MemoryBridgeExportRequest):
    return export_knowledge_bridge(
        source_session_id=request.source_session_id,
        source_mode=request.source_mode,
        fact_ids=request.fact_ids,
        node_ids=request.node_ids,
        edge_ids=request.edge_ids,
        message_ids=request.message_ids,
    )


@router.post("/memory/bridge/import")
def import_nova_bridge(request: MemoryBridgeImportRequest):
    return import_knowledge_bridge(
        target_session_id=request.target_session_id,
        bridge_payload=request.payload,
        target_mode=request.target_mode,
        include_messages=request.include_messages,
        include_facts=request.include_facts,
        include_graph=request.include_graph,
    )


# ==========================================================
# 🔹 RESEARCH PAPER ANALYZER ROUTES
# ==========================================================

# ---------- Upload ----------
@router.post("/upload")
def upload_paper(file: UploadFile = File(...)):
    _ensure_docsum_mode()

    if _use_docsum_proxy():
        return _proxy_upload(file)

    brain = _get_local_brain()

    try:
        filename = file.filename
        ext = os.path.splitext(filename)[1].lower()

        if ext not in [".pdf", ".docx", ".txt"]:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

        file_path = os.path.join(UPLOAD_DIR, filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = brain.ingest(file_path)

        return {
            "message": "Paper uploaded and processed",
            "paper_id": result["paper_id"],
            "analysis_time_sec": result["analysis_time_sec"],
            "cached": result.get("cached", False),
            "filename": filename
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- List Papers ----------
@router.get("/papers")
def get_papers():
    _ensure_docsum_mode()

    if _use_docsum_proxy():
        return _proxy_get("/papers")

    from document_summarizer.brain.persistence.paper_store import list_papers

    papers = list_papers()

    return [
        {
            "paper_id": p.id,
            "filename": p.filename,
            "created_at": p.created_at
        }
        for p in papers
    ]


# ---------- Ask ----------
@router.post("/ask")
def ask_paper(req: AskRequest):
    _ensure_docsum_mode()

    if _use_docsum_proxy():
        return _proxy_post_json(
            "/ask",
            {"paper_id": req.paper_id, "question": req.question},
        )

    brain = _get_local_brain()

    try:
        brain.load(req.paper_id)
        return brain.ask(req.question)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------- Summary ----------
@router.get("/summary/{paper_id}")
def get_summary(paper_id: int):
    _ensure_docsum_mode()

    if _use_docsum_proxy():
        return _proxy_get(f"/summary/{paper_id}")

    try:
        from document_summarizer.brain.persistence.analysis_store import get_analysis

        analysis = get_analysis(paper_id)
        if not analysis:
            raise HTTPException(status_code=404, detail="Summary not available")

        fact_points = []
        for raw_line in (analysis.key_learnings or "").splitlines():
            point = raw_line.strip().lstrip("-*").strip()
            if point:
                fact_points.append(point)

        return {
            "summary": analysis.summary,
            "fact_points": fact_points,
            "analysis_time_sec": analysis.analysis_time_sec,
            "precomputed": True,
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------- Teacher ----------
@router.get("/teach/{paper_id}")
def teach_paper(paper_id: int):
    _ensure_docsum_mode()

    if _use_docsum_proxy():
        return _proxy_get(f"/teach/{paper_id}")

    brain = _get_local_brain()

    try:
        brain.load(paper_id)
        return {"teaching": brain.teach()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------- Analysis ----------
@router.get("/analysis/{paper_id}")
def get_analysis_api(paper_id: int):
    _ensure_docsum_mode()

    if _use_docsum_proxy():
        return _proxy_get(f"/analysis/{paper_id}")

    try:
        from document_summarizer.brain.persistence.analysis_store import get_analysis

        analysis = get_analysis(paper_id)

        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not available")

        return {
            "summary": analysis.summary,
            "key_learnings": analysis.key_learnings,
            "limitations": analysis.limitations,
            "contributions": analysis.contributions,
            "analysis_time_sec": analysis.analysis_time_sec
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Stats ----------
@router.get("/stats/{paper_id}")
def get_stats(paper_id: int):
    _ensure_docsum_mode()

    if _use_docsum_proxy():
        return _proxy_get(f"/stats/{paper_id}")

    try:
        from document_summarizer.brain.persistence.paper_store import count_chunks
        from document_summarizer.brain.persistence.qa_store import count_questions

        total_chunks = count_chunks(paper_id)
        total_questions = count_questions(paper_id)

        return {
            "total_chunks": total_chunks,
            "total_questions": total_questions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Search ----------
@router.post("/search")
def search(req: SearchRequest):
    _ensure_docsum_mode()

    if _use_docsum_proxy():
        return _proxy_post_json(
            "/search",
            {"paper_id": req.paper_id, "query": req.query},
        )

    brain = _get_local_brain()

    try:
        from document_summarizer.brain.config import TOP_K

        brain.load(req.paper_id)

        if not brain.retriever:
            raise HTTPException(status_code=404, detail="Paper not loaded properly")

        results = brain.retriever.retrieve(req.query, TOP_K)

        return {"results": results}

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
