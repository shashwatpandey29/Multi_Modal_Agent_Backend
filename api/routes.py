from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Any, Dict
import os
import shutil
import requests

# ----------------------------
# AI Agents
# ----------------------------
from agents.image_generator import generate_image_and_video
from agents.coder import generate_code
from agents.chat import generate_text

# ----------------------------
# Document Summarizer Proxy Config
# ----------------------------
DOCSUM_API_BASE_URL = os.getenv("DOCSUM_API_BASE_URL", "").strip().rstrip("/")
DOCSUM_INTERNAL_TOKEN = os.getenv("DOCSUM_INTERNAL_TOKEN", "").strip()
DOCSUM_TIMEOUT_SEC = int(os.getenv("DOCSUM_TIMEOUT_SEC", "180"))
DOCSUM_PROXY_ONLY = os.getenv("DOCSUM_PROXY_ONLY", "false").strip().lower() in {"1", "true", "yes", "on"}


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


class AskRequest(BaseModel):
    paper_id: int
    question: str


class SearchRequest(BaseModel):
    paper_id: int
    query: str


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


@router.post("/generate-text")
async def text_route(request: PromptRequest):

    result = await generate_text(request.prompt)

    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result["message"])

    return result


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
