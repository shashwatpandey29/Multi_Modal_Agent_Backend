from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any
import os
import shutil

# ----------------------------
# AI Agents
# ----------------------------
from agents.image_generator import generate_image_and_video
from agents.coder import generate_code
from agents.chat import generate_text

# ----------------------------
# Research Brain Imports
# ----------------------------
from document_summarizer.brain.brain import ResearchBrain
from document_summarizer.brain.config import TOP_K
from document_summarizer.brain.persistence.analysis_store import get_analysis
from document_summarizer.brain.persistence.paper_store import list_papers, load_chunks
from document_summarizer.brain.persistence.qa_store import get_all_questions


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


def get_brain():
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
    brain = get_brain()

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
    brain = get_brain()

    try:
        brain.load(req.paper_id)
        return brain.ask(req.question)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------- Summary ----------
@router.get("/summary/{paper_id}")
def get_summary(paper_id: int):
    brain = get_brain()

    try:
        brain.load(paper_id)
        return {"summary": brain.summarize()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------- Teacher ----------
@router.get("/teach/{paper_id}")
def teach_paper(paper_id: int):
    brain = get_brain()

    try:
        brain.load(paper_id)
        return {"teaching": brain.teach()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------- Analysis ----------
@router.get("/analysis/{paper_id}")
def get_analysis_api(paper_id: int):
    try:
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
    try:
        chunks = load_chunks(paper_id) or []
        questions = get_all_questions(paper_id) or []

        return {
            "total_chunks": len(chunks),
            "total_questions": len(questions)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Search ----------
@router.post("/search")
def search(req: SearchRequest):
    brain = get_brain()

    try:
        brain.load(req.paper_id)

        if not brain.retriever:
            raise HTTPException(status_code=404, detail="Paper not loaded properly")

        results = brain.retriever.retrieve(req.query, TOP_K)

        return {"results": results}

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
