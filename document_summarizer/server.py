# server.py
import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from document_summarizer.brain.brain import ResearchBrain
from document_summarizer.brain.config import TOP_K
from document_summarizer.brain.persistence.analysis_store import get_analysis
from document_summarizer.brain.persistence.paper_store import list_papers, count_chunks
from document_summarizer.brain.persistence.qa_store import count_questions

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
DOCSUM_INTERNAL_TOKEN = os.getenv("DOCSUM_INTERNAL_TOKEN", "").strip()

app = FastAPI(title="AI Research Paper Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def verify_internal_token(request: Request, call_next):
    if request.url.path in {"/", "/health"}:
        return await call_next(request)

    if DOCSUM_INTERNAL_TOKEN:
        token = request.headers.get("X-Internal-Token", "")
        if token != DOCSUM_INTERNAL_TOKEN:
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    return await call_next(request)


# ---------- Schemas ----------
class AskRequest(BaseModel):
    paper_id: int
    question: str


class SearchRequest(BaseModel):
    paper_id: int
    query: str


def get_brain():
    return ResearchBrain()


# ---------- Health ----------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"status": "running", "service": "document-summarizer"}


# ---------- Upload ----------
@app.post("/upload")
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
@app.get("/papers")
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
@app.post("/ask")
def ask_paper(req: AskRequest):
    brain = get_brain()

    try:
        brain.load(req.paper_id)
        return brain.ask(req.question)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------- Summary ----------
@app.get("/summary/{paper_id}")
def get_summary(paper_id: int):
    try:
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

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------- Teacher ----------
@app.get("/teach/{paper_id}")
def teach_paper(paper_id: int):
    brain = get_brain()

    try:
        brain.load(paper_id)
        return {"teaching": brain.teach()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------- Analysis ----------
@app.get("/analysis/{paper_id}")
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
@app.get("/stats/{paper_id}")
def get_stats(paper_id: int):
    try:
        total_chunks = count_chunks(paper_id)
        total_questions = count_questions(paper_id)

        return {
            "total_chunks": total_chunks,
            "total_questions": total_questions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Search ----------
@app.post("/search")
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
