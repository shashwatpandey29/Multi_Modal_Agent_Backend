# server.py
import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from brain.brain import ResearchBrain
from brain.config import TOP_K
from brain.persistence.analysis_store import get_analysis
from brain.persistence.paper_store import list_papers, load_chunks
from brain.persistence.qa_store import get_all_questions

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="AI Research Paper Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    brain = get_brain()

    try:
        brain.load(paper_id)
        return {"summary": brain.summarize()}

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
        chunks = load_chunks(paper_id) or []
        questions = get_all_questions(paper_id) or []

        return {
            "total_chunks": len(chunks),
            "total_questions": len(questions)
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
