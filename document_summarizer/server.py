# server.py
import os
import shutil
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from document_summarizer.brain.brain import ResearchBrain
from document_summarizer import cache
from document_summarizer.brain.config import TOP_K
from document_summarizer.brain.exceptions import EmbeddingError, LLMError, PDFLoadError
from document_summarizer.brain.prompts.analysis import full_analysis_prompt
from document_summarizer.brain.persistence.analysis_store import get_analysis, save_analysis
from document_summarizer.brain.persistence.paper_store import list_papers, count_chunks, load_chunks
from document_summarizer.brain.persistence.qa_store import count_questions
from document_summarizer.brain.utils.timer import Timer
from document_summarizer.metrics import record_request
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import time

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
DOCSUM_INTERNAL_TOKEN = os.getenv("DOCSUM_INTERNAL_TOKEN", "").strip()
LOGGER = logging.getLogger(__name__)

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


def _upload_http_error(exc: Exception) -> HTTPException:
    detail = str(exc) or exc.__class__.__name__
    lowered = detail.lower()

    if isinstance(exc, (PDFLoadError, EmbeddingError, LLMError)):
        status_code = 503
    elif isinstance(exc, ValueError) or "no readable text" in lowered or "no valid chunks" in lowered:
        status_code = 400
    elif "nvidia_api_key" in lowered or "embedding" in lowered or "provider" in lowered:
        status_code = 503
    else:
        status_code = 500

    return HTTPException(status_code=status_code, detail=detail)


def _ask_http_error(exc: Exception) -> HTTPException:
    detail = str(exc) or exc.__class__.__name__
    lowered = detail.lower()

    if isinstance(exc, (EmbeddingError, LLMError)):
        status_code = 503
    elif "paper not ingested" in lowered or "paper has no chunks" in lowered or "summary not available" in lowered:
        status_code = 404
    elif "nvidia_api_key" in lowered or "embedding" in lowered or "provider" in lowered:
        status_code = 503
    else:
        status_code = 400

    return HTTPException(status_code=status_code, detail=detail)


def _parse_full_analysis(full_analysis: str):
    summary = ""
    key_learnings = ""
    limitations = ""
    contributions = ""

    try:
        summary = full_analysis.split("=== KEY LEARNINGS ===")[0]
        summary = summary.replace("=== SUMMARY ===", "").strip()

        part2 = full_analysis.split("=== KEY LEARNINGS ===")[1]
        key_learnings = part2.split("=== MAIN CONTRIBUTIONS ===")[0].strip()

        part3 = part2.split("=== MAIN CONTRIBUTIONS ===")[1]
        contributions = part3.split("=== LIMITATIONS ===")[0].strip()
        limitations = part3.split("=== LIMITATIONS ===")[1].strip()
    except Exception:
        summary = full_analysis
        key_learnings = "Parsing failed"
        contributions = "Parsing failed"
        limitations = "Parsing failed"

    return summary, key_learnings, limitations, contributions


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
    try:
        brain = get_brain()
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
        LOGGER.exception("Upload failed")
        raise _upload_http_error(e)


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
    try:
        brain = get_brain()
        brain.load(req.paper_id)
        return brain.ask(req.question)

    except Exception as e:
        LOGGER.exception("Ask failed")
        raise _ask_http_error(e)


# ---------- Summary ----------
@app.get("/summary/{paper_id}")
def get_summary(paper_id: int):
    try:
        t0 = time.perf_counter()
        # check shared cache first
        summary_key = cache.make_key("summary", paper_id)
        cached = cache.get(summary_key)
        if cached:
            return {
                "summary": cached.get("summary"),
                "fact_points": cached.get("fact_points", []),
                "analysis_time_sec": cached.get("analysis_time_sec", 0),
                "precomputed": True,
            }

        analysis = get_analysis(paper_id)
        if not analysis:
            brain = get_brain()
            brain.load(paper_id)

            chunks = load_chunks(paper_id)
            if not chunks:
                raise HTTPException(status_code=404, detail="Summary not available")

            all_chunks = [f"[{chunk.section.upper()}]\n{chunk.text}" for chunk in chunks]
            combined_text = "\n\n".join(all_chunks[:3])[:4000]

            with Timer() as timer:
                full_analysis = brain.analysis_llm.generate(full_analysis_prompt(combined_text))

            summary, key_learnings, limitations, contributions = _parse_full_analysis(full_analysis)
            save_analysis(paper_id, summary, key_learnings, limitations, contributions, timer.elapsed)

            fact_points = []
            for raw_line in key_learnings.splitlines():
                point = raw_line.strip().lstrip("-*").strip()
                if point:
                    fact_points.append(point)

            # cache summary in Redis for fast retrieval
            try:
                cache.set(summary_key, {
                    "summary": summary,
                    "fact_points": fact_points,
                    "analysis_time_sec": timer.elapsed,
                })
            except Exception:
                pass

            resp = {
                "summary": summary,
                "fact_points": fact_points,
                "analysis_time_sec": timer.elapsed,
                "precomputed": False,
            }
            record_request("summary", time.perf_counter() - t0)
            return resp

        fact_points = []
        for raw_line in (analysis.key_learnings or "").splitlines():
            point = raw_line.strip().lstrip("-*").strip()
            if point:
                fact_points.append(point)

        resp = {
            "summary": analysis.summary,
            "fact_points": fact_points,
            "analysis_time_sec": analysis.analysis_time_sec,
            "precomputed": True,
        }
        record_request("summary", time.perf_counter() - t0)
        return resp

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
        t0 = time.perf_counter()
        total_chunks = count_chunks(paper_id)
        total_questions = count_questions(paper_id)

        resp = {
            "total_chunks": total_chunks,
            "total_questions": total_questions
        }
        record_request("stats", time.perf_counter() - t0)
        return resp

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def metrics():
    data = generate_latest()
    return JSONResponse(content=data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


# ---------- Search ----------
@app.post("/search")
def search(req: SearchRequest):
    brain = get_brain()

    try:
        t0 = time.perf_counter()
        brain.load(req.paper_id)

        if not brain.retriever:
            raise HTTPException(status_code=404, detail="Paper not loaded properly")

        # check cache for this query
        key = cache.make_key("search", req.paper_id, cache.hash_text(req.query.strip().lower()))
        cached = cache.get(key)
        if cached:
            return {"results": cached.get("results", [])}

        results = brain.retriever.retrieve(req.query, TOP_K)

        try:
            cache.set(key, {"results": results})
        except Exception:
            pass

        record_request("search", time.perf_counter() - t0)
        return {"results": results}

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
