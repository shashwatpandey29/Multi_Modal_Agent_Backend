from document_summarizer.brain.brain import ResearchBrain, _parse_full_analysis
from document_summarizer.brain.persistence.analysis_store import save_analysis
from document_summarizer.brain.persistence.paper_store import load_chunks
from document_summarizer.brain.prompts.analysis import full_analysis_prompt
from document_summarizer import cache
from document_summarizer.brain.utils.timer import Timer


def precompute_analysis(paper_id: int):
    brain = ResearchBrain()
    brain.load(paper_id)

    chunks = load_chunks(paper_id)
    if not chunks:
        return {"error": "no chunks"}

    all_chunks = [f"[{chunk.section.upper()}]\n{chunk.text}" for chunk in chunks]
    max_chunks_for_summary = 3
    max_chars = 4000
    important_chunks = all_chunks[:max_chunks_for_summary]
    combined_text = "\n\n".join(important_chunks)[:max_chars]

    with Timer() as timer:
        full_analysis = brain.analysis_llm.generate(full_analysis_prompt(combined_text))

    summary, key_learnings, limitations, contributions = _parse_full_analysis(full_analysis)

    save_analysis(
        paper_id,
        summary,
        key_learnings,
        limitations,
        contributions,
        timer.elapsed,
    )

    # cache summary
    try:
        cache.set(cache.make_key("summary", paper_id), {
            "summary": summary,
            "fact_points": [l.strip().lstrip("-* ") for l in (key_learnings or "").splitlines() if l.strip()],
            "analysis_time_sec": timer.elapsed,
        })
    except Exception:
        pass

    return {"paper_id": paper_id, "analysis_time_sec": timer.elapsed}
