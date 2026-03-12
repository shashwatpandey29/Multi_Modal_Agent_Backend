from .database import SessionLocal
from .models import PaperAnalysis


def save_analysis(
    paper_id,
    summary,
    key_learnings,
    limitations,
    contributions,
    analysis_time_sec
):
    db = SessionLocal()

    analysis = PaperAnalysis(
        paper_id=paper_id,
        summary=summary,
        key_learnings=key_learnings,
        limitations=limitations,
        contributions=contributions,
        analysis_time_sec=analysis_time_sec
    )

    db.add(analysis)
    db.commit()
    db.close()


def get_analysis(paper_id):
    db = SessionLocal()
    analysis = db.query(PaperAnalysis).filter(
        PaperAnalysis.paper_id == paper_id
    ).first()
    db.close()
    return analysis
