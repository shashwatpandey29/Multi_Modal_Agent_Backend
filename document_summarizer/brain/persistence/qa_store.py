from .database import SessionLocal
from .models import QAHistory

def save_qa(paper_id, question, answer):
    db = SessionLocal()
    db.add(QAHistory(
        paper_id=paper_id,
        question=question,
        answer=answer
    ))
    db.commit()
    db.close()

def get_all_questions(paper_id):
    db = SessionLocal()
    qas = db.query(QAHistory).filter(
        QAHistory.paper_id == paper_id
    ).all()
    db.close()
    return qas


def get_recent_questions(paper_id, limit=20):
    db = SessionLocal()
    qas = (
        db.query(QAHistory)
        .filter(QAHistory.paper_id == paper_id)
        .order_by(QAHistory.id.desc())
        .limit(limit)
        .all()
    )
    db.close()
    return qas


def get_exact_answer(paper_id, question):
    db = SessionLocal()
    qa = (
        db.query(QAHistory)
        .filter(
            QAHistory.paper_id == paper_id,
            QAHistory.question == question,
        )
        .order_by(QAHistory.id.desc())
        .first()
    )
    db.close()
    return qa


def count_questions(paper_id):
    db = SessionLocal()
    count = db.query(QAHistory).filter(QAHistory.paper_id == paper_id).count()
    db.close()
    return count
