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
