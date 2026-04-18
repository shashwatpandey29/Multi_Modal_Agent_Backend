from .database import SessionLocal
from .models import Paper, Chunk


def get_paper_by_filename(filename: str):
    db = SessionLocal()
    paper = db.query(Paper).filter(Paper.filename == filename).first()
    db.close()
    return paper


def save_paper_and_chunks(filename: str, sections: dict):
    db = SessionLocal()

    paper = Paper(filename=filename)
    db.add(paper)
    db.commit()
    db.refresh(paper)

    paper_id = paper.id  

    for section, chunks in sections.items():
        for chunk in chunks:
            db.add(
                Chunk(
                    paper_id=paper_id,
                    section=section,
                    text=chunk
                )
            )

    db.commit()
    db.close()

    return paper_id


def load_chunks(paper_id: int):
    db = SessionLocal()
    chunks = db.query(Chunk).filter(Chunk.paper_id == paper_id).all()
    db.close()
    return chunks


def count_chunks(paper_id: int):
    db = SessionLocal()
    count = db.query(Chunk).filter(Chunk.paper_id == paper_id).count()
    db.close()
    return count


def list_papers():
    db = SessionLocal()
    papers = db.query(Paper).all()
    db.close()
    return papers
