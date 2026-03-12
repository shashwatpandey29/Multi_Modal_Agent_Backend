from sqlalchemy import Column, Integer, String, Text, DateTime
from datetime import datetime
from .database import Base

class Paper(Base):
    __tablename__ = "papers"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)
    paper_id = Column(Integer, index=True)
    section = Column(String)
    text = Column(Text)


class QAHistory(Base):
    __tablename__ = "qa_history"

    id = Column(Integer, primary_key=True)
    paper_id = Column(Integer, index=True)
    question = Column(Text)
    answer = Column(Text)


from sqlalchemy import Column, Integer, String, Text, DateTime, Float
from datetime import datetime
from .database import Base


class PaperAnalysis(Base):
    __tablename__ = "paper_analysis"

    id = Column(Integer, primary_key=True)
    paper_id = Column(Integer, index=True, unique=True)

    summary = Column(Text)
    key_learnings = Column(Text)
    limitations = Column(Text)
    contributions = Column(Text)

    analysis_time_sec = Column(Float)  


