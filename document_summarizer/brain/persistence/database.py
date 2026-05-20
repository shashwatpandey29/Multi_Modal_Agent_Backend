import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Allow overriding the DB location via env to avoid CWD-dependent paths
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./brain.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()
