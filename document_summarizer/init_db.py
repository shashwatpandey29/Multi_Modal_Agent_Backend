from brain.persistence.database import engine
from brain.persistence.models import Base

Base.metadata.create_all(bind=engine)
print("Database initialized")
