from sentence_transformers import SentenceTransformer
from document_summarizer.brain.exceptions import EmbeddingError

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, texts):
        try:
            return self.model.encode(texts, convert_to_numpy=True)
        except Exception as e:
            raise EmbeddingError(str(e))
