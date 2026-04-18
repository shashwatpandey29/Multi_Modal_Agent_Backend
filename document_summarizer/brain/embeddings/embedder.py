from threading import Lock

from sentence_transformers import SentenceTransformer

from document_summarizer.brain.exceptions import EmbeddingError


_MODEL = None
_MODEL_LOCK = Lock()


def _get_shared_model() -> SentenceTransformer:
    global _MODEL

    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                _MODEL = SentenceTransformer("all-MiniLM-L6-v2")

    return _MODEL

class Embedder:
    def __init__(self):
        self.model = _get_shared_model()

    def embed(self, texts):
        try:
            return self.model.encode(texts, convert_to_numpy=True)
        except Exception as e:
            raise EmbeddingError(str(e))
