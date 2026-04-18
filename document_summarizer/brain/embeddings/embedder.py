import os

import numpy as np
from openai import OpenAI

from document_summarizer.brain.exceptions import EmbeddingError


class Embedder:
    def __init__(self):
        self.provider = os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower()
        self.openai_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.local_model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

        self._openai_client = None
        self._local_model = None

        if self.provider in {"openai", "api"}:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EmbeddingError(
                    "OPENAI_API_KEY is required when EMBEDDING_PROVIDER is openai/api"
                )
            self._openai_client = OpenAI(api_key=api_key)
        elif self.provider == "local":
            # Lazy local import so free-tier deploys do not load torch/transformers unless explicitly requested.
            from sentence_transformers import SentenceTransformer

            self._local_model = SentenceTransformer(self.local_model_name)
        else:
            raise EmbeddingError(
                "Unsupported EMBEDDING_PROVIDER. Use one of: openai, api, local"
            )

    def embed(self, texts):
        try:
            if not texts:
                return np.empty((0, 0), dtype=np.float32)

            if self.provider in {"openai", "api"}:
                response = self._openai_client.embeddings.create(
                    model=self.openai_model,
                    input=texts,
                )
                vectors = [item.embedding for item in response.data]
                return np.array(vectors, dtype=np.float32)

            vectors = self._local_model.encode(texts, convert_to_numpy=True)
            return np.array(vectors, dtype=np.float32)
        except Exception as e:
            raise EmbeddingError(str(e))
