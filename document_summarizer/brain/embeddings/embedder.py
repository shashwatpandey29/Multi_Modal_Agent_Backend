import os

import numpy as np
from openai import OpenAI

from document_summarizer.brain.exceptions import EmbeddingError


class Embedder:
    def __init__(self):
        self.provider = os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower()
        self.openai_model = os.getenv("OPENAI_EMBEDDING_MODEL", "nvidia/nv-embed-v1")
        self.local_model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

        self._openai_client = None
        self._local_model = None

        if self.provider in {"openai", "api"}:
            nvidia_api_key = os.getenv("NVIDIA_API_KEY", "").strip()
            api_key = nvidia_api_key or os.getenv("OPENAI_API_KEY", "").strip()
            if not api_key:
                raise EmbeddingError(
                    "NVIDIA_API_KEY or OPENAI_API_KEY is required when EMBEDDING_PROVIDER is openai/api"
                )

            if self.openai_model.startswith(("nvidia/", "openai/gpt-oss-")) and not nvidia_api_key:
                raise EmbeddingError("NVIDIA_API_KEY is required for NVIDIA embedding models")

            if nvidia_api_key:
                base_url = os.getenv("NVIDIA_API_BASE_URL", "").strip() or "https://integrate.api.nvidia.com/v1"
            else:
                base_url = os.getenv("OPENAI_BASE_URL", "").strip() or os.getenv("NVIDIA_API_BASE_URL", "").strip() or "https://integrate.api.nvidia.com/v1"

            client_kwargs = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url

            self._openai_client = OpenAI(**client_kwargs)
        elif self.provider == "local":
            # Lazy local import so free-tier deploys do not load torch/transformers unless explicitly requested.
            from sentence_transformers import SentenceTransformer

            self._local_model = SentenceTransformer(self.local_model_name)
        else:
            raise EmbeddingError(
                "Unsupported EMBEDDING_PROVIDER. Use one of: openai, api, local"
            )

    def embed(self, texts, input_type: str = "passage"):
        try:
            if not texts:
                return np.empty((0, 0), dtype=np.float32)

            if self.provider in {"openai", "api"}:
                request_kwargs = {
                    "model": self.openai_model,
                    "input": texts,
                }

                if self.openai_model.startswith(("nvidia/nv-embedqa-", "nvidia/llama-nemotron-embed-", "nvidia/nv-embedcode-")):
                    request_kwargs["input_type"] = input_type

                response = self._openai_client.embeddings.create(**request_kwargs)
                vectors = [item.embedding for item in response.data]
                return np.array(vectors, dtype=np.float32)

            vectors = self._local_model.encode(texts, convert_to_numpy=True)
            return np.array(vectors, dtype=np.float32)
        except Exception as e:
            raise EmbeddingError(str(e))
