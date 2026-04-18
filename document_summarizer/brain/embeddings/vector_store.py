import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

    def add(self, embeddings, texts):
        self.index.add(np.array(embeddings))
        self.texts.extend(texts)

    def search(self, query_embedding, k):
        _, idx = self.index.search(query_embedding.reshape(1, -1), k)
        results = []
        for i in idx[0]:
            if i < 0 or i >= len(self.texts):
                continue
            results.append(self.texts[i])
        return results
