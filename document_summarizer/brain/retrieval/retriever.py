class Retriever:
    def __init__(self, embedder, store):
        self.embedder = embedder
        self.store = store

    def retrieve(self, query, k):
        q_emb = self.embedder.embed([query])[0]
        return self.store.search(q_emb, k)
