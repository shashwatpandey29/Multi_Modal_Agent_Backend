import faiss
import os

INDEX_DIR = "vector_indexes"
os.makedirs(INDEX_DIR, exist_ok=True)

def save_index(index, paper_id):
    path = f"{INDEX_DIR}/paper_{paper_id}.index"
    faiss.write_index(index, path)
    return path

def load_index(paper_id):
    path = f"{INDEX_DIR}/paper_{paper_id}.index"
    if not os.path.exists(path):
        return None
    return faiss.read_index(path)
