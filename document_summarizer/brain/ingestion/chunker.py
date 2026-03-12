from document_summarizer.brain.config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_text(text: str):
    words = text.split()
    chunks = []
    i = 0

    while i < len(words):
        chunks.append(" ".join(words[i:i + CHUNK_SIZE]))
        i += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks
