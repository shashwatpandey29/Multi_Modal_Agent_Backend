from document_summarizer.brain.logger import get_logger
from document_summarizer.brain.ingestion.pdf_loader import load_pdf_text
from document_summarizer.brain.ingestion.section_parser import split_into_sections
from document_summarizer.brain.ingestion.chunker import chunk_text
from document_summarizer.brain.embeddings.embedder import Embedder
from document_summarizer.brain.embeddings.vector_store import VectorStore
from document_summarizer.brain.persistence.qa_store import save_qa
from document_summarizer.brain.retrieval.retriever import Retriever
from document_summarizer.brain.llm.ollama_llm import OllamaLLM
from document_summarizer.brain.prompts.teacher import teacher_prompt
from document_summarizer.brain.prompts.summary import summary_prompt
from document_summarizer.brain.config import TOP_K
from document_summarizer.brain.prompts.ask import ask_prompt
from document_summarizer.brain.prompts.analysis import full_analysis_prompt
from document_summarizer.brain.persistence.paper_store import (
    get_paper_by_filename,
    save_paper_and_chunks,
    load_chunks
)
from document_summarizer.brain.persistence.vector_store import save_index, load_index
import os

from document_summarizer.brain.utils.context import trim_context
from document_summarizer.brain.config import CHAT_MODEL, ANALYSIS_MODEL
from document_summarizer.brain.utils.timer import Timer
from document_summarizer.brain.persistence.qa_store import save_qa, get_all_questions
import numpy as np
from document_summarizer.brain.persistence.analysis_store import get_analysis

def cosine_sim(a, b):
    """
    a: (d,)
    b: (n, d)
    returns: (n,)
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(b, a)


class ResearchBrain:
    def __init__(self):
        self.logger = get_logger("ResearchBrain")
        self.embedder = Embedder()
        self.retriever = None
        self.paper_id = None
        self.chat_llm = OllamaLLM(CHAT_MODEL)
        self.analysis_llm = OllamaLLM(ANALYSIS_MODEL)

    def load(self, paper_id: int):
        self.paper_id = paper_id
        self._load_from_storage()

    def ingest(self, file_path: str):
        import os

        from document_summarizer.brain.ingestion.document_loader import load_document_text
        from document_summarizer.brain.persistence.paper_store import (
            get_paper_by_filename,
            save_paper_and_chunks
        )
        from document_summarizer.brain.persistence.vector_store import save_index
        from document_summarizer.brain.persistence.analysis_store import save_analysis
        from document_summarizer.brain.prompts.summary import summary_prompt

        filename = os.path.basename(file_path)

# 1️⃣ Check if already ingested
        paper = get_paper_by_filename(filename)
        if paper:
            self.paper_id = paper.id
            self._load_from_storage()
            self.logger.info("Paper already ingested, loaded from storage")

            return {
                "paper_id": self.paper_id,
                "analysis_time_sec": 0,
                "cached": True
            }

        # 2️⃣ Fresh ingestion
        self.logger.info("Ingesting paper for the first time")

        # ---- Load document text (PDF / DOCX / DOC / TXT) ----
        text = load_document_text(file_path)

        if not text or not text.strip():
            raise ValueError("Document contains no readable text")

        # ---- Split into sections ----
        sections = split_into_sections(text)

        # ---- Chunk sections ----
        section_chunks = {}
        for sec, content in sections.items():
            chunks = chunk_text(content)
            if chunks:
                section_chunks[sec] = chunks

        if not section_chunks:
            raise ValueError("No valid chunks created from document")

        # 3️⃣ Save chunks to DB
        self.paper_id = save_paper_and_chunks(filename, section_chunks)

        # 4️⃣ Build embeddings + FAISS
        all_chunks = []
        for sec, chunks in section_chunks.items():
            for chunk in chunks:
                all_chunks.append(f"[{sec.upper()}]\n{chunk}")

        embeddings = self.embedder.embed(all_chunks)

        store = VectorStore(embeddings.shape[1])
        store.add(embeddings, all_chunks)

        # 5️⃣ Save FAISS index
        save_index(store.index, self.paper_id)

        # 6️⃣ Generate & store paper analysis (RUNS ONCE)
        self.logger.info("Generating paper analysis (summary & insights)")
        
        with Timer() as t:
            try:
                # Limit chunks used for summary
                MAX_CHUNKS_FOR_SUMMARY = 3
                MAX_CHARS = 4000

                important_chunks = all_chunks[:MAX_CHUNKS_FOR_SUMMARY]
                combined_text = "\n\n".join(important_chunks)
                combined_text = combined_text[:MAX_CHARS]

                # 🔥 ONE structured prompt instead of 4 calls

                full_analysis = self.analysis_llm.generate(full_analysis_prompt(combined_text))

            except Exception as e:
                self.logger.error(f"Error generating analysis: {e}")
                full_analysis = "Analysis generation failed due to timeout."
        
        summary = "" 
        key_learnings = "" 
        limitations = "" 
        contributions = ""
        try:
            summary = full_analysis.split("=== KEY LEARNINGS ===")[0] \
                .replace("=== SUMMARY ===", "").strip()

            part2 = full_analysis.split("=== KEY LEARNINGS ===")[1]

            key_learnings = part2.split("=== MAIN CONTRIBUTIONS ===")[0].strip()

            part3 = part2.split("=== MAIN CONTRIBUTIONS ===")[1]

            contributions = part3.split("=== LIMITATIONS ===")[0].strip()

            limitations = part3.split("=== LIMITATIONS ===")[1].strip()

        except Exception as e:
            self.logger.warning(f"Parsing failed: {e}")
            summary = full_analysis
            key_learnings = "Parsing failed"
            contributions = "Parsing failed"
            limitations = "Parsing failed"



        save_analysis(
            self.paper_id,
            summary,
            key_learnings,
            limitations,
            contributions,
            t.elapsed
        )

        # 7️⃣ Finalize retriever
        self.retriever = Retriever(self.embedder, store)

        self.logger.info("Ingestion complete (persisted + analyzed)")
        return {
    "paper_id": self.paper_id,
    "analysis_time_sec": t.elapsed
}

    def _load_from_storage(self):
        chunks = load_chunks(self.paper_id)

        texts = [
            f"[{c.section.upper()}]\n{c.text}"
            for c in chunks
        ]

        index = load_index(self.paper_id)

        # 🔥 FIX: rebuild index if missing
        if index is None:
            self.logger.warning(
                "FAISS index missing, rebuilding from stored chunks"
            )

            embeddings = self.embedder.embed(texts)

            store = VectorStore(embeddings.shape[1])
            store.add(embeddings, texts)

            save_index(store.index, self.paper_id)

        else:
            store = VectorStore(index.d)
            store.index = index
            store.texts = texts

        self.retriever = Retriever(self.embedder, store)
        self.logger.info("Loaded paper from storage")

    def summarize(self):
        analysis = get_analysis(self.paper_id)
        return analysis.summary if analysis else "Summary not available"

    def teach(self):
        context = "\n".join(self.retriever.retrieve("explain methodology", TOP_K))
        return self.analysis_llm.generate(teacher_prompt(context))


    def ask(self, question: str):
        if not self.retriever:
            raise RuntimeError("Paper not ingested")

        # 1️⃣ Check cache
        past_qas = get_all_questions(self.paper_id)
        if past_qas:
            q_emb = self.embedder.embed([question])[0]
            past_embs = self.embedder.embed([qa.question for qa in past_qas])

            sims = cosine_sim(q_emb, np.array(past_embs))

            best_idx = sims.argmax()
            if sims[best_idx] > 0.85:
                return {
                    "answer": past_qas[best_idx].answer,
                    "response_time_sec": 0.01,
                    "cached": True
                }


        context = trim_context(
            self.retriever.retrieve(question, TOP_K)
        )

        with Timer() as t:
            answer = self.chat_llm.generate(
                ask_prompt(context, question)
            )

        save_qa(self.paper_id, question, answer)

        return {
            "answer": answer,
            "response_time_sec": t.elapsed,
            "cached": False
        }
