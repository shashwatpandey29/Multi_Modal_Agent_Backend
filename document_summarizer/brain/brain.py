import os
from threading import Lock
from typing import Dict, List

import numpy as np

from document_summarizer.brain.logger import get_logger
from document_summarizer.brain.ingestion.section_parser import split_into_sections
from document_summarizer.brain.ingestion.chunker import chunk_text
from document_summarizer.brain.ingestion.document_loader import load_document_text
from document_summarizer.brain.embeddings.embedder import Embedder
from document_summarizer.brain.embeddings.vector_store import VectorStore
from document_summarizer.brain.retrieval.retriever import Retriever
from document_summarizer.brain.llm.provider_llm import ProviderLLM
from document_summarizer.brain.prompts.teacher import teacher_prompt
from document_summarizer.brain.prompts.ask import ask_prompt
from document_summarizer.brain.prompts.analysis import full_analysis_prompt
from document_summarizer.brain.persistence.paper_store import (
    get_paper_by_filename,
    save_paper_and_chunks,
    load_chunks,
)
from document_summarizer.brain.persistence.vector_store import save_index, load_index
from document_summarizer.brain.persistence.qa_store import (
    save_qa,
    get_recent_questions,
    get_exact_answer,
)
from document_summarizer.brain.persistence.analysis_store import get_analysis, save_analysis
from document_summarizer.brain.utils.context import trim_context
from document_summarizer.brain.config import TOP_K
from document_summarizer.brain.utils.timer import Timer


_RETRIEVER_CACHE: Dict[int, Retriever] = {}
_RETRIEVER_CACHE_LOCK = Lock()

_FILE_LOCKS: Dict[str, Lock] = {}
_FILE_LOCKS_GUARD = Lock()


def cosine_sim(a, b):
    """
    a: (d,)
    b: (n, d)
    returns: (n,)
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(b, a)


def _get_file_lock(filename: str) -> Lock:
    with _FILE_LOCKS_GUARD:
        lock = _FILE_LOCKS.get(filename)
        if lock is None:
            lock = Lock()
            _FILE_LOCKS[filename] = lock
    return lock


def _get_cached_retriever(paper_id: int):
    with _RETRIEVER_CACHE_LOCK:
        return _RETRIEVER_CACHE.get(paper_id)


def _set_cached_retriever(paper_id: int, retriever: Retriever):
    with _RETRIEVER_CACHE_LOCK:
        _RETRIEVER_CACHE[paper_id] = retriever


def _parse_full_analysis(full_analysis: str):
    summary = ""
    key_learnings = ""
    limitations = ""
    contributions = ""

    try:
        summary = full_analysis.split("=== KEY LEARNINGS ===")[0]
        summary = summary.replace("=== SUMMARY ===", "").strip()

        part2 = full_analysis.split("=== KEY LEARNINGS ===")[1]
        key_learnings = part2.split("=== MAIN CONTRIBUTIONS ===")[0].strip()

        part3 = part2.split("=== MAIN CONTRIBUTIONS ===")[1]
        contributions = part3.split("=== LIMITATIONS ===")[0].strip()
        limitations = part3.split("=== LIMITATIONS ===")[1].strip()
    except Exception:
        summary = full_analysis
        key_learnings = "Parsing failed"
        contributions = "Parsing failed"
        limitations = "Parsing failed"

    return summary, key_learnings, limitations, contributions


class ResearchBrain:
    def __init__(self):
        self.logger = get_logger("ResearchBrain")
        self.embedder = Embedder()
        self.retriever = None
        self.paper_id = None
        self.chat_llm = ProviderLLM(
            use_case="chat",
            system_prompt="You are an academic assistant. Answer only from provided document context.",
        )
        self.analysis_llm = ProviderLLM(
            use_case="analysis",
            system_prompt="You are an academic research analyzer. Return structured, factual outputs only.",
        )

    def load(self, paper_id: int):
        self.paper_id = paper_id
        self._load_from_storage()

    def ingest(self, file_path: str):
        filename = os.path.basename(file_path)

        # Avoid duplicate concurrent ingestion for the same filename.
        with _get_file_lock(filename):
            existing = get_paper_by_filename(filename)
            if existing:
                self.paper_id = existing.id
                self._load_from_storage()
                self.logger.info("Paper already ingested, loaded from storage")
                return {
                    "paper_id": self.paper_id,
                    "analysis_time_sec": 0,
                    "cached": True,
                }

            self.logger.info("Ingesting paper for the first time")
            text = load_document_text(file_path)
            if not text or not text.strip():
                raise ValueError("Document contains no readable text")

            sections = split_into_sections(text)
            section_chunks = {}
            for section, content in sections.items():
                chunks = chunk_text(content)
                if chunks:
                    section_chunks[section] = chunks

            if not section_chunks:
                raise ValueError("No valid chunks created from document")

            self.paper_id = save_paper_and_chunks(filename, section_chunks)

            all_chunks = []
            for section, chunks in section_chunks.items():
                for chunk in chunks:
                    all_chunks.append(f"[{section.upper()}]\n{chunk}")

            embeddings = self.embedder.embed(all_chunks)
            store = VectorStore(embeddings.shape[1])
            store.add(embeddings, all_chunks)
            save_index(store.index, self.paper_id)

            self.logger.info("Generating precomputed summary and fact points")
            with Timer() as timer:
                try:
                    max_chunks_for_summary = 3
                    max_chars = 4000
                    important_chunks = all_chunks[:max_chunks_for_summary]
                    combined_text = "\n\n".join(important_chunks)[:max_chars]
                    full_analysis = self.analysis_llm.generate(full_analysis_prompt(combined_text))
                except Exception as exc:
                    self.logger.error(f"Error generating analysis: {exc}")
                    full_analysis = "Analysis generation failed due to timeout."

            summary, key_learnings, limitations, contributions = _parse_full_analysis(full_analysis)

            save_analysis(
                self.paper_id,
                summary,
                key_learnings,
                limitations,
                contributions,
                timer.elapsed,
            )

            self.retriever = Retriever(self.embedder, store)
            _set_cached_retriever(self.paper_id, self.retriever)

            self.logger.info("Ingestion complete (persisted + analyzed)")
            return {
                "paper_id": self.paper_id,
                "analysis_time_sec": timer.elapsed,
                "cached": False,
            }

    def _load_from_storage(self):
        cached = _get_cached_retriever(self.paper_id)
        if cached:
            self.retriever = cached
            return

        chunks = load_chunks(self.paper_id)
        texts = [f"[{c.section.upper()}]\n{c.text}" for c in chunks]
        if not texts:
            raise RuntimeError("Paper has no chunks in storage")

        index = load_index(self.paper_id)
        if index is None:
            self.logger.warning("FAISS index missing, rebuilding from stored chunks")
            embeddings = self.embedder.embed(texts)
            store = VectorStore(embeddings.shape[1])
            store.add(embeddings, texts)
            save_index(store.index, self.paper_id)
        else:
            sample_vector = self.embedder.embed([texts[0]])
            current_dim = int(sample_vector.shape[1]) if sample_vector.ndim == 2 else int(sample_vector.shape[0])

            if index.d != current_dim:
                self.logger.warning(
                    "Embedding dimension changed (index=%s, provider=%s). Rebuilding index.",
                    index.d,
                    current_dim,
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
        _set_cached_retriever(self.paper_id, self.retriever)
        self.logger.info("Loaded paper from storage")

    def summarize(self):
        analysis = get_analysis(self.paper_id)
        return analysis.summary if analysis else "Summary not available"

    def get_fact_points(self) -> List[str]:
        analysis = get_analysis(self.paper_id)
        if not analysis or not analysis.key_learnings:
            return []

        points = []
        for raw_line in analysis.key_learnings.splitlines():
            line = raw_line.strip().lstrip("-*").strip()
            if line:
                points.append(line)
        return points

    def teach(self):
        context = "\n".join(self.retriever.retrieve("explain methodology", TOP_K))
        return self.analysis_llm.generate(teacher_prompt(context))

    def ask(self, question: str):
        if not self.retriever:
            raise RuntimeError("Paper not ingested")

        exact = get_exact_answer(self.paper_id, question)
        if exact:
            return {
                "answer": exact.answer,
                "response_time_sec": 0.01,
                "cached": True,
            }

        past_qas = get_recent_questions(self.paper_id, limit=20)
        if past_qas:
            q_emb = self.embedder.embed([question])[0]
            past_embs = self.embedder.embed([qa.question for qa in past_qas])
            sims = cosine_sim(q_emb, np.array(past_embs))
            best_idx = sims.argmax()

            if sims[best_idx] > 0.9:
                return {
                    "answer": past_qas[best_idx].answer,
                    "response_time_sec": 0.01,
                    "cached": True,
                }

        context = trim_context(self.retriever.retrieve(question, TOP_K))

        with Timer() as timer:
            answer = self.chat_llm.generate(ask_prompt(context, question))

        save_qa(self.paper_id, question, answer)
        return {
            "answer": answer,
            "response_time_sec": timer.elapsed,
            "cached": False,
        }
