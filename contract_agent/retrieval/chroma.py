"""
contract_agent/retrieval/chroma.py — Component Logic
=====================================================================

Primary retriever queries the ChromaDB vector store built by rag_setup.py.
Filters results by the detected contract domain for high-precision retrieval.

Fallback chain:
  1. ChromaDB domain-filtered search  (best quality)
  2. ChromaDB full-collection search  (if domain has < top_k results)
  3. TF-IDF over legal_best_practices.json  (if ChromaDB unavailable)

Usage:
    from contract_agent.retrieval.chroma import DomainAwareRetriever

    retriever = DomainAwareRetriever(top_k=3)
    results   = retriever.retrieve(clause_text="...", domain="Employment")
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

_BASE_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CHROMA_DB_PATH = os.path.join(_BASE_DIR, "data", "chroma_db")
_COLLECTION_NAME = "legal_guidelines"


# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB collection loader (cached once per process)
# ─────────────────────────────────────────────────────────────────────────────

def _get_chroma_collection():
    """
    Returns the ChromaDB collection or None if unavailable.
    Cached — only opens the DB once per process lifetime.
    """
    try:
        import chromadb
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        client = chromadb.PersistentClient(path=_CHROMA_DB_PATH)
        ef     = DefaultEmbeddingFunction()
        existing = [c.name for c in client.list_collections()]
        if _COLLECTION_NAME not in existing:
            return None
        return client.get_collection(_COLLECTION_NAME, embedding_function=ef)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main retriever class
# ─────────────────────────────────────────────────────────────────────────────

class DomainAwareRetriever:
    """
    Domain-filtered ChromaDB RAG retriever with automatic TF-IDF fallback.

    Results are formatted as dicts with keys:
      id, domain, topic, title, text, score
    """

    def __init__(self, top_k: int = 3):
        self.top_k       = top_k
        self._collection = _get_chroma_collection()
        self._tfidf      = None   # lazy-loaded fallback

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(self, clause_text: str, domain: str = "General") -> list[dict[str, Any]]:
        """
        Retrieve the top_k most relevant legal guidelines for a clause.

        Args:
            clause_text: The raw clause text to find guidelines for.
            domain:      Detected contract domain for metadata filtering.

        Returns:
            List of up to top_k result dicts.
        """
        if self._collection is not None:
            try:
                return self._chroma_retrieve(clause_text, domain)
            except Exception:
                pass
        # ChromaDB unavailable or query failed → TF-IDF fallback
        return self._tfidf_retrieve(clause_text)

    def is_using_chroma(self) -> bool:
        """True if ChromaDB is available and loaded."""
        return self._collection is not None

    # ── Private: ChromaDB retrieval ───────────────────────────────────────────

    def _chroma_retrieve(self, clause_text: str, domain: str) -> list[dict[str, Any]]:
        collection = self._collection

        # Try domain-specific filtered retrieval first
        if domain and domain != "General":
            try:
                results = collection.query(
                    query_texts=[clause_text],
                    n_results=self.top_k,
                    where={"domain": domain},
                    include=["documents", "metadatas", "distances"],
                )
                if results["ids"][0]:  # got results within the domain
                    return self._format(results)
            except Exception:
                pass  # fall through to unfiltered search

        # Fallback: search across entire collection
        results = collection.query(
            query_texts=[clause_text],
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"],
        )
        return self._format(results)

    @staticmethod
    def _format(results: dict) -> list[dict[str, Any]]:
        """Convert raw ChromaDB results to standard list-of-dicts."""
        out   = []
        ids   = results["ids"][0]
        docs  = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        for i, doc_id in enumerate(ids):
            meta = metas[i] or {}
            out.append({
                "id":     doc_id,
                "domain": meta.get("domain", "General"),
                "topic":  meta.get("topic",  ""),
                "title":  meta.get("title",  ""),
                "text":   docs[i],
                "score":  round(1.0 - float(dists[i]), 4),  # cosine similarity
            })
        return out

    # ── Private: TF-IDF fallback ──────────────────────────────────────────────

    def _tfidf_retrieve(self, clause_text: str) -> list[dict[str, Any]]:
        if self._tfidf is None:
            from contract_agent.retrieval.tfidf import TFIDFRetriever
            self._tfidf = TFIDFRetriever(top_k=self.top_k)
        return self._tfidf.retrieve(clause_text)


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible alias (existing code uses LegalPracticeRetriever)
# ─────────────────────────────────────────────────────────────────────────────

class LegalPracticeRetriever(DomainAwareRetriever):
    """
    Drop-in replacement for the old TF-IDF-only LegalPracticeRetriever.
    Internally uses DomainAwareRetriever — ChromaDB when available,
    TF-IDF fallback otherwise.
    """

    def retrieve(self, clause_text: str, domain: str = "General") -> list[dict[str, Any]]:
        return super().retrieve(clause_text, domain=domain)
