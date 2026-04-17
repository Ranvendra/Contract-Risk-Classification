"""
contract_agent/retrieval/tfidf.py — Component Logic
===============================================================

Backup retriever using the existing data/legal_best_practices.json.
Used automatically when ChromaDB is unavailable (e.g., before rag_setup.py
has been run, or in environments without sentence-transformers installed).
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _infer_topic_hint(text: str) -> str | None:
    t = text.lower()
    rules = [
        ("termination",            r"\bterminat(e|ion|able)\b|\bfor\s+cause\b|\bconvenience\b"),
        ("indemnification",        r"\bindemnif"),
        ("limitation_of_liability",r"\bliabilit|\bconsequential\b|\bincidental\b"),
        ("confidentiality",        r"\bconfidential\b|\bnda\b|\bnon-?disclosure\b"),
        ("intellectual_property",  r"\bintellectual\s+property\b|\bwork\s+for\s+hire\b|\blicense\b"),
        ("payment",                r"\bpayment\b|\binvoice\b|\bfees?\b"),
        ("representations_warranties", r"\brepresent(s|ation)|\bwarrant"),
        ("force_majeure",          r"\bforce\s+majeure\b"),
        ("data_protection",        r"\bdata\s+protection\b|\bgdpr\b|\bprocessor\b|\bcontroller\b"),
    ]
    for topic, pattern in rules:
        if re.search(pattern, t):
            return topic
    return None


@lru_cache(maxsize=1)
def _load_kb_records() -> list[dict]:
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(base_dir, "data", "legal_best_practices.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class TFIDFRetriever:
    """TF-IDF retriever over data/legal_best_practices.json (fallback mode)."""

    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        records = _load_kb_records()
        self._records = records
        corpus = [f"{r['title']} {r['topic']} {r['text']}" for r in records]
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=2000,
            stop_words="english",
        )
        self._matrix = self._vectorizer.fit_transform(corpus)

    def retrieve(self, clause_text: str, domain: str = "General") -> list[dict[str, Any]]:
        hint = _infer_topic_hint(clause_text)
        q    = self._vectorizer.transform([clause_text])
        sims = cosine_similarity(q, self._matrix).flatten()
        ranked = np.argsort(-sims)

        picked: list[dict] = []
        seen:   set[int]   = set()

        if hint:
            topic_indices = [
                i for i, rec in enumerate(self._records) if rec.get("topic") == hint
            ]
            topic_indices.sort(key=lambda i: float(sims[i]), reverse=True)
            for i in topic_indices:
                if len(picked) >= self.top_k:
                    break
                picked.append({"score": float(sims[i]), **self._records[i]})
                seen.add(i)

        for idx in ranked:
            idx = int(idx)
            if idx in seen:
                continue
            picked.append({"score": float(sims[idx]), **self._records[idx]})
            seen.add(idx)
            if len(picked) >= self.top_k:
                break

        return picked[: self.top_k]
