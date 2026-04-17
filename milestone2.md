# Milestone 2 — Agentic AI Contract Analyst

**Phase: Milestone 2 · LangGraph Agentic Pipeline + Cloud/Local LLM Reasoning**

> 📄 This document covers **Milestone 2** exclusively — the Agentic AI layer built on top of the Milestone 1 ML model.
> For the original ML classification system, see [`README.md`](./README.md).

---

## Table of Contents

- [What Changed in Milestone 2](#what-changed-in-milestone-2)
- [Agentic Architecture Overview](#agentic-architecture-overview)
- [AgentState — Shared Pipeline State](#agentstate--shared-pipeline-state)
- [LangGraph Pipeline — Node by Node](#langgraph-pipeline--node-by-node)
  - [Node 1: classify — ML Risk Classification](#node-1-classify--ml-risk-classification)
  - [Node 2: research — RAG Legal Best-Practice Retrieval](#node-2-research--rag-legal-best-practice-retrieval)
  - [Node 3: reason — LLM Deep Analysis](#node-3-reason--llm-deep-analysis)
- [Cloud LLM Client — 3-Level Fallback Chain](#cloud-llm-client--3-level-fallback-chain)
- [Offline Mode — Ollama Local LLM](#offline-mode--ollama-local-llm)
- [RAG Knowledge Base Retriever](#rag-knowledge-base-retriever)
- [Shared Prompt System & Analysis Schema](#shared-prompt-system--analysis-schema)
- [Report Builder](#report-builder)
- [Module Reference](#module-reference)
- [System Architecture & Code Structure](#system-architecture--code-structure)
- [Environment Configuration](#environment-configuration)
- [Installation & Setup](#installation--setup)
- [Running the Application](#running-the-application)
- [Team Contribution](#-team-contribution)

---

## What Changed in Milestone 2

Milestone 2 transforms the system from a **static ML classifier** into a **fully agentic AI system**. The trained Logistic Regression model from Milestone 1 is now the *entry point* of a multi-step reasoning pipeline, not the final output.

### Milestone 1 vs Milestone 2

| Capability | Milestone 1 | Milestone 2 |
|---|---|---|
| Contract parsing | ✅ PDF/text upload | ✅ PDF/text upload |
| Clause segmentation | ✅ Regex-based | ✅ Regex-based (unchanged) |
| Risk classification | ✅ TF-IDF + Logistic Regression | ✅ Same model (entry point) |
| Confidence thresholding | ✅ User-adjustable | ✅ User-adjustable |
| Legal best-practice lookup | ❌ | ✅ TF-IDF RAG over curated KB |
| Plain-English explanation | ❌ | ✅ LLM-generated (8 structured fields) |
| Safer clause rewrite | ❌ | ✅ Complete legal clause replacement |
| Negotiation tips | ❌ | ✅ 2–3 actionable numbered tips |
| Cloud LLM fallback chain | ❌ | ✅ OpenRouter × 2 + Groq |
| Offline LLM support | ❌ | ✅ Local Ollama (auto-starts daemon) |
| Structured JSON report | ❌ | ✅ Full JSON output |
| Professional Markdown report | ❌ | ✅ Executive-ready Markdown |

Each flagged clause now receives:

- ✅ ML risk label + confidence score (from M1 model)
- ✅ Retrieved legal best-practice context (RAG)
- ✅ Deep LLM analysis across **8 structured dimensions**
- ✅ Actionable mitigation strategies and safer clause rewrites
- ✅ A professional, structured Markdown report

---

## Agentic Architecture Overview

```
User uploads contract (PDF / plain text)
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│                    LangGraph StateGraph                       │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │   classify   │───▶│   research   │───▶│    reason    │   │
│  │  (ML node)   │    │  (RAG node)  │    │  (LLM node)  │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│                                                              │
│       AgentState flows through each node, accumulating      │
│       data at each step without losing prior context         │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
  structured_report (JSON)  +  markdown_report (rendered)
```

The pipeline is orchestrated by **LangGraph** using a typed `AgentState` (`TypedDict`). Each node reads from and writes back to this shared state object, giving the pipeline:

- Full **traceability** — what each node added is inspectable
- **Deterministic error propagation** — an error in `classify` flows through cleanly without crashing `reason`
- **Cached graph compilation** — the compiled `StateGraph` is cached with `@lru_cache(maxsize=2)` to avoid re-compilation on every request

---

## AgentState — Shared Pipeline State

**File:** `contract_agent/workflow.py`

```python
class AgentState(TypedDict, total=False):
    raw_text: str                        # Full original contract text
    confidence_threshold: float          # Min ML confidence to include a clause
    mode: str                            # "online" | "offline"
    contract_overview: str               # First 500 chars – used in report header
    flagged_clauses: list[dict]          # ML-classified clauses above threshold
    researched: list[dict]               # Clauses + retrieved KB best practices
    clause_assessments: list[dict]       # Clauses + full 8-key LLM analysis
    structured_report: dict              # Final JSON report (all data)
    markdown_report: str                 # Final rendered Markdown (for UI display)
    error: str | None                    # Pipeline error — propagates gracefully
```

`total=False` means all keys are optional — nodes only write back the keys they produce.

---

## LangGraph Pipeline — Node by Node

### Node 1: `classify` — ML Risk Classification

**File:** `contract_agent/workflow.py` → `_classify_node()`

This node loads the Milestone 1 trained model and processes every clause extracted from the uploaded contract.

**Execution Steps:**

1. Loads `models/best_model.pkl` via `ml_utils.load_sklearn_pipeline()` — the TF-IDF + Logistic Regression scikit-learn pipeline. Cached with `@lru_cache` so it is only deserialized once per session.
2. Calls `text_utils.segment_clauses()` to split raw text into individual clauses by double-newline boundaries and numbered/lettered sub-paragraph markers.
3. Calls `text_utils.clean_text()` on each clause — lowercases, normalises whitespace, strips non-legal punctuation.
4. Runs `model.predict([cleaned])` → risk label (`High`, `Medium`, `Low`)
5. Runs `model.predict_proba([cleaned])` → confidence score (`0.0–1.0`)
6. Applies the **confidence threshold** — clauses below the threshold are silently discarded.
7. Returns `flagged_clauses` and a `contract_overview` (first 500 characters of the document).

**Output shape per clause:**

```python
{
    "clause_text": "The Vendor shall not be liable for any indirect...",
    "risk_level":  "High",
    "confidence":  0.91
}
```

**Error handling:** If `best_model.pkl` is not found, the node returns `{"error": "Trained model not found at models/best_model.pkl."}` and the remaining nodes handle this gracefully.

---

### Node 2: `research` — RAG Legal Best-Practice Retrieval

**File:** `contract_agent/workflow.py` → `_research_node()`  
**Retriever:** `contract_agent/kb_retriever.py` → `LegalPracticeRetriever`

For every flagged clause, this node retrieves the **top-3 most relevant legal best-practice excerpts** from a curated local knowledge base at `data/legal_best_practices.json`.

**Retrieval Algorithm:**

```
clause_text
    │
    ├──► _infer_topic_hint()  ──► regex match → topic label
    │       (e.g. "termination", "indemnification")
    │
    └──► TfidfVectorizer.transform()
              │
              ▼
         cosine_similarity vs full KB matrix
              │
              ▼
         argsort (descending) → ranked indices
              │
              ▼
    Merge: topic-boosted first → fill remaining with global top-k
              │
              ▼
    Return top_k records (default: 3)
```

**Topic Hint Injection** — `_infer_topic_hint()` uses a regex rule-engine to detect semantic topic from the clause text:

| Topic Detected | Trigger Pattern |
|---|---|
| `termination` | `terminate`, `termination`, `for cause`, `convenience` |
| `indemnification` | `indemnif` |
| `limitation_of_liability` | `liability`, `consequential`, `incidental` |
| `confidentiality` | `confidential`, `nda`, `non-disclosure` |
| `intellectual_property` | `intellectual property`, `work for hire`, `license` |
| `payment` | `payment`, `invoice`, `fees` |
| `representations_warranties` | `represents`, `representation`, `warrant` |
| `force_majeure` | `force majeure` |
| `data_protection` | `data protection`, `gdpr`, `processor`, `controller` |

When a topic is detected, records tagged with that topic are ranked to the top — ensuring legal context is always domain-relevant, not just statistically similar.

**Node output per clause:**

```python
{
    "clause_text": "...",
    "risk_level": "High",
    "confidence": 0.91,
    "best_practices": [
        {"score": 0.74, "id": "limitation_001", "topic": "limitation_of_liability",
         "title": "Mutual Cap on Liability", "text": "Industry standard..."},
        ...
    ]
}
```

**Error skip:** If `state.get("error")` is set, this node returns `{}` immediately without processing.

---

### Node 3: `reason` — LLM Deep Analysis

**File:** `contract_agent/workflow.py` → `_reason_node()`

This is the **core intelligence node**. For every researched clause, it dispatches to the appropriate LLM backend based on `state["mode"]`:

```python
if mode == "offline":
    analysis = analyze_clause_with_ollama(...)   # Local Ollama
else:
    analysis = analyze_clause_with_cloud(...)    # Cloud (OpenRouter + Groq fallback)
```

Each LLM call returns the **8-key analysis dict** (see [Analysis Schema](#shared-prompt-system--analysis-schema) below).

After all clauses are assessed, this node assembles the final outputs:

```python
structured = build_structured_report(contract_overview, assessments)  # → JSON dict
md         = render_markdown_report(structured)                        # → Markdown str
```

**LLM error isolation:** If a single clause's LLM call raises any exception, `safe_parse_analysis("{}", fallback_text=...)` returns a graceful placeholder dict — the pipeline never crashes due to a single clause failure. Other clauses continue processing normally.

**Error state output:** If `state["error"]` is set from a prior node (e.g., model not found), this node returns a minimal error report with the disclaimer attached:

```markdown
## Error

Trained model not found at models/best_model.pkl.

## Disclaimer

⚠️ This report is generated by an academic AI research prototype...
```

---

## Cloud LLM Client — 3-Level Fallback Chain

**File:** `contract_agent/cloud_client.py`

This module implements a **resilient, provider-agnostic fallback chain** ensuring continuous LLM availability even if API keys are exhausted, rate-limited, or invalid.

### Fallback Priority Diagram

```
┌─────────────────────────────────────┐
│  Attempt 1: OpenRouter Key 1        │  OPENROUTER_API_KEY_1
│  _call_openrouter(key_1, model, …)  │
└──────────────┬──────────────────────┘
               │ Exception raised?
               ▼
┌─────────────────────────────────────┐
│  Attempt 2: OpenRouter Key 2        │  OPENROUTER_API_KEY_2
│  _call_openrouter(key_2, model, …)  │
└──────────────┬──────────────────────┘
               │ Exception raised?
               ▼
┌─────────────────────────────────────┐
│  Attempt 3: Groq (Final Fallback)   │  GROQ_API_KEY
│  _call_groq(groq_key, model, …)     │
└──────────────┬──────────────────────┘
               │ Exception raised?
               ▼
     safe_parse_analysis("{}", fallback_text=error_msg)
     ← Never crashes; always returns usable dict →
```

### Provider Configuration

| Provider | Env Variable | Default Model | Max Tokens | Timeout |
|---|---|---|---|---|
| OpenRouter (Primary) | `OPENROUTER_API_KEY_1` | `openai/gpt-oss-120b` | 1800 | 60s |
| OpenRouter (Secondary) | `OPENROUTER_API_KEY_2` | `openai/gpt-oss-120b` | 1800 | 60s |
| Groq (Final Fallback) | `GROQ_API_KEY` | `llama-3.3-70b-versatile` | 1500 | 30s |

Both providers share the same interface — Groq is called via `OpenAI(base_url="https://api.groq.com/openai/v1")` so no additional SDK is needed.

### LLM Call Settings (applied to all providers)

```python
temperature       = 0.15                        # Low → consistent, minimal hallucination
response_format   = {"type": "json_object"}     # Forces valid JSON output
```

### Logging

All attempts are logged to the standard Python logger:

```
✅ OpenRouter Key 1 succeeded.
⚠️  OpenRouter Key 1 failed: 429 Too Many Requests
✅ OpenRouter Key 2 succeeded (Key 1 fallback).
```

### Health Check

Used by the Streamlit sidebar to show live provider status indicators:

```python
from contract_agent.cloud_client import check_cloud_health

check_cloud_health()
# → {"openrouter": True, "groq": False}
```

Returns `True` for a provider if **at least one** of its keys is configured in the environment.

---

## Offline Mode — Ollama Local LLM

**File:** `contract_agent/ollama_client.py`

The offline mode runs a fully local LLM via [Ollama](https://ollama.com/) — **no internet connection required** after the initial model download.

### Auto-Start Daemon

The client checks if Ollama is running at startup and automatically launches it if not:

```python
# Step 1: Health check
GET http://localhost:11434/api/tags  →  200 OK?

# Step 2: If not running, auto-start
subprocess.Popen(["ollama", "serve"], start_new_session=True)

# Step 3: Poll every 500ms, up to 10 seconds
for _ in range(20):
    time.sleep(0.5)
    if _is_ollama_running(base_url):
        return True   # ← daemon is up
```

### Configuration

| Env Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL (change for remote server) |
| `OLLAMA_MODEL` | `qwen3.5:2b` | Local model to use (must be pulled first) |

### Inference Parameters

```python
payload = {
    "model":   "qwen3.5:2b",
    "stream":  False,
    "options": {
        "temperature": 0.15,
        "num_predict": 2048,
        "top_p":       0.9,
    }
}
```

Timeout is **180 seconds** to handle slow first-load of quantised local models.

### Pulling a Model

```bash
ollama pull qwen3.5:2b        # Recommended — fast, 2B param, good JSON output
ollama pull llama3.2:3b       # Alternative option
```

### Health Check

```python
from contract_agent.ollama_client import check_ollama_health

check_ollama_health()
# →
# {
#     "reachable":        True,
#     "model":            "qwen3.5:2b",
#     "available_models": ["qwen3.5:2b", "llama3.2:3b"],
#     "auto_started":     False
# }
```

---

## RAG Knowledge Base Retriever

**File:** `contract_agent/kb_retriever.py`

The `LegalPracticeRetriever` is a lightweight, fully local **Retrieval-Augmented Generation** component — no vector database or external embedding API required.

### Knowledge Base Format

Loaded from `data/legal_best_practices.json` — cached with `@lru_cache` so the file is only read once:

```json
[
  {
    "id":    "termination_001",
    "topic": "termination",
    "title": "Termination for Convenience",
    "text":  "Industry standard allows either party to terminate with 30 days written notice..."
  },
  ...
]
```

### How the TF-IDF KB Index Is Built

When `LegalPracticeRetriever` is instantiated, it immediately builds its in-memory index:

```python
corpus = [f"{r['title']} {r['topic']} {r['text']}" for r in records]
self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=2000, stop_words="english")
self._matrix     = self._vectorizer.fit_transform(corpus)
```

### Retrieval Call Flow

```python
retriever = LegalPracticeRetriever(top_k=3)
results   = retriever.retrieve("The Vendor shall not be liable for any consequential damages...")

# → [
#     {"score": 0.81, "id": "limitation_001", "topic": "limitation_of_liability", ...},
#     {"score": 0.63, "id": "limitation_003", "topic": "limitation_of_liability", ...},
#     {"score": 0.41, "id": "general_009",    "topic": "general", ...},
#   ]
```

---

## Shared Prompt System & Analysis Schema

**File:** `contract_agent/_shared_prompt.py`

This module **centralises all prompt engineering** so that both cloud and Ollama clients produce structurally identical output — regardless of which LLM backend is used.

### Analysis Schema — 8 Keys

Every LLM response must conform to this schema:

| Key | Type | Description |
|---|---|---|
| `plain_english_summary` | `str` | What the clause says in plain language (2–3 sentences) |
| `what_makes_it_risky` | `str` | Specific legal danger with real-world consequence (2–4 sentences) |
| `who_bears_the_risk` | `str` | Exactly one of: `"Client"`, `"Vendor"`, `"Both"`, `"Third Party"` |
| `severity_rationale` | `str` | Why the ML model rated this clause High/Medium/Low (1–2 sentences) |
| `industry_standard_practice` | `str` | What a fair, balanced version of this clause looks like (2–3 sentences) |
| `negotiation_tips` | `str` | 2–3 specific negotiation points as a numbered list |
| `safer_rewrite` | `str` | Complete, ready-to-use replacement clause in formal legal style |
| `action_required` | `str` | Exactly one of: `"Remove Clause"`, `"Negotiate Terms"`, `"Seek Legal Review"`, `"Accept with Caution"` |

### System Prompt Design

`build_system_prompt(risk_label, confidence)` constructs the LLM system message with:

- **Persona**: _"Senior Legal Risk Analyst AI with expertise in commercial contract law"_
- **Context injection**: ML risk label (e.g., `HIGH RISK`) and confidence percentage (e.g., `91%`) are embedded directly in the prompt
- **Audience framing**: Output is targeted at a business executive, not a lawyer — all explanations must be practical and non-legal-jargon
- **Quality rules**: Every output must be Specific, Practical, Actionable, and Balanced
- **Anti-hallucination guardrails**:
  - Do NOT invent citations — only reference provided KB excerpts
  - Do NOT use "consult a lawyer" as the only advice
  - `safer_rewrite` must be a complete formal clause, not a summary

### User Message Structure

`build_user_message(clause_text, best_practices)` builds the user turn:

```
## CLAUSE TO ANALYSE

<clause text>

## RETRIEVED LEGAL BEST-PRACTICE EXCERPTS

[EXCERPT 1 | Topic: limitation_of_liability]
Title: Mutual Cap on Liability
Standard: Industry standard limits liability to the total fees paid...

---

[EXCERPT 2 | Topic: limitation_of_liability]
...
```

### `safe_parse_analysis` — Never Raises

The JSON parser handles all edge cases defensively:

| Problem | Recovery Strategy |
|---|---|
| `<think>...</think>` tags (Qwen3/DeepSeek) | Stripped via `re.sub(r"<think>.*?</think>", ...)` |
| Markdown fences (` ```json `) | Stripped via regex |
| Invalid JSON | Regex-search for `{...}` substring, retry `json.loads` |
| Complete parse failure | Returns `{}`, filled with defaults |
| Old 3-key schema (`legal_concern`, `comparison_to_best_practice`) | Automatically remapped to new 8-key schema |
| Missing keys | All 8 keys backfilled with sensible defaults |

**Default fallbacks if a key is missing:**

```python
"plain_english_summary":      fallback_text or "Analysis could not be completed."
"what_makes_it_risky":        "Analysis could not be completed."
"who_bears_the_risk":         "Unknown"
"severity_rationale":         "Model flagged this clause based on statistical patterns."
"industry_standard_practice": "No best-practice data available."
"negotiation_tips":           "1. Consult a legal professional before signing."
"safer_rewrite":              "Clause requires manual legal review."
"action_required":            "Seek Legal Review"
```

---

## Report Builder

**File:** `contract_agent/report.py`

The report module converts raw clause assessments into two output formats: a **structured JSON dict** and a **professional Markdown report**.

### 1. Structured JSON Report — `build_structured_report()`

```json
{
  "contract_overview": "This Agreement is entered into as of...",
  "risk_severity_breakdown": {
    "High": 3,
    "Medium": 2,
    "Low": 1
  },
  "flagged_clauses_and_mitigation": [
    {
      "clause_text":                 "The Vendor shall not be liable...",
      "model_risk_level":            "High",
      "model_confidence":            0.91,
      "retrieved_practice_ids":      ["limitation_001", "limitation_003"],
      "plain_english_summary":       "This clause removes all financial accountability...",
      "what_makes_it_risky":         "Without a liability cap, your vendor can cause...",
      "who_bears_the_risk":          "Client",
      "severity_rationale":          "Clauses absolving liability are statistically...",
      "industry_standard_practice":  "A balanced clause caps liability at 12 months...",
      "negotiation_tips":            "1. Request a mutual cap... 2. Define consequential...",
      "safer_rewrite":               "Notwithstanding anything to the contrary...",
      "action_required":             "Negotiate Terms"
    }
  ],
  "disclaimer": "⚠️ Legal Disclaimer: ..."
}
```

### 2. Markdown Report — `render_markdown_report()`

The Markdown renderer outputs a professional, client-ready document with these sections:

#### Overall Risk Signal Logic

| Condition | Signal Displayed |
|---|---|
| `High ≥ 2` | 🔴 **HIGH RISK DOCUMENT** — Multiple critical issues detected |
| `High == 1` | 🟠 **ELEVATED RISK** — At least one critical clause requires attention |
| `Medium ≥ 2` | 🟡 **MODERATE RISK** — Several clauses need review |
| Otherwise | 🟢 **LOWER RISK** — Minor concerns only. Standard review recommended |

#### Report Sections

```
# 📋 Contract Risk Assessment Report
## 📌 Executive Summary
   > contract_overview
   Overall Signal: 🔴 HIGH RISK DOCUMENT
   ### Risk Breakdown Table (High / Medium / Low counts)

## 🔍 Clause-by-Clause Analysis
   (Ordered: High → Medium → Low)
   ### 🔴 Clause 1 — HIGH RISK
      ML Confidence | Risk Bearer | Action
      #### 📄 Original Clause Text
      #### 💬 What This Clause Says (Plain English)
      #### ⚠️ Why This Is Risky
      #### 📊 Why It Was Rated High
      #### 📚 Industry Standard Practice
      #### 🤝 Negotiation Tips
      #### ✅ Suggested Safer Wording (code block)

## 📋 Action Summary Table
   | # | Risk | Action Required | Clause Preview |

## ⚠️ Disclaimer
```

#### Action & Risk Icons

| Action | Icon | Risk Level | Icon |
|---|---|---|---|
| Remove Clause | 🚫 | High | 🔴 |
| Negotiate Terms | 🤝 | Medium | 🟠 |
| Seek Legal Review | ⚖️ | Low | 🟢 |
| Accept with Caution | ⚠️ | — | — |

---

## Module Reference

```
contract_agent/
│
├── __init__.py            Exports run_hybrid_agent_pipeline as the public API
│
├── workflow.py            LangGraph StateGraph definition
│                          • _classify_node()   – ML risk classification (Node 1)
│                          • _research_node()   – RAG retrieval (Node 2)
│                          • _reason_node()     – LLM analysis + report build (Node 3)
│                          • _build_compiled_graph()  – cached graph compiler
│                          • run_hybrid_agent_pipeline()  – public entry point
│
├── cloud_client.py        3-level cloud LLM fallback chain
│                          • _call_openrouter() – OpenRouter provider
│                          • _call_groq()       – Groq provider
│                          • analyze_clause_with_cloud()  – public function
│                          • check_cloud_health()         – sidebar health check
│
├── ollama_client.py       Offline Ollama LLM integration
│                          • _is_ollama_running()   – daemon health check
│                          • _start_ollama_daemon() – auto-start subprocess
│                          • analyze_clause_with_ollama() – public function
│                          • check_ollama_health()        – sidebar health check
│
├── openrouter_client.py   Legacy single-key OpenRouter client
│                          (kept for backward compatibility; cloud_client.py supersedes it)
│
├── kb_retriever.py        TF-IDF RAG retriever over legal_best_practices.json
│                          • _infer_topic_hint()     – regex topic detector
│                          • _load_kb_records()      – cached JSON loader
│                          • LegalPracticeRetriever  – retriever class
│                            └── retrieve(clause_text) → list[dict]
│
├── _shared_prompt.py      Centralised prompt engineering & output schema
│                          • ANALYSIS_SCHEMA        – 8-key schema definition
│                          • build_system_prompt()  – LLM system message
│                          • build_user_message()   – LLM user message (clause + KB)
│                          • safe_parse_analysis()  – fault-tolerant JSON parser
│
├── report.py              Report generation
│                          • build_structured_report()  – JSON report builder
│                          • render_markdown_report()   – Markdown renderer
│                          • DISCLAIMER                 – legal disclaimer string
│
├── text_utils.py          Text preprocessing utilities
│                          • clean_text(text)           – normalize clause text
│                          • segment_clauses(text)      – split into clauses
│                          • get_summary(text, limit)   – truncate with ellipsis
│
└── ml_utils.py            ML model loader
                           • load_sklearn_pipeline()    – cached pkl loader
```

---

## System Architecture & Code Structure

```text
Contract-Risk-Classification/
├── CUAD_Dataset/                  # Raw CUAD v1 dataset (Milestone 1)
├── data/
│   └── legal_best_practices.json  # ✨ NEW — Curated KB for RAG retrieval
├── models/
│   └── best_model.pkl             # M1 deployed pipeline (TF-IDF + LogReg)
├── src/                           # Jupyter Notebooks — M1 data pipeline
│   ├── 1_inspect.ipynb
│   ├── 2_data_preprocessing.ipynb
│   ├── 3_feature_engineering.ipynb
│   ├── 4_train.ipynb
│   └── 5_evaluate.ipynb
│
├── contract_agent/                # ✨ NEW in Milestone 2 — Agentic AI package
│   ├── __init__.py
│   ├── workflow.py
│   ├── cloud_client.py
│   ├── ollama_client.py
│   ├── openrouter_client.py
│   ├── kb_retriever.py
│   ├── _shared_prompt.py
│   ├── report.py
│   ├── text_utils.py
│   └── ml_utils.py
│
├── app.py                         # Streamlit UI (updated for M2 agentic pipeline)
├── test_pipeline.py               # ✨ NEW — End-to-end pipeline smoke test
├── .env.example                   # ✨ NEW — API key configuration template
├── requirements.txt               # Updated with langgraph, langchain-core, openai
├── README.md                      # Milestone 1 documentation
└── milestone2.md                  # ← This file
```

---

## Environment Configuration

Copy `.env.example` to `.env` and fill in your keys. The system degrades gracefully if some keys are absent.

```bash
cp .env.example .env
```

### Complete `.env` Reference

```env
# ── Cloud Mode (Online) ──────────────────────────────────────────────────────

# OpenRouter — Primary API key (required for online mode)
OPENROUTER_API_KEY_1=sk-or-v1-...

# OpenRouter — Secondary key (optional; used automatically if Key 1 fails)
OPENROUTER_API_KEY_2=sk-or-v1-...

# OpenRouter model (default below; change to any model on openrouter.ai)
OPENROUTER_MODEL=openai/gpt-oss-120b

# Groq — Final fallback provider key (optional)
GROQ_API_KEY=gsk_...

# Groq model (default shown below)
GROQ_MODEL=llama-3.3-70b-versatile

# ── Offline Mode (Ollama) ────────────────────────────────────────────────────

# Ollama base URL (default shown; change for remote Ollama server)
OLLAMA_BASE_URL=http://localhost:11434

# Local model name — must be pulled: `ollama pull qwen3.5:2b`
OLLAMA_MODEL=qwen3.5:2b
```

> **Security:** Never commit your `.env` file. It is listed in `.gitignore`.

### New Dependencies (added in M2)

```
langgraph>=0.2.0          # Agentic StateGraph orchestration
langchain-core>=0.3.0     # LangChain base primitives used by LangGraph
openai>=1.40.0            # OpenAI-compatible client (used for OpenRouter & Groq)
python-dotenv>=1.0.0      # .env file loader
requests>=2.31.0          # HTTP client for Ollama REST API
```

---

## Installation & Setup

### Prerequisites

- Python **3.10+**
- For **online mode**: at least one valid API key (OpenRouter or Groq)
- For **offline mode**: [Ollama](https://ollama.com/) installed locally

### 1. Clone the repository

```bash
git clone https://github.com/ashyou09/Contract-Risk-Classification.git
cd Contract-Risk-Classification
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
# Open .env in your editor and add your API keys
```

### 4. (Offline mode only) Pull the local model

```bash
# Install Ollama first: https://ollama.com/download
ollama pull qwen3.5:2b
```

### 5. Verify the pipeline (optional smoke test)

```bash
python test_pipeline.py
```

A successful run prints a summary of flagged clauses and the overall risk signal.

---

## Running the Application

```bash
streamlit run app.py
```

The Streamlit UI will open in your browser at `http://localhost:8501`.

### UI Features (after M2 upgrade)

From the sidebar you can:

- **Select mode**: Online (Cloud LLM) or Offline (Local Ollama)
- **Adjust confidence threshold**: slider to filter low-confidence ML predictions
- **View provider health**: live status of OpenRouter, Groq, and Ollama
- **Upload contract**: PDF or paste text directly

In the main panel:
- View the **ML risk breakdown** chart (from M1)
- Expand the **full agentic report** with clause-by-clause analysis
- Download the **structured JSON report**
- Copy the **Markdown report** for sharing

---

## 👥 Team Contribution

| Member | ID | Milestone 2 Responsibilities |
|---|---|---|
| **Ashutosh Singh** | 2401010109 | LangGraph agentic workflow design (`workflow.py`), `cloud_client.py` 3-level fallback chain, `kb_retriever.py` RAG system architecture, `_shared_prompt.py` schema design, Streamlit M2 UI integration |
| **Ranvendra Pratap Singh** | 2401010373 | `ollama_client.py` offline LLM integration, auto-start daemon implementation, offline mode end-to-end testing, Ollama model evaluation |
| **Shreya Suman** | 2401020068 | `report.py` JSON & Markdown report builder, executive summary logic, action summary table, overall risk signal algorithm |
| **Avishkar Meher** | 2401010116 | `text_utils.py` preprocessing utilities, `ml_utils.py` caching layer, `test_pipeline.py` smoke tests, `.env.example` and configuration documentation |

---

**Disclaimer:** This tool is an academic project designed for educational purposes to demonstrate LLM-augmented and ML-based contract analysis. It does **NOT** constitute professional legal advice. Always consult a qualified legal professional before signing, modifying, or rejecting any contract clause.

---

*Intelligent Contract Risk Analysis System · Phase 2 (Agentic AI) · Milestone 2*
