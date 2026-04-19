<img src="https://images.ctfassets.net/em6l9zw4tzag/oVfiswjNH7DuCb7qGEBPK/b391db3a1d0d3290b96ce7f6aacb32b0/python.png" height="60" alt="Python" />&nbsp;&nbsp;<img src="https://uxwing.com/wp-content/themes/uxwing/download/brands-and-social-media/streamlit-icon.png" height="60" alt="Streamlit" />&nbsp;&nbsp;<img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" height="60" alt="Scikit-Learn" />&nbsp;&nbsp;<img src="https://avatars.githubusercontent.com/u/126733545?s=80&v=4" height="60" alt="LangGraph" />&nbsp;&nbsp;<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original.svg" height="60" alt="Pandas" />

# Intelligent Contract Risk Analysis & Agentic Legal Assistance System

> **Final Integrated Submission — Milestone 1 + Milestone 2**

An AI-driven legal document intelligence platform that evolved from a classical Machine Learning contract risk classifier into a full **Agentic AI Legal Assistant** — combining supervised learning, retrieval-augmented generation, and multi-step LLM reasoning to analyze contracts, detect risky clauses, and generate structured mitigation reports.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Solution Overview](#solution-overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Project Evolution](#project-evolution-milestone-1--milestone-2)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Environment Configuration](#environment-configuration)
- [Running the Application](#running-the-application)
- [ML Model & Dataset](#ml-model--dataset)
- [Agentic Pipeline](#agentic-pipeline)
- [LLM Analysis Output Schema](#llm-analysis-output-schema)
- [System Design Decisions](#system-design-decisions)
- [Evaluation Metrics](#evaluation-metrics)
- [Team Contribution](#team-contribution)
- [Disclaimer](#disclaimer)

---

## Problem Statement

Legal professionals, startups, freelancers, and businesses sign contracts without fully understanding hidden risks such as:

- Unlimited or one-sided liability clauses
- Unfair indemnification terms
- Exploitative termination rights
- Confidentiality traps and NDA overreach
- Ambiguous payment and fee dispute clauses
- Intellectual property ownership conflicts

Manual contract review is **expensive**, **time-consuming**, and **inaccessible** to most individuals and small businesses.

---

## Solution Overview

We built the **Intelligent Contract Risk Analysis System** — a two-phase platform that automates contract review:

**Phase 1 (Milestone 1) — ML-Based Clause Classification**

Uses NLP + supervised machine learning to classify every clause as High Risk, Medium Risk, or Low Risk with a confidence score.

**Phase 2 (Milestone 2) — Agentic AI Legal Assistant**

Transforms the ML output into a full agentic reasoning engine that understands risks deeply, retrieves legal best practices, rewrites dangerous clauses, and generates executive-ready reports.

---

## Key Features

| Feature | Description |
|---|---|
| **PDF & Text Upload** | Upload `.pdf` or `.txt` contracts directly in the browser |
| **Clause Segmentation** | Automatically splits contracts into logical clauses using regex boundaries |
| **ML Risk Classification** | TF-IDF + Logistic Regression model trained on the CUAD dataset |
| **Confidence Threshold Slider** | Filter weak predictions dynamically in the UI |
| **RAG Legal Knowledge Retrieval** | Retrieves top-3 relevant legal best-practice excerpts per risky clause |
| **LangGraph Agentic Reasoning** | Multi-node StateGraph pipeline: Classify → Research → Reason |
| **Cloud LLM with 3-Level Fallback** | OpenRouter Key 1 → OpenRouter Key 2 → Groq (never fails) |
| **Offline Local LLM** | Runs fully on-device via Ollama (no internet after model download) |
| **Structured JSON Report** | Machine-readable full analysis per clause |
| **Markdown Executive Report** | Professional human-readable summary with risk signal and action items |
| **PDF Report Download** | Export the full analysis as a downloadable PDF |

---

## System Architecture

```
User Uploads Contract (PDF / TXT)
          |
          v
  Clause Segmentation Engine
  (Regex + Paragraph Boundaries)
          |
          v
  ML Risk Classification
  (TF-IDF Vectorizer + Logistic Regression)
          |
          v
  Flagged Risky Clauses
  (above confidence threshold)
          |
          v
+----------------------------------------------------------------+
|                      LangGraph StateGraph                       |
|                                                                 |
|  +--------------+     +--------------+     +--------------+    |
|  |   classify   |---->|   research   |---->|    reason    |    |
|  |  (ML Node)   |     |  (RAG Node)  |     |  (LLM Node)  |    |
|  +--------------+     +--------------+     +--------------+    |
|                                                                 |
|      AgentState flows through each node, accumulating data      |
+----------------------------------------------------------------+
          |
          v
  Structured JSON Report  +  Markdown Executive Report  +  PDF Export
```

### Cloud LLM Fallback Chain

```
Attempt 1: OpenRouter Key 1
     | (exception?)
     v
Attempt 2: OpenRouter Key 2
     | (exception?)
     v
Attempt 3: Groq (Final Fallback)
     | (exception?)
     v
safe_parse_analysis() — always returns a usable dict, never crashes
```

---

## Project Evolution: Milestone 1 -> Milestone 2

| Capability | Milestone 1 | Milestone 2 |
|---|:---:|:---:|
| PDF / Text Upload | + | + |
| Clause Segmentation | + | + |
| TF-IDF + Logistic Regression Classification | + | + |
| Confidence Scoring & Threshold Filter | + | + |
| Dashboard Visualizations | + | + |
| Legal Best Practice Retrieval (RAG) | - | + |
| LangGraph Multi-Step Agentic Reasoning | - | + |
| Plain-English Risk Explanation | - | + |
| Negotiation Tips & Safer Rewrites | - | + |
| Cloud LLM with 3-Level Fallback | - | + |
| Offline Local LLM Support (Ollama) | - | + |
| Structured JSON + Markdown Report | - | + |
| PDF Report Export | - | + |

---

## Tech Stack

### Machine Learning
- **Scikit-learn** — TF-IDF Vectorizer + Logistic Regression pipeline
- **Pandas** — Data preprocessing and EDA
- **NumPy** — Numerical operations
- **NLTK / spaCy** — NLP text processing

### Agentic AI
- **LangGraph** >= 0.2.0 — Multi-step agentic StateGraph orchestration
- **LangChain Core** >= 0.3.0 — Base primitives for node/state management

### LLM Providers
- **OpenRouter** — Primary cloud LLM (dual-key fallback)
- **Groq** — Secondary cloud fallback (Llama 3.3 70B)
- **Ollama** — Offline local LLM (Qwen, Llama)

### UI & Visualization
- **Streamlit** >= 1.32.0 — Web application frontend
- **Plotly** — Interactive risk distribution charts
- **FPDF2** — PDF report generation

### Retrieval (RAG)
- **TF-IDF Similarity Search** — Local, zero-dependency vector retrieval
- **Legal Knowledge Base** — Curated `data/legal_best_practices.json`
- **ChromaDB** >= 0.5.0 — Persistent vector DB (optional extended retrieval)
- **Sentence-Transformers** >= 3.0.0 — Semantic embedding support

### Utilities
- **python-dotenv** — Secure API key management via `.env`
- **PyPDF2** — Contract PDF parsing
- **Requests** — Ollama REST API client

---

## Project Structure

```
Contract-Risk-Classification/
|
+-- data/
|   +-- legal_best_practices.json     # Curated KB for RAG retrieval
|
+-- models/
|   +-- best_model.pkl                # Trained TF-IDF + LogReg pipeline
|
+-- src/                              # Jupyter Notebooks — Milestone 1 pipeline
|   +-- 1_inspect.ipynb
|   +-- 2_data_preprocessing.ipynb
|   +-- 3_feature_engineering.ipynb
|   +-- 4_train.ipynb
|   +-- 5_evaluate.ipynb
|
+-- CUAD_Dataset/                     # Raw CUAD v1 dataset (Milestone 1)
|
+-- contract_agent/                   # Agentic AI package (Milestone 2)
|   +-- __init__.py                   # Public API: run_hybrid_agent_pipeline()
|   +-- workflow.py                   # LangGraph StateGraph -- 3 nodes
|   +-- cloud_client.py               # 3-level cloud LLM fallback chain
|   +-- ollama_client.py              # Offline Ollama LLM integration
|   +-- kb_retriever.py               # TF-IDF RAG retriever
|   +-- _shared_prompt.py             # Centralised prompt engineering + schema
|   +-- report.py                     # JSON + Markdown report builder
|   +-- text_utils.py                 # Text cleaning and clause segmentation
|   +-- ml_utils.py                   # Cached ML model loader
|
+-- app.py                            # Streamlit UI (full M1 + M2 integration)
+-- run_app.sh                        # One-click auto-install and launch script
+-- requirements.txt                  # All dependencies
+-- .env.example                      # API key configuration template
+-- .streamlit/config.toml            # Theme and UI configuration
+-- milestone2.md                     # Detailed Milestone 2 technical reference
+-- README.md                         # This file
```

---

## Installation & Setup

### Prerequisites

- **Python 3.10+** installed on your system
- For **Online Mode**: at least one valid API key (OpenRouter or Groq)
- For **Offline Mode**: [Ollama](https://ollama.com/) installed locally

### Option A — Automated Setup (Recommended)

We provide a single startup script that automatically creates a virtual environment, installs all dependencies, and launches the application:

```bash
# 1. Clone the repository
git clone https://github.com/ashyou09/Contract-Risk-Classification.git
cd Contract-Risk-Classification

# 2. Configure your API keys
cp .env.example .env
# Open .env and fill in your API keys

# 3. Run the automated launcher
chmod +x run_app.sh
./run_app.sh
```

### Option B — Manual Setup

```bash
# 1. Clone the repository
git clone https://github.com/ashyou09/Contract-Risk-Classification.git
cd Contract-Risk-Classification

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 5. Launch the app
streamlit run app.py
```

### Offline Mode — Pull a Local Model

```bash
# Install Ollama: https://ollama.com/download
ollama pull qwen3.5:2b        # Recommended -- fast, good JSON output
# Alternative:
ollama pull llama3.2:3b
```

---

## Environment Configuration

Copy `.env.example` to `.env` and fill in your credentials:

```env
# -- Cloud Mode (Online) -----------------------------------------------

# OpenRouter -- Primary key (required for online mode)
OPENROUTER_API_KEY_1=sk-or-v1-...

# OpenRouter -- Secondary key (automatic fallback if Key 1 fails)
OPENROUTER_API_KEY_2=sk-or-v1-...

# OpenRouter model
OPENROUTER_MODEL=openai/gpt-oss-120b

# Groq -- Final fallback provider
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.3-70b-versatile

# -- Offline Mode (Ollama) --------------------------------------------

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3.5:2b
```

> **Security:** The `.env` file is listed in `.gitignore` and must never be committed to version control.

---

## Running the Application

```bash
streamlit run app.py
# App opens at: http://localhost:8501
```

### UI Overview

**Sidebar — Quick Settings**
- Processing Mode: Switch between Online (cloud LLM) or Local (Ollama)
- Risk Sensitivity Slider: Adjust the ML confidence threshold (0.0 – 1.0)
- System Health Checks: Live status of Cloud Server and Local Ollama
- Legal Knowledge Base: Status of the RAG vector store
- Document Upload: Upload `.pdf` or `.txt` contract files

**Main Panel**
- Risk distribution chart (High / Medium / Low breakdown)
- Clause-by-clause analysis with expandable details
- Full agentic reasoning report (8-dimension analysis per clause)
- Download report as **PDF** or copy as **Markdown**

---

## ML Model & Dataset

### Dataset — CUAD (Contract Understanding Atticus Dataset)

- 41 clause categories identified from real commercial contracts
- Strong class imbalance handled with stratified splits
- Categories mapped to: **High Risk**, **Medium Risk**, **Low Risk**

### Preprocessing Pipeline

1. Lowercase normalisation
2. Stopword removal
3. Punctuation cleaning (preserving legal terms)
4. Clause boundary segmentation

### Feature Engineering

- **TF-IDF Vectorizer** — up to 5,000 features, unigrams + bigrams

### Models Evaluated

| Model | Weighted F1 Score |
|---|:---:|
| **Logistic Regression** | **0.8859** (Selected) |
| Decision Tree | 0.8084 |
| Random Forest | 0.8058 |

**Why Logistic Regression?**
- Highest F1 score on legal text
- Fast, real-time inference
- Calibrated confidence probabilities
- Naturally handles high-dimensional sparse TF-IDF features

---

## Agentic Pipeline

### Node 1 — classify (ML Risk Classification)

Loads `models/best_model.pkl` (TF-IDF + Logistic Regression), segments the uploaded contract into individual clauses, classifies each clause, and applies the user-defined confidence threshold.

**Output per clause:**
```json
{ "clause_text": "...", "risk_level": "High", "confidence": 0.91 }
```

### Node 2 — research (RAG Legal Best-Practice Retrieval)

For every flagged clause, retrieves the **top-3 most relevant legal best-practice entries** from `data/legal_best_practices.json` using:
- TF-IDF cosine similarity over the full knowledge base
- Topic-aware boosting — regex detects clause domain (termination, indemnification, liability, IP, etc.) and prioritises domain-matched entries

### Node 3 — reason (LLM Deep Analysis)

Dispatches each clause + retrieved best practices to the selected LLM backend. Returns a full 8-dimension analysis, then assembles the final JSON and Markdown reports.

**LLM error isolation:** If a single clause fails, `safe_parse_analysis()` returns a graceful placeholder — the pipeline never crashes due to a single LLM failure.

---

## LLM Analysis Output Schema

Every flagged clause receives an 8-key structured analysis:

| Field | Description |
|---|---|
| `plain_english_summary` | What the clause says in plain language (2-3 sentences) |
| `what_makes_it_risky` | Specific legal danger with real-world consequence |
| `who_bears_the_risk` | "Client" / "Vendor" / "Both" / "Third Party" |
| `severity_rationale` | Why the ML model rated this risk level |
| `industry_standard_practice` | What a fair, balanced version looks like |
| `negotiation_tips` | 2-3 specific, numbered actionable negotiation points |
| `safer_rewrite` | Complete, ready-to-use replacement clause in legal style |
| `action_required` | "Remove Clause" / "Negotiate Terms" / "Seek Legal Review" / "Accept with Caution" |

---

## System Design Decisions

| Decision | Rationale |
|---|---|
| **LangGraph over plain LangChain** | Explicit StateGraph with typed nodes gives full traceability and deterministic error propagation |
| **TF-IDF RAG over ChromaDB** | Zero external dependency for retrieval; fast, deterministic, local-only |
| **3-Level LLM Fallback** | Ensures uninterrupted analysis even with rate limits or API key exhaustion |
| **JSON response format enforcement** | `response_format={"type": "json_object"}` on all cloud calls minimises parse failures |
| **`safe_parse_analysis()` defensive parser** | Handles Qwen `<think>` tags, markdown fences, malformed JSON — always returns a usable dict |
| **`temperature=0.15`** | Keeps LLM output consistent, factual, and minimally hallucinated |
| **`@lru_cache` on model and graph** | Avoids re-deserialising the ML model or re-compiling the LangGraph StateGraph on every request |

---

## Evaluation Metrics

### ML Classification — Milestone 1

| Model | Weighted F1 Score |
|---|:---:|
| **Logistic Regression** | **0.8859** |
| Decision Tree | 0.8084 |
| Random Forest | 0.8058 |

### Overall Risk Signal — Milestone 2 Report

| Condition | Signal |
|---|---|
| High Risk clauses >= 2 | HIGH RISK DOCUMENT — Multiple critical issues detected |
| High Risk clauses = 1 | ELEVATED RISK — At least one critical clause requires attention |
| Medium Risk clauses >= 2 | MODERATE RISK — Several clauses need review |
| Otherwise | LOWER RISK — Minor concerns only, standard review recommended |

---

## Team Contribution

| Member | Student ID | Responsibilities |
|---|---|---|
| **Ashutosh Singh** | 2401010109 | LangGraph agentic workflow (`workflow.py`), cloud fallback chain (`cloud_client.py`), RAG retriever (`kb_retriever.py`), prompt schema (`_shared_prompt.py`), Streamlit M2 UI integration |
| **Ranvendra Pratap Singh** | 2401010373 | Offline Ollama integration (`ollama_client.py`), auto-start daemon, offline mode end-to-end testing, local model evaluation |
| **Shreya Suman** | 2401020068 | Report builder (`report.py`), JSON + Markdown generation, executive summary logic, action summary table, overall risk signal algorithm |
| **Avishkar Meher** | 2401010116 | Text preprocessing (`text_utils.py`), ML model caching (`ml_utils.py`), configuration documentation (`.env.example`) |

---

## Disclaimer

This tool is an **academic project** designed for educational purposes to demonstrate ML-based contract analysis and autonomous AI reasoning. It does **NOT** constitute professional legal advice.

Always consult a qualified legal professional before signing, modifying, or rejecting any contract clause.

---

*Intelligent Contract Risk Analysis System — Final Integrated Submission — Milestone 1 + Milestone 2*
