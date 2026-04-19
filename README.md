<img src="https://images.ctfassets.net/em6l9zw4tzag/oVfiswjNH7DuCb7qGEBPK/b391db3a1d0d3290b96ce7f6aacb32b0/python.png" height="70" alt="Python"/> &nbsp; <img src="https://uxwing.com/wp-content/themes/uxwing/download/brands-and-social-media/streamlit-icon.png" height="70" alt="Streamlit"/> &nbsp; <img src="https://png.pngtree.com/png-clipart/20250210/original/pngtree-blue-robot-toys-png-image_20412795.png" height="70" alt="Scikit-Learn"/> &nbsp; <img src="https://adibrata.com/wp-content/uploads/2024/09/LangGraph-logo.webp" height="70" alt="LangGraph"/>

# Intelligent Contract Risk Analysis & Agentic Legal Assistance System

> **Final Integrated Submission — Milestone 1 + Milestone 2**

An AI-driven legal document intelligence platform that evolved from a classical Machine Learning contract risk classifier into a full **Agentic AI Legal Assistant**.

This project combines **NLP + Supervised Learning + Retrieval-Augmented Generation (RAG) + LangGraph multi-step reasoning workflows** to analyze contracts, detect risky clauses, explain legal concerns in plain English, and generate structured mitigation reports.

---

# Project Overview

Legal professionals, startups, freelancers, and businesses often sign contracts without fully understanding hidden risks such as:

- Unlimited liability clauses  
- One-sided indemnification terms  
- Unfair termination rights  
- Confidentiality traps  
- Payment disputes  
- Intellectual property ownership risks  

Manual review is expensive, time-consuming, and inaccessible to many individuals.

## Our Solution

We designed an **Intelligent Contract Risk Analysis System** that automates contract review in two progressive phases:

### Milestone 1 — ML-Based Contract Risk Detection

Uses classical NLP and supervised machine learning to classify clauses into:

- High Risk  
- Medium Risk  
- Low Risk  

### Milestone 2 — Agentic AI Legal Assistant

Transforms the system into an autonomous reasoning engine that:

- Understands risky clauses deeply  
- Retrieves relevant legal best practices  
- Explains risks in plain language  
- Suggests safer clause rewrites  
- Generates executive-ready reports  

---

# Milestone 1 → Milestone 2 Evolution

| Capability | Milestone 1 | Milestone 2 |
|-----------|------------|------------|
| PDF / Text Upload | ✅ | ✅ |
| Clause Segmentation | ✅ | ✅ |
| TF-IDF + Logistic Regression Classification | ✅ | ✅ |
| Confidence Scoring | ✅ | ✅ |
| Dashboard Visualizations | ✅ | ✅ |
| Legal Best Practice Retrieval | ❌ | ✅ |
| Agentic Multi-Step Reasoning | ❌ | ✅ |
| Plain-English Explanation | ❌ | ✅ |
| Negotiation Advice | ❌ | ✅ |
| Safer Clause Rewrite | ❌ | ✅ |
| JSON + Markdown Reports | ❌ | ✅ |
| Cloud + Offline LLM Support | ❌ | ✅ |

---

# Core Declaration

This project began as a **strict non-GenAI Milestone 1 system**, where all clause classification logic used classical Machine Learning only.

Milestone 2 extends the project responsibly using controlled LLM reasoning layers supported by:

- Structured prompts  
- Retrieval-grounded responses  
- Multi-provider fallback systems  
- Output schema enforcement  
- Hallucination reduction techniques  
- Legal disclaimers  

---

# Technical Features

## 1. Legal Clause Segmentation Engine

Contracts are automatically divided into logical clauses using:

- Regex boundaries  
- Numbered sections  
- Paragraph structures  
- Formatting cleanup  

---

## 2. ML Risk Prediction Layer

Each clause is classified using the trained production pipeline:

- TF-IDF Vectorizer  
- Logistic Regression

Outputs:

- Risk label  
- Confidence score  

---

## 3. Confidence Threshold Filtering

Users can dynamically filter weaker predictions.

Example:

- Show only clauses above 80% confidence

---

## 4. Retrieval-Augmented Legal Intelligence (RAG)

For every risky clause, the system retrieves relevant entries from a curated legal knowledge base containing:

- Industry-standard wording  
- Balanced clause structures  
- Negotiation norms  
- Best practices  

Implemented using:

- TF-IDF similarity search  
- Topic-aware ranking  
- Local retrieval pipeline  

---

## 5. LangGraph Agentic Workflow

The complete reasoning pipeline is orchestrated using **LangGraph StateGraph**.

### Multi-Step Nodes:

1. **Classify Node** → ML risk detection  
2. **Research Node** → Legal best-practice retrieval  
3. **Reason Node** → LLM deep legal analysis + reporting  

---

## 6. Cloud + Offline AI Modes

### Online Mode

Fallback chain:

- OpenRouter Key 1  
- OpenRouter Key 2  
- Groq API  

### Offline Mode

Runs local LLM using **Ollama**

Examples:

- Qwen models  
- Llama models  

---

## 7. Structured Executive Reports

Final outputs generated in:

### JSON Report

Machine-readable structured output

### Markdown Report

Human-readable professional summary

---

# Dataset & EDA

## Primary Dataset

**CUAD — Contract Understanding Atticus Dataset**

Used during Milestone 1 to train clause classification models.

## Insights

- 41 clause categories identified  
- Strong class imbalance  
- Variable clause lengths  
- Complex legal vocabulary  

Mapped into:

- High Risk  
- Medium Risk  
- Low Risk  

---

# Methodology & Optimisation

## Preprocessing

- Lowercasing  
- Stopword removal  
- Cleaning punctuation  
- Clause segmentation  

## Feature Engineering

TF-IDF with:

- Up to 5000 features  
- Unigrams + Bigrams  

## Models Evaluated

- Logistic Regression  
- Decision Tree  
- Random Forest  

## Final Selection

**Logistic Regression** selected because of:

- Best weighted F1 score  
- Fast inference  
- Explainable probabilities  
- Strong sparse-text performance  

---

# Evaluation Metrics

| Model | Weighted F1 Score |
|------|-------------------|
| Logistic Regression | **0.8859** |
| Decision Tree | 0.8084 |
| Random Forest | 0.8058 |

---

# Milestone 2 Analysis Output Schema

Each flagged clause receives deep reasoning across:

- Plain English Summary  
- What Makes It Risky  
- Who Bears the Risk  
- Severity Rationale  
- Industry Standard Practice  
- Negotiation Tips  
- Safer Rewrite  
- Recommended Action  

---

# Tech Stack

## Machine Learning

- Scikit-learn  
- Pandas  
- NumPy  

## Agentic AI

- LangGraph  
- LangChain Core  

## LLM Providers

- OpenRouter  
- Groq  
- Ollama  

## Frontend

- Streamlit  

## Retrieval

- TF-IDF Vector Search  
- Local Knowledge Base  

## Utilities

- Python Dotenv  
- Requests  
- Pickle  

---

# System Architecture

```text
User Uploads Contract
        │
        ▼
Clause Segmentation Engine
        │
        ▼
ML Risk Classification
(TF-IDF + Logistic Regression)
        │
        ▼
Flagged Risky Clauses
        │
        ▼
RAG Legal Knowledge Retrieval
        │
        ▼
LangGraph Agent Workflow
 ┌──────────────┬──────────────┬──────────────┐
 │ Classify     │ Research     │ Reason       │
 └──────────────┴──────────────┴──────────────┘
        │
        ▼
Structured JSON Report
Markdown Executive Report
```
# Project Structure

```text
Contract-Risk-Classification/
├── data/
│   └── legal_best_practices.json
├── models/
│   └── best_model.pkl
├── src/
│   ├── preprocessing notebooks
│   ├── training notebooks
│   └── evaluation notebooks
├── contract_agent/
│   ├── workflow.py
│   ├── cloud_client.py
│   ├── ollama_client.py
│   ├── kb_retriever.py
│   ├── report.py
│   ├── text_utils.py
│   └── ml_utils.py
├── app.py
├── requirements.txt
└── README.md
```

## 🚀 Installation & Setup

### Prerequisites

Make sure you have Python 3.8+ installed on your system.

### 1. Clone the repository

```bash
git clone https://github.com/ashyou09/Contract-Risk-Classification.git
cd Contract-Risk-Classification
```

### 2. Configure Environment

Copy the `.env.example` file to `.env` and fill in your keys (required for cloud model capabilities):

```bash
cp .env.example .env
```

### 3. Auto-Install & Run Application

We have implemented an automated startup script that will automatically create a secure Python virtual environment, download all necessary dependencies without throwing errors, and launch the UI.

```bash
# Make the script executable securely and run it:
chmod +x run_app.sh
./run_app.sh
```

*The script manages `requirements.txt` dynamically. If you prefer to manually install, use `pip install -r requirements.txt` followed by `streamlit run app.py`.*

---

## 👥 Team Contribution


- **Member 1 (Ashutosh Singh - 2401010109)**: Agent workflow, backend architecture, RAG system, Streamlit integration
- **Member 2 (Ranvendra Pratap Singh - 2401010373)**: Offline Ollama mode, testing, deployment support.
- **Member 3 (Shreya Suman - 2401020068)**: Report generation, executive summaries, UI logic, documentation

- **Member 4 (Avishkar Meher - 2401010116)**: Utilities, preprocessing support, integration testing.

---

**Disclaimer:** This tool is an academic project designed for educational purposes to demonstrate ML-based analysis and autonomous AI reasoning. It does NOT constitute professional legal advice.
