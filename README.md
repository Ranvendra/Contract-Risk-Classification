<img src="https://images.ctfassets.net/em6l9zw4tzag/oVfiswjNH7DuCb7qGEBPK/b391db3a1d0d3290b96ce7f6aacb32b0/python.png" height="70" alt="Python"/> &nbsp; <img src="https://uxwing.com/wp-content/themes/uxwing/download/brands-and-social-media/streamlit-icon.png" height="70" alt="Streamlit"/> &nbsp; <img src="https://png.pngtree.com/png-clipart/20250210/original/pngtree-blue-robot-toys-png-image_20412795.png" height="70" alt="Scikit-Learn"/> &nbsp; <img src="https://adibrata.com/wp-content/uploads/2024/09/LangGraph-logo.webp" height="70" alt="LangGraph"/>

# Intelligent Contract Risk Analysis - Agentic AI Legal Assistance Tool

An advanced AI-driven legal document analysis system that evaluates contractual risk. This project extends our initial machine learning classification system into a fully autonomous agentic AI legal assistance tool that reasons about contract risks, retrieves legal best practices, and generates structured risk reports.

---

## 🎯 Objective

Extend the analysis system into an agentic AI legal assistance tool that autonomously reasons about contract risks, retrieves legal best practices, and generates structured risk reports.

## ⚙️ Functional Requirements

- **Accept legal queries and analyzed contract data**: Seamlessly process user-uploaded contract documents (PDF/Text).
- **Analyze contract risk patterns autonomously**: Extract evaluable clauses and automatically identify risk areas utilizing intelligent components.
- **Retrieve relevant legal guidelines/standards**: Dynamically fetch contextual legal best practices and industry-standard alternative clauses.
- **Generate structured risk assessment reports**: Automatically synthesize a high-fidelity, client-ready summary of findings and recommended actions.

## 🛠️ Technical Requirements (Agentic)

- **Framework:** LangGraph (Workflow & State). Orchestrates the pipeline nodes efficiently.
- **RAG:** Retrieval of legal standards (Chroma/FAISS / local TF-IDF Knowledge Base).
- **State:** Explicit state management across steps. The `AgentState` gracefully propagates data and isolated errors without disruption.
- **Prompting:** Strict prompting strategies configured with constrained output schemas to significantly reduce hallucinations.

## 📊 Structured Output

For every parsed contract, the system generates deep actionable intelligence organized into these components:

- **Problem understanding & Legal use case:** Tailored intelligence explicitly built to aid legal and business reviewers.
- **Input–output specification:** Clear transparency tracking raw clause extraction, inference, and final analysis.
- **System architecture diagram:** Orchestration between the Interface, ML Classifier, RAG Retrieval, and the LangGraph Reasoning agent.
- **Working local application with UI:** Interactive local dashboard indicating confidence scores and progress.
- **Model performance & NLP evaluation:** Base F1 risk categorization overlaid with qualitative LLM reasoning accuracy.
- **Summary:** Contract overview & Risk severity breakdown.
- **Deep Dive:** Clause references & Plain-English Explanations defining exact liabilities.
- **Mitigation:** Recommended strategic actions, negotiation tips, and safer rewritten clauses.
- **Disclaimer:** Prominent Legal & Ethical notices clarifying the system boundary.

---

## 🏗️ System Architecture / Code Structure

```text
Contract-Risk-Classification/
├── CUAD_Dataset/                  # Raw CUAD v1 dataset repository
├── data/
│   └── legal_best_practices.json  # Curated KB for RAG retrieval
├── models/
│   └── best_model.pkl             # Deployed ML pipeline (TF-IDF + LogReg)
├── src/                           # Jupyter Notebooks detailing data/ML pipeline
├── contract_agent/                # Agentic AI package (LangGraph, Prompts, Client)
│   ├── workflow.py                # LangGraph StateGraph definition
│   ├── cloud_client.py            # LLM fallback chain logic
│   ├── kb_retriever.py            # TF-IDF RAG Retriever
│   └── ...
├── app.py                         # Main Streamlit web application & Inference Engine
├── README.md                      # This Project documentation page
├── milestone2.md                  # Comprehensive Agentic architecture specification
└── requirements.txt               # Project python dependencies
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

_(Please update these fields with your exact team members and responsibilities before submission)_

- **Member 1 (Ashutosh Singh - 2401010109)**: Developed the custom data preprocessing pipeline, CUAD JSON parsing, and risk mapping logic, Scikit-learn Pipeline architecture, Trained and evaluated ML, Streamlit UI.
- **Member 2 (Ranvendra Pratap Singh - 2401010373)**: Implemented the TF-IDF feature engineering and setup Scikit-learn Pipeline architecture, Trained and evaluated ML, Streamlit UI.
- **Member 3 (Shreya Suman - 2401020068)**: Scikit-learn Pipeline architecture, Trained and evaluated ML models (Logistic Regression, Decision Trees), performed Cross-validation,Streamlit UI.

- **Member 4 (Avishkar Meher - 2401010116)**: Contributed to quality assurance checks and supporting end-to-end integration testing. Participated in structuring and workflow refinement to ensure project readiness.

---

**Disclaimer:** This tool is an academic project designed for educational purposes to demonstrate ML-based analysis and autonomous AI reasoning. It does NOT constitute professional legal advice.
