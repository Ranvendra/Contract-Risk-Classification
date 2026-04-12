import json
import os
import re
from io import BytesIO, StringIO

import pandas as pd
import plotly.express as px
import PyPDF2
import streamlit as st
from dotenv import load_dotenv

from contract_agent.text_utils import clean_text, get_summary, segment_clauses
from contract_agent.workflow import run_hybrid_agent_pipeline

load_dotenv()

# =============================
# Page Config
# =============================
st.set_page_config(
    page_title="Contract Risk Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================
# Custom CSS
# =============================
st.markdown("""
<style>
/* Main App Background and Typography */
.stApp {
    font-family: 'Inter', 'Segoe UI', Roboto, sans-serif;
}

/* Metric Cards Styling */
[data-testid="stMetric"] {
    padding: 15px 20px;
    border-radius: 12px;
    border: none;
    background: var(--secondary-background-color);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2), 0 4px 6px -2px rgba(0, 0, 0, 0.1);
}

/* Risk Clause Cards Typography and Spacing */
.risk-card {
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 12px;
    background: var(--secondary-background-color);
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    border: 1px solid rgba(128,128,128,0.2);
    transition: all 0.2s ease;
}
.risk-card:hover {
    box-shadow: 0 4px 6px rgba(0,0,0,0.2);
}
.high-risk { border-left: 6px solid #ef4444 !important; }
.medium-risk { border-left: 6px solid #f97316 !important; }
.low-risk { border-left: 6px solid #22c55e !important; }

/* Upload Dropzone Styling */
[data-testid="stFileUploadDropzone"] {
    border-radius: 16px;
    border: 2px dashed #3b82f6;
    background-color: rgba(59, 130, 246, 0.05);
    padding: 30px 20px;
    transition: all 0.3s ease;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #2563eb;
    background-color: rgba(59, 130, 246, 0.1);
    transform: scale(1.01);
}

/* Customizing Streamlit Expander Header */
.streamlit-expanderHeader {
    font-weight: 600 !important;
    border-radius: 8px !important;
}

/* Mode badge inline */
.mode-badge-online {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    background: linear-gradient(135deg, #1e40af, #3b82f6);
    color: white;
}
.mode-badge-offline {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    background: linear-gradient(135deg, #14532d, #22c55e);
    color: white;
}
</style>
""", unsafe_allow_html=True)

# =============================
# Load Pipeline Model
# =============================
@st.cache_resource
def load_model():
    from contract_agent.ml_utils import load_sklearn_pipeline
    return load_sklearn_pipeline()


@st.cache_data(ttl=30)
def _get_ollama_health():
    """Cached Ollama health-check (refresh every 30 s)."""
    from contract_agent.ollama_client import check_ollama_health
    return check_ollama_health()


# =============================
# ML Dashboard (Milestone 1)
# =============================
def _render_ml_dashboard(df):
    st.markdown("### 📊 Executive Summary")

    m1, m2, m3, m4 = st.columns(4)

    high_count = len(df[df["Risk Level"] == "High"])
    med_count = len(df[df["Risk Level"] == "Medium"])
    low_count = len(df[df["Risk Level"] == "Low"])
    avg_conf = df["Confidence"].mean()

    m1.metric(
        "High Risk Clauses",
        high_count,
        delta="Attention Needed" if high_count > 0 else "Safe",
        delta_color="inverse",
    )

    m2.metric("Medium Risk Clauses", med_count)
    m3.metric("Low Risk Clauses", low_count)
    m4.metric("Avg Confidence", f"{avg_conf:.1%}")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Risk Distribution")
        fig_pie = px.pie(
            df,
            names="Risk Level",
            color="Risk Level",
            color_discrete_map={
                "High": "#085566",
                "Medium": "#440866",
                "Low": "#085566",
            },
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.markdown("#### Confidence Score Distribution")
        fig_hist = px.histogram(
            df,
            x="Confidence",
            color="Risk Level",
            nbins=20,
            color_discrete_map={
                "High": "#085566",
                "Medium": "#440866",
                "Low": "#085566",
            },
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        st.markdown("#### Risk Intensity Across Document")

        df = df.copy()
        df["Risk_Score"] = df["Risk Level"].map({"High": 3, "Medium": 2, "Low": 1})

        fig_scatter = px.scatter(
            df,
            x=df.index,
            y="Risk_Score",
            color="Risk Level",
            color_discrete_map={
                "High": "#085566",
                "Medium": "#440866",
                "Low": "#085566",
            },
            labels={"index": "Clause Position", "Risk_Score": "Risk Intensity"},
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

    with c4:
        st.markdown("#### Top Keywords (High Risk Clauses)")

        from collections import Counter

        high_text = " ".join(
            df[df["Risk Level"] == "High"]["Clause Text"].tolist()
        ).lower()

        words = re.findall(r"\b\w+\b", high_text)

        stop_words = {
            "the", "and", "to", "of", "in", "a", "is", "that", "for", "it",
            "on", "be", "as", "by", "or", "this", "an", "are", "with", "from",
            "at", "not", "will",
        }

        filtered = [w for w in words if w not in stop_words and len(w) > 3]
        common_words = Counter(filtered).most_common(10)

        if common_words:
            kw_df = pd.DataFrame(common_words, columns=["Keyword", "Count"])
            fig_bar = px.bar(
                kw_df,
                x="Keyword",
                y="Count",
                color="Count",
                color_continuous_scale="Reds",
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No high risk clauses detected.")

    st.markdown("---")
    st.markdown("### 🔍 Detailed Clause Review")

    filter_risk = st.multiselect(
        "Filter Risk",
        ["High", "Medium", "Low"],
        default=["High", "Medium"],
    )

    df_filtered = df[df["Risk Level"].isin(filter_risk)]

    for _, row in df_filtered.iterrows():
        risk = row["Risk Level"]
        color_class = risk.lower() + "-risk"
        icon = "🔴" if risk == "High" else "🟠" if risk == "Medium" else "🟢"

        summary_text = get_summary(row["Clause Text"], 100)

        with st.expander(f"{icon} {risk}: {summary_text} ({row['Confidence']:.1%})"):
            st.markdown(
                f"""
                    <div class="risk-card {color_class}">
                        <p><strong>Full Clause:</strong></p>
                        <p>{row['Clause Text']}</p>
                    </div>
                    """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    csv = df.drop(columns=["Risk_Score"], errors="ignore").to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Analysis Report (CSV)",
        data=csv,
        file_name="contract_risk_analysis.csv",
        mime="text/csv",
        key="dl_csv_ml",
    )


# =============================
# Agentic Panel (Milestone 2)
# =============================
def _render_agentic_panel(raw_contract_text: str, confidence_threshold: float, mode: str):
    is_online = mode == "online"

    # ── Mode description ──────────────────────────────────────────────────────
    badge_cls = "mode-badge-online" if is_online else "mode-badge-offline"
    badge_lbl = "🌐 Online — OpenRouter" if is_online else "🖥 Offline — Ollama"
    st.markdown(
        f"""
        **Milestone 2 — Hybrid Agent** &nbsp;
        <span class="{badge_cls}">{badge_lbl}</span>

        The sklearn pipeline scores each clause; **LangGraph** chains
        *Classification → TF-IDF RAG (legal best-practices corpus) → {'OpenRouter LLM' if is_online else 'Ollama LLM'}*
        so the model explains risk and proposes wording grounded in retrieved legal excerpts.
        """,
        unsafe_allow_html=True,
    )

    # ── Guard: Online needs API key ───────────────────────────────────────────
    if is_online and not os.environ.get("OPENROUTER_API_KEY", "").strip():
        st.warning(
            "Set `OPENROUTER_API_KEY` in a `.env` file (see `.env.example`). "
            "Never commit real keys to git."
        )
        return

    # ── Guard: Offline needs Ollama ───────────────────────────────────────────
    if not is_online:
        health = _get_ollama_health()
        if not health["reachable"]:
            st.error(
                "❌ **Ollama is not reachable.** "
                "Make sure it is installed and run: `ollama serve` in a terminal. "
                "Then refresh this page."
            )
            st.code("ollama serve", language="bash")
            return

    # ── Run button ────────────────────────────────────────────────────────────
    btn_label = (
        "🚀 Run Hybrid Risk Report (OpenRouter)" if is_online
        else "🖥 Run Hybrid Risk Report (Ollama — Local)"
    )

    if st.button(btn_label, type="primary", use_container_width=True, key="run_agent_btn"):
        spinner_msg = (
            "Running LangGraph workflow (classify → RAG → OpenRouter)…"
            if is_online
            else f"Running LangGraph workflow (classify → RAG → Ollama `{os.environ.get('OLLAMA_MODEL', 'qwen3.5:2b')}`)…"
        )
        with st.spinner(spinner_msg):
            result = run_hybrid_agent_pipeline(
                raw_contract_text,
                confidence_threshold,
                mode=mode,
            )

        if result.get("error"):
            st.error(result["error"])
            return

        # ── Render report ──────────────────────────────────────────────────
        st.success("✅ Analysis complete!")
        st.markdown(result.get("markdown_report", ""))

        rep = result.get("structured_report")
        if rep:
            st.download_button(
                label="📥 Download Structured Report (JSON)",
                data=json.dumps(rep, indent=2).encode("utf-8"),
                file_name="agentic_risk_report.json",
                mime="application/json",
                key="dl_agent_json",
            )

    # ── LangGraph workflow diagram ────────────────────────────────────────────
    with st.expander("ℹ️ How the Agent Works (LangGraph Flow)", expanded=False):
        st.markdown("""
        ```
        START
          │
          ▼
        [classify]  ← Sklearn TF-IDF + Logistic Regression
          │            Segments clauses, predicts risk & confidence
          ▼
        [research]  ← TF-IDF RAG over legal_best_practices.json
          │            Retrieves top-3 relevant legal guidelines
          ▼
        [reason]    ← LLM (OpenRouter or Ollama)
          │            Explains risk, compares to best practice,
          │            proposes safer clause rewrite
          ▼
         END  →  Structured JSON + Markdown Report
        ```

        **Three Nodes, One Goal:** Every flagged clause gets a full
        *explain → compare → rewrite* treatment grounded in retrieved legal knowledge.
        No hallucinated citations — the LLM only reasons about what was retrieved.
        """)


# =============================
# Main App
# =============================
def main():
    if "file_data" not in st.session_state:
        st.session_state.file_data = None
        st.session_state.file_name = None
        st.session_state.file_type = None

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Settings")

        # ── AI Mode Toggle ─────────────────────────────────────────────────
        st.markdown("#### 🤖 AI Mode")
        mode_choice = st.radio(
            "Select inference backend:",
            options=["🌐 Online (OpenRouter)", "🖥 Offline (Ollama)"],
            index=0,
            key="mode_radio",
            help="Online uses the OpenRouter cloud API. Offline uses your local Ollama models — no internet needed.",
        )
        selected_mode = "online" if "Online" in mode_choice else "offline"

        # ── Mode-specific status ───────────────────────────────────────────
        if selected_mode == "online":
            api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
            model_name = os.environ.get("OPENROUTER_MODEL", "openai/gpt-oss-120b")
            if api_key:
                st.success(f"🟢 API Key set  \n`{model_name}`")
            else:
                st.warning("🔴 `OPENROUTER_API_KEY` not set in `.env`")
        else:
            # Offline: show Ollama health
            with st.spinner("Checking Ollama…"):
                health = _get_ollama_health()

            if health["reachable"]:
                auto_msg = " _(auto-started)_" if health.get("auto_started") else ""
                st.success(
                    f"🟢 Ollama ready{auto_msg}  \n"
                    f"Model: `{health['model']}`"
                )
                # Model selector if multiple models available
                available = health.get("available_models", [])
                if len(available) > 1:
                    chosen_model = st.selectbox(
                        "Choose Ollama model:",
                        options=available,
                        index=available.index(health["model"]) if health["model"] in available else 0,
                        key="ollama_model_select",
                    )
                    # Override env var for this session
                    os.environ["OLLAMA_MODEL"] = chosen_model
            else:
                st.error("🔴 Ollama not reachable")
                st.caption("Run `ollama serve` in a terminal, then refresh.")

        st.markdown("---")

        # ── File controls + confidence ─────────────────────────────────────
        if st.session_state.file_data is not None:
            st.markdown("### 📄 Update Document")
            new_file = st.file_uploader(
                "Upload a New Contract",
                type=["txt", "pdf"],
                help="Drag and drop or click to select.",
                key="sidebar_uploader",
            )
            if new_file:
                st.session_state.file_data = new_file.getvalue()
                st.session_state.file_name = new_file.name
                st.session_state.file_type = new_file.type
                st.rerun()

            st.markdown("---")

        st.markdown("#### 🎚 Confidence Threshold")
        confidence_threshold = st.slider(
            "Min Confidence Score",
            0.0, 1.0, 0.5, 0.05,
            help="Clauses below this score are excluded from analysis.",
        )

        st.markdown("---")
        if st.session_state.file_data is not None:
            if st.button("🗑 Clear Current File", use_container_width=True):
                st.session_state.file_data = None
                st.session_state.file_name = None
                st.session_state.file_type = None
                st.rerun()

        # ── Sidebar footer ─────────────────────────────────────────────────
        st.markdown("---")
        st.caption(
            "⚖️ **Contract Risk Analyzer**  \n"
            "Milestone 2 — Agentic AI  \n"
            "Academic prototype · Not legal advice"
        )

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0 1rem 0;'>
            <h1 style='
                font-size: 3rem; 
                font-weight: 800; 
                background: linear-gradient(90deg, #3B82F6, #60A5FA, #3B82F6); 
                -webkit-background-clip: text; 
                -webkit-text-fill-color: transparent; 
                margin-bottom: 0.5rem;'>
                Intelligent Contract Risk Analysis
            </h1>
            <p style='color: var(--text-color); opacity: 0.8; font-size: 1.25rem; font-weight: 400;'>
                Uncover hidden risks in legal documents instantly using AI
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='width: 50%; margin: 10px auto 30px auto; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)

    # ── File upload (home state) ──────────────────────────────────────────────
    if st.session_state.file_data is None:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            main_file = st.file_uploader(
                "Upload a Contract (PDF or TXT)",
                type=["txt", "pdf"],
                help="Drag and drop or click to select.",
                key="main_uploader",
            )
            if main_file:
                st.session_state.file_data = main_file.getvalue()
                st.session_state.file_name = main_file.name
                st.session_state.file_type = main_file.type
                st.rerun()

        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    else:
        st.success(f"Analysis Ready: **{st.session_state.file_name}** — Scroll down for results")
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # ── Load ML Model ─────────────────────────────────────────────────────────
    model = load_model()
    if model is None:
        st.error("⚠️ Model missing! Run training first (see `src/4_train.ipynb`).")
        return

    # ── Main analysis area ───────────────────────────────────────────────────
    if st.session_state.file_data is not None:
        text = ""
        try:
            if st.session_state.file_type == "application/pdf":
                reader = PyPDF2.PdfReader(BytesIO(st.session_state.file_data))
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            else:
                stringio = StringIO(st.session_state.file_data.decode("utf-8"))
                text = stringio.read()
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return

        if text:
            clauses = segment_clauses(text)
            data = []
            for clause in clauses:
                cleaned = clean_text(clause)
                risk = model.predict([cleaned])[0]
                probs = model.predict_proba([cleaned])[0]
                confidence = max(probs)
                if confidence >= confidence_threshold:
                    data.append({
                        "Clause Text": clause,
                        "Risk Level": risk,
                        "Confidence": confidence,
                    })

            df = pd.DataFrame(data)

            tab_m1, tab_m2 = st.tabs(
                ["📊 Milestone 1 — Classical ML", "🤖 Milestone 2 — Agentic (LangGraph)"]
            )

            with tab_m1:
                if df.empty:
                    st.warning("No clauses found matching the confidence criteria.")
                else:
                    _render_ml_dashboard(df)

            with tab_m2:
                _render_agentic_panel(text, confidence_threshold, mode=selected_mode)

        else:
            st.warning("No extractable text was found in this file.")

    else:
        st.markdown("""
        <div style="text-align: center; padding: 60px; font-family: 'Inter', sans-serif;">
            <div style="font-size: 4rem; margin-bottom: 1rem; opacity: 0.4;">📄</div>
            <h3 style="font-weight: 500; color: var(--text-color); opacity: 0.8;">Awaiting document upload</h3>
            <p style="font-size: 1.1rem; color: var(--text-color); opacity: 0.6;">Please upload a contract to begin the analysis.</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()