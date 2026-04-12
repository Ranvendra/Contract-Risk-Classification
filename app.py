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

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Contract Risk Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — premium dark-accent design
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, .stApp { font-family: 'Inter', sans-serif; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    padding: 14px 18px;
    border-radius: 12px;
    background: var(--secondary-background-color);
    box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    transition: transform .18s ease, box-shadow .18s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(0,0,0,0.22);
}

/* ── Upload dropzone ── */
[data-testid="stFileUploadDropzone"] {
    border-radius: 16px;
    border: 2px dashed #3b82f6;
    background: rgba(59,130,246,.05);
    padding: 32px 20px;
    transition: all .25s ease;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #2563eb;
    background: rgba(59,130,246,.1);
    transform: scale(1.01);
}

/* ── Expander header ── */
.streamlit-expanderHeader { font-weight: 600 !important; border-radius: 8px !important; }

/* ── Clause risk cards ── */
.risk-card {
    padding: 18px 20px; border-radius: 10px; margin-bottom: 10px;
    background: var(--secondary-background-color);
    box-shadow: 0 1px 4px rgba(0,0,0,.1);
    border: 1px solid rgba(128,128,128,.15);
    transition: box-shadow .18s;
}
.risk-card:hover { box-shadow: 0 4px 14px rgba(0,0,0,.18); }
.high-risk   { border-left: 5px solid #ef4444 !important; }
.medium-risk { border-left: 5px solid #f97316 !important; }
.low-risk    { border-left: 5px solid #22c55e !important; }

/* ── Mode badge ── */
.badge-online  { display:inline-block; padding:3px 12px; border-radius:20px;
                 font-size:.78rem; font-weight:700;
                 background:linear-gradient(135deg,#1e40af,#3b82f6); color:#fff; }
.badge-offline { display:inline-block; padding:3px 12px; border-radius:20px;
                 font-size:.78rem; font-weight:700;
                 background:linear-gradient(135deg,#14532d,#22c55e); color:#fff; }

/* ── Action pill ── */
.action-pill {
    display:inline-block; padding:3px 12px; border-radius:20px;
    font-size:.78rem; font-weight:600; margin-left:8px;
}
.pill-remove    { background:#fee2e2; color:#b91c1c; }
.pill-negotiate { background:#fef3c7; color:#92400e; }
.pill-legal     { background:#ede9fe; color:#5b21b6; }
.pill-accept    { background:#d1fae5; color:#065f46; }

/* ── Welcome card ── */
.welcome-card {
    border-radius: 16px;
    padding: 32px 36px;
    background: linear-gradient(135deg, rgba(59,130,246,.08), rgba(99,102,241,.06));
    border: 1px solid rgba(99,102,241,.2);
    margin-bottom: 24px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Cached helpers
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    from contract_agent.ml_utils import load_sklearn_pipeline
    return load_sklearn_pipeline()


@st.cache_data(ttl=30)
def _ollama_health():
    from contract_agent.ollama_client import check_ollama_health
    return check_ollama_health()


# ─────────────────────────────────────────────────────────────────────────────
# Milestone 1 dashboard
# ─────────────────────────────────────────────────────────────────────────────
def _render_ml_dashboard(df: pd.DataFrame):
    st.markdown("### 📊 Executive Summary")
    m1, m2, m3, m4 = st.columns(4)
    high = len(df[df["Risk Level"] == "High"])
    med  = len(df[df["Risk Level"] == "Medium"])
    low  = len(df[df["Risk Level"] == "Low"])
    m1.metric("High Risk", high,  delta="Needs attention" if high else "Safe", delta_color="inverse")
    m2.metric("Medium Risk", med)
    m3.metric("Low Risk",    low)
    m4.metric("Avg Confidence", f"{df['Confidence'].mean():.1%}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Risk Distribution")
        st.plotly_chart(px.pie(df, names="Risk Level", color="Risk Level",
            color_discrete_map={"High":"#ef4444","Medium":"#f97316","Low":"#22c55e"}),
            use_container_width=True)
    with c2:
        st.markdown("#### Confidence Score Distribution")
        st.plotly_chart(px.histogram(df, x="Confidence", color="Risk Level", nbins=20,
            color_discrete_map={"High":"#ef4444","Medium":"#f97316","Low":"#22c55e"}),
            use_container_width=True)

    df = df.copy()
    df["Risk_Score"] = df["Risk Level"].map({"High":3,"Medium":2,"Low":1})
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Risk Intensity Across Document")
        st.plotly_chart(px.scatter(df, x=df.index, y="Risk_Score", color="Risk Level",
            color_discrete_map={"High":"#ef4444","Medium":"#f97316","Low":"#22c55e"},
            labels={"index":"Clause Position","Risk_Score":"Intensity"}),
            use_container_width=True)
    with c4:
        st.markdown("#### Top Keywords (High Risk)")
        from collections import Counter
        words = re.findall(r"\b\w+\b",
            " ".join(df[df["Risk Level"]=="High"]["Clause Text"].tolist()).lower())
        stops = {"the","and","to","of","in","a","is","that","for","it","on","be",
                 "as","by","or","this","an","are","with","from","at","not","will"}
        filtered = Counter(w for w in words if w not in stops and len(w)>3).most_common(10)
        if filtered:
            kw = pd.DataFrame(filtered, columns=["Keyword","Count"])
            st.plotly_chart(px.bar(kw, x="Keyword", y="Count", color="Count",
                color_continuous_scale="Reds"), use_container_width=True)
        else:
            st.info("No high-risk clauses detected.")

    st.markdown("---")
    st.markdown("### 🔍 Clause Detail")
    picks = st.multiselect("Filter by risk", ["High","Medium","Low"], default=["High","Medium"])
    for _, row in df[df["Risk Level"].isin(picks)].iterrows():
        risk = row["Risk Level"]
        icon = "🔴" if risk=="High" else "🟠" if risk=="Medium" else "🟢"
        cls  = risk.lower()+"-risk"
        with st.expander(f"{icon} {risk} — {get_summary(row['Clause Text'],90)} ({row['Confidence']:.1%})"):
            st.markdown(f'<div class="risk-card {cls}"><p>{row["Clause Text"]}</p></div>',
                unsafe_allow_html=True)

    st.markdown("---")
    st.download_button("📥 Download CSV", df.drop(columns=["Risk_Score"],errors="ignore")
        .to_csv(index=False).encode(), "contract_risk.csv", "text/csv", key="dl_m1_csv")


# ─────────────────────────────────────────────────────────────────────────────
# Agentic report renderer (replaces markdown blob with interactive expanders)
# ─────────────────────────────────────────────────────────────────────────────
_ACTION_PILL = {
    "Remove Clause":       ("🚫", "pill-remove"),
    "Negotiate Terms":     ("🤝", "pill-negotiate"),
    "Seek Legal Review":   ("⚖️", "pill-legal"),
    "Accept with Caution": ("⚠️", "pill-accept"),
}
_RISK_ICON = {"High":"🔴","Medium":"🟠","Low":"🟢"}


def _render_structured_report(rep: dict):
    """Render the structured_report dict with interactive expanders."""
    b     = rep.get("risk_severity_breakdown", {})
    high  = int(b.get("High",   0))
    med   = int(b.get("Medium", 0))
    low   = int(b.get("Low",    0))
    total = high + med + low

    # ── Overall signal banner ─────────────────────────────────────────────────
    if high >= 2:
        st.error(  "🔴 **HIGH RISK DOCUMENT** — Multiple critical clauses detected. Do not sign without review.")
    elif high == 1:
        st.warning("🟠 **ELEVATED RISK** — At least one critical clause requires negotiation.")
    elif med >= 2:
        st.warning("🟡 **MODERATE RISK** — Several clauses need attention.")
    else:
        st.success("🟢 **LOWER RISK** — Minor concerns only. Standard review recommended.")

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.markdown("#### 📊 Risk Breakdown")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔴 High Risk",   high)
    c2.metric("🟠 Medium Risk", med)
    c3.metric("🟢 Low Risk",    low)
    c4.metric("📄 Total Clauses", total)

    # ── Overview ──────────────────────────────────────────────────────────────
    overview = rep.get("contract_overview", "")
    if overview:
        with st.expander("📌 Contract Overview", expanded=True):
            st.info(overview)

    st.markdown("---")
    st.markdown("### 🔍 Clause-by-Clause Analysis")
    st.caption("Click any clause below to expand the full AI assessment.")

    clauses = rep.get("flagged_clauses_and_mitigation", [])
    # Sort: High → Medium → Low
    order = {"High": 0, "Medium": 1, "Low": 2}
    clauses = sorted(clauses, key=lambda c: order.get(c.get("model_risk_level","Low"), 2))

    for i, row in enumerate(clauses, 1):
        risk    = row.get("model_risk_level", "Low")
        conf    = float(row.get("model_confidence") or 0)
        action  = row.get("action_required", "Seek Legal Review")
        icon    = _RISK_ICON.get(risk, "⚪")
        pill_icon, pill_cls = _ACTION_PILL.get(action, ("⚖️","pill-legal"))
        who     = row.get("who_bears_the_risk", "Unknown")
        preview = row.get("clause_text","")[:80].strip().replace("\n"," ")

        header = (
            f"{icon} **Clause {i}** — {risk} Risk &nbsp;"
            f'<span class="action-pill {pill_cls}">{pill_icon} {action}</span>'
        )

        with st.expander(f"{icon} Clause {i} — {risk} Risk | {action} | {preview}…"):

            # ── Action badge row ───────────────────────────────────────────
            col_a, col_b, col_c = st.columns(3)
            col_a.markdown(f"**Risk Bearer:** {who}")
            col_b.markdown(f"**ML Confidence:** {conf:.0%}")
            col_c.markdown(
                f'<span class="action-pill {pill_cls}">{pill_icon} {action}</span>',
                unsafe_allow_html=True,
            )

            st.markdown("---")

            # ── Original clause text (nested expander) ─────────────────────
            with st.expander("📄 Original Clause Text"):
                st.code(row.get("clause_text","").strip(), language=None)

            # ── Plain English ──────────────────────────────────────────────
            st.markdown("#### 💬 What This Clause Actually Says")
            st.info(row.get("plain_english_summary","—"))

            # ── Why risky ─────────────────────────────────────────────────
            st.markdown("#### ⚠️ Why This Is Risky")
            st.warning(row.get("what_makes_it_risky","—"))

            # ── Severity rationale (why ML rated it this way) ──────────────
            st.markdown("#### 📊 Why The AI Rated It As " + risk)
            st.caption(row.get("severity_rationale","—"))

            # ── Industry standard ──────────────────────────────────────────
            st.markdown("#### 📚 Industry Standard Practice")
            st.success(row.get("industry_standard_practice","—"))

            # ── Negotiation tips ───────────────────────────────────────────
            st.markdown("#### 🤝 Negotiation Tips")
            tips = row.get("negotiation_tips","—")
            st.markdown(tips)

            # ── Safer rewrite ──────────────────────────────────────────────
            st.markdown("#### ✅ Suggested Safer Clause")
            st.code(row.get("safer_rewrite","—"), language=None)

    # ── Action summary table ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Action Summary")
    summary_rows = []
    for i, row in enumerate(clauses, 1):
        risk   = row.get("model_risk_level","?")
        action = row.get("action_required","Seek Legal Review")
        pill_icon, _ = _ACTION_PILL.get(action, ("⚖️","pill-legal"))
        summary_rows.append({
            "#":        i,
            "Risk":     f"{_RISK_ICON.get(risk,'⚪')} {risk}",
            "Action":   f"{pill_icon} {action}",
            "Clause":   row.get("clause_text","")[:70].strip()+"…",
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(rep.get("disclaimer",""))


# ─────────────────────────────────────────────────────────────────────────────
# Milestone 2 — Agentic panel
# ─────────────────────────────────────────────────────────────────────────────
def _render_agentic_panel(raw_text: str, threshold: float, mode: str):
    from contract_agent.kb_retriever import LegalPracticeRetriever
    from contract_agent.ml_utils import load_sklearn_pipeline
    from contract_agent.report import build_structured_report, render_markdown_report
    from contract_agent.text_utils import clean_text, segment_clauses, get_summary
    from contract_agent._shared_prompt import safe_parse_analysis

    is_online  = mode == "online"
    badge_cls  = "badge-online"  if is_online else "badge-offline"
    badge_lbl  = "🌐 Online" if is_online else "🖥 Offline"
    # model_name is only used internally for ollama model selection dropdown
    _local_model = os.environ.get("OLLAMA_MODEL", "qwen3.5:2b")

    # ── Guard checks ──────────────────────────────────────────────────────────
    if is_online and not os.environ.get("OPENROUTER_API_KEY","").strip():
        st.warning("⚠️ Set `OPENROUTER_API_KEY` in your `.env` file to use Online mode.")
        return
    if not is_online:
        h = _ollama_health()
        if not h["reachable"]:
            st.error("❌ **Ollama is not reachable.** Run `ollama serve` in a terminal, then refresh.")
            st.code("ollama serve")
            return

    # ── Welcome card (shown before any run) ───────────────────────────────────
    report_key = f"report_{mode}"
    if report_key not in st.session_state:
        # Check cloud status
        from contract_agent.cloud_client import check_cloud_health
        health = check_cloud_health()
        groq_tag = " + Groq Fallback" if health["groq"] else ""

        reliability = (
            "⚡ Dual-provider reliability — an automatic fallback is ready if the primary is unavailable."
            if health.get("groq") else
            "⚡ Cloud AI is ready to analyse your contract."
        )
        st.markdown(f"""
        <div class="welcome-card">
            <h3 style="margin:0 0 6px 0;">👋 Your AI Legal Assistant is Ready</h3>
            <span class="{badge_cls}">{badge_lbl}</span>
            <p style="margin:16px 0 8px 0; opacity:.85; font-size:1rem;">
                I will scan every clause in your contract, explain what it means in plain English,
                flag what could go wrong, and suggest a safer alternative — grounded in legal best practices.
            </p>
            <p style="margin:0; opacity:.65; font-size:.9rem;">{reliability}</p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📖 About Phase 2 — Agentic AI System", expanded=False):
            st.markdown("""
**The Hybrid Agent** — Phase 1 (Logistic Regression, F1=0.89) detects *which* clauses
are risky. Phase 2 (LangGraph + LLM) explains *why* they are risky and proposes fixes.

| Step | Node | What happens |
|------|------|-------------|
| 1 | **Classify** | TF-IDF + Logistic Regression segments and scores each clause |
| 2 | **Research** | TF-IDF RAG retrieves the 3 most relevant legal best-practice excerpts |
| 3 | **Reason** | LLM receives clause + excerpts → explains risk + proposes safer wording |

The LLM only reasons about *retrieved* content — no hallucinated citations.
            """)

    # ── Run button ────────────────────────────────────────────────────────────
    btn_lbl = "🚀 Run AI Risk Analysis" if is_online else "🖥 Run AI Risk Analysis (Local)"

    if st.button(btn_lbl, type="primary", use_container_width=True, key=f"run_{mode}"):
        # Clear previous report
        st.session_state.pop(report_key, None)

        ml_model   = load_model()
        if ml_model is None:
            st.error("⚠️ ML model not found. Run training first.")
            return

        # ── Step-by-step progress via st.status ──────────────────────────────
        with st.status("⏳ Running AI Agent Pipeline…", expanded=True) as status:

            # Step 1 — ML Classification
            st.write("🔍 **Step 1 / 3** — Classifying clauses with ML model…")
            clauses   = segment_clauses(raw_text)
            flagged   = []
            for clause in clauses:
                cleaned    = clean_text(clause)
                risk       = ml_model.predict([cleaned])[0]
                probs      = ml_model.predict_proba([cleaned])[0]
                confidence = float(max(probs))
                if confidence >= threshold:
                    flagged.append({"clause_text": clause,
                                    "risk_level": risk, "confidence": confidence})

            n_high = sum(1 for f in flagged if f["risk_level"]=="High")
            n_med  = sum(1 for f in flagged if f["risk_level"]=="Medium")
            st.write(f"   ✅ Found **{len(flagged)} clauses** to assess "
                     f"({n_high} High · {n_med} Medium)")

            # Step 2 — RAG Research
            st.write("📚 **Step 2 / 3** — Retrieving legal best-practice guidelines (RAG)…")
            retriever = LegalPracticeRetriever(top_k=3)
            researched = []
            for row in flagged:
                bp = retriever.retrieve(row["clause_text"])
                researched.append({**row, "best_practices": bp})
            st.write(f"   ✅ Retrieved guidelines for all {len(researched)} clauses")

            # Step 3 — LLM Reasoning (per clause)
            backend_label = "Cloud AI" if is_online else "Local AI"
            st.write(f"🤖 **Step 3 / 3** — AI reasoning via **{backend_label}**…")
            st.caption("This is the slowest step — each clause is sent to the AI individually.")

            if is_online:
                from contract_agent.cloud_client import analyze_clause_with_cloud as _analyze
            else:
                from contract_agent.ollama_client import analyze_clause_with_ollama as _analyze

            assessments = []
            progress_bar = st.progress(0, text="Analysing clause 1…")
            for idx, row in enumerate(researched):
                progress_bar.progress(
                    (idx) / max(len(researched), 1),
                    text=f"Analysing clause {idx+1} of {len(researched)}…"
                )
                try:
                    analysis = _analyze(
                        row["clause_text"],
                        str(row["risk_level"]),
                        float(row["confidence"]),
                        row.get("best_practices", []),
                    )
                except Exception as exc:
                    analysis = safe_parse_analysis("{}", fallback_text=f"LLM error: {exc}")
                assessments.append({**row, "analysis": analysis})

            progress_bar.progress(1.0, text=f"✅ All {len(researched)} clauses assessed!")

            # Build report
            overview   = get_summary(raw_text, 500)
            structured = build_structured_report(overview, assessments)
            markdown   = render_markdown_report(structured)

            status.update(label="✅ Analysis complete!", state="complete", expanded=False)

        # Persist
        st.session_state[report_key] = {"structured": structured, "markdown": markdown}
        st.rerun()

    # ── Show persisted report ─────────────────────────────────────────────────
    if report_key in st.session_state:
        data = st.session_state[report_key]
        st.success("✅ Report ready — expand any clause below to read the full assessment.")
        st.markdown("---")
        _render_structured_report(data["structured"])

        # Download buttons
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📥 Download JSON Report",
                json.dumps(data["structured"], indent=2).encode(),
                file_name="risk_report.json",
                mime="application/json",
                key=f"dl_json_{mode}",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                "📥 Download Markdown Report",
                data["markdown"].encode(),
                file_name="risk_report.md",
                mime="text/markdown",
                key=f"dl_md_{mode}",
                use_container_width=True,
            )

        if st.button("🔄 Re-run Analysis", key=f"rerun_{mode}", use_container_width=True):
            st.session_state.pop(report_key, None)
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Session state init
    for k in ("file_data","file_name","file_type"):
        st.session_state.setdefault(k, None)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Settings")

        # AI mode toggle
        st.markdown("#### 🤖 AI Mode")
        mode_choice = st.radio(
            "Select inference backend:",
            ["🌐 Online", "🖥 Offline"],
            index=0, key="mode_radio",
            help="Online uses cloud AI (no setup needed). Offline runs on your local machine — no internet required.",
        )
        selected_mode = "online" if "Online" in mode_choice else "offline"

        # Status indicators — no model names shown
        if selected_mode == "online":
            from contract_agent.cloud_client import check_cloud_health
            health = check_cloud_health()
            if health["openrouter"]:
                st.success("🟢 **Cloud AI — Ready**")
            else:
                st.warning("🔴 **Cloud API key missing**")
        else:
            with st.spinner("Checking local AI…"):
                h = _ollama_health()
            if h["reachable"]:
                st.success("🟢 **Local AI — Ready**")
                avail = h.get("available_models",[])
                if len(avail) > 1:
                    chosen = st.selectbox("Local model:", avail,
                        index=avail.index(h["model"]) if h["model"] in avail else 0,
                        key="ollama_model_select")
                    os.environ["OLLAMA_MODEL"] = chosen
            else:
                st.error("🔴 Local AI not reachable")
                st.caption("Run `ollama serve` in a terminal, then refresh.")

        st.markdown("---")

        # File & confidence
        if st.session_state.file_data is not None:
            st.markdown("#### 📄 Update Document")
            nf = st.file_uploader("Upload New Contract", type=["txt","pdf"],
                key="sidebar_uploader")
            if nf:
                st.session_state.file_data = nf.getvalue()
                st.session_state.file_name = nf.name
                st.session_state.file_type = nf.type
                # Clear any cached reports on new file
                for k in list(st.session_state.keys()):
                    if k.startswith("report_"):
                        del st.session_state[k]
                st.rerun()
            st.markdown("---")

        st.markdown("#### 🎚 Confidence Threshold")
        confidence_threshold = st.slider("Min Confidence Score", 0.0, 1.0, 0.5, 0.05,
            help="Clauses below this score are excluded.")

        st.markdown("---")
        if st.session_state.file_data is not None:
            if st.button("🗑 Clear File", use_container_width=True):
                for k in ["file_data","file_name","file_type"]:
                    st.session_state[k] = None
                for k in list(st.session_state.keys()):
                    if k.startswith("report_"):
                        del st.session_state[k]
                st.rerun()

        st.markdown("---")
        st.caption("⚖️ **Contract Risk Analyzer**\nMilestone 2 — Agentic AI\nAcademic prototype · Not legal advice")

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center; padding:2rem 0 1rem;'>
        <h1 style='font-size:3rem; font-weight:800;
            background:linear-gradient(90deg,#3B82F6,#818CF8,#3B82F6);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;
            margin-bottom:.5rem;'>
            Intelligent Contract Risk Analysis
        </h1>
        <p style='opacity:.75; font-size:1.15rem; font-weight:400;'>
            Uncover hidden risks in legal documents instantly using AI
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='width:50%;margin:6px auto 28px;border:none;border-top:1px solid rgba(128,128,128,.25);'>",
        unsafe_allow_html=True)

    # ── File upload (home state) ──────────────────────────────────────────────
    if st.session_state.file_data is None:
        _, col, _ = st.columns([1, 2, 1])
        with col:
            mf = st.file_uploader("Upload a Contract (PDF or TXT)", type=["txt","pdf"],
                key="main_uploader")
            if mf:
                st.session_state.file_data = mf.getvalue()
                st.session_state.file_name = mf.name
                st.session_state.file_type = mf.type
                st.rerun()
        st.markdown("<div style='height:30px;'></div>", unsafe_allow_html=True)
    else:
        st.success(f"📄 **{st.session_state.file_name}** loaded — scroll down for analysis")

    # ── ML model check ────────────────────────────────────────────────────────
    model = load_model()
    if model is None:
        st.error("⚠️ ML model not found. Run `src/4_train.ipynb` first.")
        return

    # ── Main content ──────────────────────────────────────────────────────────
    if st.session_state.file_data is not None:
        # Parse file
        text = ""
        try:
            if st.session_state.file_type == "application/pdf":
                reader = PyPDF2.PdfReader(BytesIO(st.session_state.file_data))
                for p in reader.pages:
                    t = p.extract_text()
                    if t: text += t + "\n"
            else:
                text = StringIO(st.session_state.file_data.decode("utf-8")).read()
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return

        if not text.strip():
            st.warning("No extractable text found in this file.")
            return

        # Build ML results for Milestone 1 tab
        clauses_ml = segment_clauses(text)
        data_ml    = []
        for clause in clauses_ml:
            cleaned    = clean_text(clause)
            risk       = model.predict([cleaned])[0]
            probs      = model.predict_proba([cleaned])[0]
            conf       = max(probs)
            if conf >= confidence_threshold:
                data_ml.append({"Clause Text": clause,
                                 "Risk Level": risk, "Confidence": conf})
        df = pd.DataFrame(data_ml)

        tab1, tab2 = st.tabs([
            "📊 Milestone 1 — Classical ML",
            "🤖 Milestone 2 — Agentic (LangGraph)",
        ])

        with tab1:
            if df.empty:
                st.warning("No clauses above the confidence threshold.")
            else:
                _render_ml_dashboard(df)

        with tab2:
            _render_agentic_panel(text, confidence_threshold, mode=selected_mode)

    else:
        st.markdown("""
        <div style='text-align:center; padding:70px 40px; opacity:.6;'>
            <div style='font-size:4rem; margin-bottom:1rem;'>📄</div>
            <h3 style='font-weight:500;'>Awaiting contract upload</h3>
            <p style='font-size:1.05rem;'>Upload a PDF or TXT file to begin analysis</p>
        </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()