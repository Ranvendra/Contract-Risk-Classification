import streamlit as st
import pandas as pd
import pickle
import os
import re
import plotly.express as px
import plotly.io as pio
from io import StringIO
import PyPDF2

# make plotly charts dark-compatible
pio.templates.default = "plotly_dark"

# =============================
# Page Config
# =============================
st.set_page_config(
    page_title="Contract Risk Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================
# DARK THEME CSS
# =============================
st.markdown("""
<style>

/* ===== GLOBAL ===== */
html, body, [class*="css"]  {
    font-family: Arial, Helvetica, sans-serif;
    color: #E5E7EB;   /* LIGHT TEXT */
}

/* main page background */
.stApp {
    background-color: #0B1220;
}

/* sidebar */
[data-testid="stSidebar"] {
    background-color: #020617;
}
[data-testid="stSidebar"] * {
    color: #F9FAFB !important;
}

/* headings */
h1 {
    color: #F9FAFB !important;
    font-weight: 800 !important;
}
h2, h3 {
    color: #E5E7EB !important;
    font-weight: 700 !important;
}

/* metric cards */
[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #374151;
    border-radius: 10px;
    padding: 1rem;
}

/* expanders */
.streamlit-expanderHeader {
    color: #F9FAFB !important;
}

/* risk cards */
.risk-card {
    background: #111827;
    padding: 1.2rem;
    border-radius: 10px;
    border: 1px solid #374151;
    margin-bottom: 1rem;
    color: #F3F4F6;
}

.high-risk { border-left: 6px solid #EF4444; }
.medium-risk { border-left: 6px solid #F59E0B; }
.low-risk { border-left: 6px solid #22C55E; }

/* badges */
.risk-badge {
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: bold;
    color: white;
}
.badge-high { background: #EF4444; }
.badge-medium { background: #F59E0B; }
.badge-low { background: #22C55E; }

/* welcome box */
.welcome-card {
    background: #111827;
    padding: 3rem;
    border-radius: 12px;
    border: 1px solid #374151;
    text-align: center;
    color: #E5E7EB;
}

</style>
""", unsafe_allow_html=True)

# =============================
# Load Model
# =============================
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "best_model.pkl")

    if not os.path.exists(model_path):
        return None

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model

# =============================
# Helpers
# =============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^a-z0-9\s.,;:\-\'\"()/]', '', text)
    return text

def segment_clauses(text):
    paragraphs = text.split('\n\n')
    clauses = []
    for para in paragraphs:
        split_para = re.split(r'(?:\n|^)\s*(?:\d+\.|[a-z]\))\s+', para)
        for p in split_para:
            p = p.strip()
            if len(p) > 50:
                clauses.append(p)
    return clauses

def get_summary(text, limit=200):
    return text if len(text) <= limit else text[:limit] + "..."

def get_risk_badge(risk):
    css_class = {"High":"badge-high","Medium":"badge-medium","Low":"badge-low"}.get(risk,"badge-low")
    return f'<span class="risk-badge {css_class}">{risk}</span>'

# =============================
# Chart Colors
# =============================
CHART_COLORS = {
    "High": "#5C0E4C",
    "Medium": "#24064D",
    "Low": "#064E52"
}

PLOTLY_LAYOUT = dict(
    font=dict(color="#E5E7EB"),
    paper_bgcolor="#111827",
    plot_bgcolor="#111827"
)

# =============================
# MAIN APP
# =============================
def main():

    with st.sidebar:
        st.markdown("## Upload Contract")
        uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["txt", "pdf"])

        st.markdown("---")
        st.markdown("### Settings")
        confidence_threshold = st.slider("Minimum Confidence Score", 0.0, 1.0, 0.5, 0.05)

    st.title("Intelligent Contract Risk Analysis")
    st.write("Automated clause-level risk assessment for legal documents")

    model = load_model()

    if model is None:
        st.error("Model not found. Place best_model.pkl inside models folder.")
        return

    if uploaded_file is None:
        st.markdown("""
        <div class="welcome-card">
            <h3>Contract Risk Analyzer</h3>
            <p>Upload a contract to begin analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Read File
    text = ""
    if uploaded_file.type == "application/pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    else:
        text = StringIO(uploaded_file.getvalue().decode("utf-8")).read()

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
                "Confidence": confidence
            })

    df = pd.DataFrame(data)

    if df.empty:
        st.warning("No clauses matched threshold.")
        return

    # Metrics
    st.subheader("Executive Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("High Risk", len(df[df["Risk Level"]=="High"]))
    m2.metric("Medium Risk", len(df[df["Risk Level"]=="Medium"]))
    m3.metric("Low Risk", len(df[df["Risk Level"]=="Low"]))
    m4.metric("Avg Confidence", f"{df['Confidence'].mean():.1%}")

    # Charts
    c1, c2 = st.columns(2)

    with c1:
        fig_pie = px.pie(df, names="Risk Level", color="Risk Level", color_discrete_map=CHART_COLORS, hole=0.4)
        fig_pie.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        fig_hist = px.histogram(df, x="Confidence", color="Risk Level", nbins=20, color_discrete_map=CHART_COLORS)
        fig_hist.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_hist, use_container_width=True)

    # Details
    st.subheader("Detailed Clause Review")

    for _, row in df.iterrows():
        risk = row["Risk Level"]
        badge = get_risk_badge(risk)
        summary = get_summary(row["Clause Text"], 120)

        with st.expander(f"[{risk}] {summary} — {row['Confidence']:.1%}"):
            st.markdown(f"""
            <div class="risk-card {risk.lower()}-risk">
                <p>{badge} Confidence: <strong>{row['Confidence']:.1%}</strong></p>
                <p>{row['Clause Text']}</p>
            """, unsafe_allow_html=True)

    # Download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Report", csv, "contract_risk_analysis.csv", "text/csv")


if __name__ == "__main__":
    main()