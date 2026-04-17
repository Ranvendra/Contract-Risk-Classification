"""
contract_agent/workflow.py — LangGraph Agent Pipeline (Milestone 2)
====================================================================

Graph topology:
  START → domain_detect → classify → research → reason → END

Nodes:
  domain_detect  Infers contract domain (NDA/Employment/Lease/SaaS/Vendor/General)
  classify       ML risk classification; passes ONLY High + Medium clauses forward
  research       Domain-filtered ChromaDB RAG retrieval per clause
  reason         LLM analysis and mitigation per clause

State is a typed dict passed immutably between nodes.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from contract_agent.domain_detector import detect_domain
from contract_agent.kb_retriever    import DomainAwareRetriever
from contract_agent.ml_utils        import load_sklearn_pipeline
from contract_agent.cloud_client    import analyze_clause_with_cloud
from contract_agent.ollama_client   import analyze_clause_with_ollama
from contract_agent._shared_prompt  import safe_parse_analysis
from contract_agent.report import (
    DISCLAIMER,
    build_structured_report,
    render_markdown_report,
)
from contract_agent.text_utils import clean_text, get_summary, segment_clauses

Mode = Literal["online", "offline"]

# Maximum characters per clause chunk sent to the LLM
_MAX_CLAUSE_CHARS = 2000


# ─────────────────────────────────────────────────────────────────────────────
# Agent State
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    raw_text:             str
    domain:               str               # detected contract domain
    confidence_threshold: float
    mode:                 str               # "online" | "offline"
    contract_overview:    str
    flagged_clauses:      list[dict[str, Any]]   # High + Medium only
    researched:           list[dict[str, Any]]
    clause_assessments:   list[dict[str, Any]]
    structured_report:    dict[str, Any]
    markdown_report:      str
    error:                str | None


# ─────────────────────────────────────────────────────────────────────────────
# Node 1: Domain Detection
# ─────────────────────────────────────────────────────────────────────────────

def _domain_detect_node(state: AgentState) -> dict[str, Any]:
    """Infer the contract domain from the raw text using keyword scoring + LLM fallback."""
    domain = detect_domain(
        state["raw_text"],
        llm_fallback=True,
    )
    return {
        "domain":           domain,
        "contract_overview": get_summary(state["raw_text"], 500),
        "error":            None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 2: ML Classification (High + Medium only — Low is filtered out)
# ─────────────────────────────────────────────────────────────────────────────

def _classify_node(state: AgentState) -> dict[str, Any]:
    """
    Segment the contract, classify each clause, and return ONLY High and
    Medium risk clauses above the confidence threshold.

    Low-risk clauses are deliberately excluded to save LLM tokens.
    """
    if state.get("error"):
        return {}

    model = load_sklearn_pipeline()
    if model is None:
        return {"error": "Trained model not found at models/best_model.pkl. Run training first."}

    thr = float(state.get("confidence_threshold", 0.5))
    flagged: list[dict[str, Any]] = []

    for clause in segment_clauses(state["raw_text"]):
        cleaned    = clean_text(clause)
        risk       = model.predict([cleaned])[0]
        probs      = model.predict_proba([cleaned])[0]
        confidence = float(max(probs))

        # Only keep High and Medium — skip Low to reduce LLM cost
        if confidence >= thr and risk in ("High", "Medium"):
            # Chunk very long clauses to stay within LLM context limits
            text_to_use = clause[:_MAX_CLAUSE_CHARS]
            if len(clause) > _MAX_CLAUSE_CHARS:
                text_to_use += "… [truncated for analysis]"

            flagged.append({
                "clause_text": text_to_use,
                "risk_level":  risk,
                "confidence":  confidence,
            })

    return {
        "flagged_clauses": flagged,
        "error":           None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 3: Domain-Aware RAG Research
# ─────────────────────────────────────────────────────────────────────────────

def _research_node(state: AgentState) -> dict[str, Any]:
    """
    For each flagged clause, retrieve the top-k most relevant legal guidelines
    from ChromaDB, filtered to the detected contract domain.
    """
    if state.get("error"):
        return {}

    domain    = state.get("domain", "General")
    retriever = DomainAwareRetriever(top_k=3)
    researched: list[dict[str, Any]] = []

    for row in state.get("flagged_clauses", []):
        guidelines = retriever.retrieve(row["clause_text"], domain=domain)
        researched.append({**row, "best_practices": guidelines})

    return {"researched": researched}


# ─────────────────────────────────────────────────────────────────────────────
# Node 4: LLM Analysis and Mitigation
# ─────────────────────────────────────────────────────────────────────────────

def _reason_node(state: AgentState) -> dict[str, Any]:
    """
    Call the LLM for each researched clause to generate:
      - Plain-English explanation
      - Why it is risky (specific, not generic)
      - Mitigation steps and safer rewrite
    """
    if state.get("error"):
        return {
            "structured_report": {
                "contract_overview":         state.get("contract_overview", ""),
                "domain":                    state.get("domain", "General"),
                "risk_severity_breakdown":   {"High": 0, "Medium": 0, "Low": 0},
                "flagged_clauses_and_mitigation": [],
                "disclaimer": DISCLAIMER,
            },
            "markdown_report": (
                f"## Error\n\n{state.get('error')}\n\n## Disclaimer\n\n{DISCLAIMER}"
            ),
            "clause_assessments": [],
        }

    mode       = state.get("mode", "online")
    domain     = state.get("domain", "General")
    researched = state.get("researched", [])
    assessments: list[dict[str, Any]] = []

    for row in researched:
        try:
            if mode == "offline":
                analysis = analyze_clause_with_ollama(
                    row["clause_text"],
                    str(row["risk_level"]),
                    float(row["confidence"]),
                    row.get("best_practices", []),
                    domain=domain,
                )
            else:
                analysis = analyze_clause_with_cloud(
                    row["clause_text"],
                    str(row["risk_level"]),
                    float(row["confidence"]),
                    row.get("best_practices", []),
                    domain=domain,
                )
        except Exception as exc:
            analysis = safe_parse_analysis(
                "{}",
                fallback_text=f"LLM analysis failed: {exc}",
            )
        assessments.append({**row, "analysis": analysis})

    overview   = state.get("contract_overview", "")
    structured = build_structured_report(overview, assessments, domain=domain)
    md         = render_markdown_report(structured)

    return {
        "clause_assessments": assessments,
        "structured_report":  structured,
        "markdown_report":    md,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _build_compiled_graph():
    g = StateGraph(AgentState)
    g.add_node("domain_detect", _domain_detect_node)
    g.add_node("classify",      _classify_node)
    g.add_node("research",      _research_node)
    g.add_node("reason",        _reason_node)

    g.add_edge(START,          "domain_detect")
    g.add_edge("domain_detect", "classify")
    g.add_edge("classify",      "research")
    g.add_edge("research",      "reason")
    g.add_edge("reason",        END)
    return g.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_hybrid_agent_pipeline(
    raw_text: str,
    confidence_threshold: float = 0.5,
    mode: Mode = "online",
    domain: str | None = None,
) -> AgentState:
    """
    Run the full LangGraph pipeline:
        domain_detect → classify → research (domain-aware RAG) → reason (LLM)

    Args:
        raw_text:             Full contract text.
        confidence_threshold: Min ML model confidence to include a clause.
        mode:                 "online" (cloud LLM) | "offline" (local Ollama).
        domain:               Pre-detected domain (skips domain_detect node if provided).

    Returns:
        Final AgentState with `structured_report` and `markdown_report`.
    """
    graph = _build_compiled_graph()
    init: AgentState = {
        "raw_text":             raw_text,
        "confidence_threshold": confidence_threshold,
        "mode":                 mode,
    }
    if domain:
        init["domain"] = domain   # bypass LLM detection if already known

    return graph.invoke(init)
