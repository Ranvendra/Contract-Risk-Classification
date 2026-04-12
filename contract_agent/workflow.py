from functools import lru_cache
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from contract_agent.kb_retriever import LegalPracticeRetriever
from contract_agent.ml_utils import load_sklearn_pipeline
from contract_agent.openrouter_client import analyze_clause_with_openrouter
from contract_agent.ollama_client import analyze_clause_with_ollama
from contract_agent._shared_prompt import safe_parse_analysis
from contract_agent.report import (
    DISCLAIMER,
    build_structured_report,
    render_markdown_report,
)
from contract_agent.text_utils import clean_text, get_summary, segment_clauses

# Supported modes
Mode = Literal["online", "offline"]


class AgentState(TypedDict, total=False):
    raw_text: str
    confidence_threshold: float
    mode: str                          # "online" | "offline"
    contract_overview: str
    flagged_clauses: list[dict[str, Any]]
    researched: list[dict[str, Any]]
    clause_assessments: list[dict[str, Any]]
    structured_report: dict[str, Any]
    markdown_report: str
    error: str | None


# ---------------------------------------------------------------------------
# Node: Classification
# ---------------------------------------------------------------------------

def _classify_node(state: AgentState) -> dict[str, Any]:
    model = load_sklearn_pipeline()
    if model is None:
        return {"error": "Trained model not found at models/best_model.pkl."}

    thr = float(state.get("confidence_threshold", 0.5))
    flagged: list[dict[str, Any]] = []
    for clause in segment_clauses(state["raw_text"]):
        cleaned = clean_text(clause)
        risk = model.predict([cleaned])[0]
        probs = model.predict_proba([cleaned])[0]
        confidence = float(max(probs))
        if confidence >= thr:
            flagged.append(
                {
                    "clause_text": clause,
                    "risk_level": risk,
                    "confidence": confidence,
                }
            )

    return {
        "flagged_clauses": flagged,
        "contract_overview": get_summary(state["raw_text"], 500),
        "error": None,
    }


# ---------------------------------------------------------------------------
# Node: RAG Research
# ---------------------------------------------------------------------------

def _research_node(state: AgentState) -> dict[str, Any]:
    if state.get("error"):
        return {}
    retriever = LegalPracticeRetriever(top_k=3)
    researched = []
    for row in state.get("flagged_clauses", []):
        bp = retriever.retrieve(row["clause_text"])
        researched.append({**row, "best_practices": bp})
    return {"researched": researched}


# ---------------------------------------------------------------------------
# Node: LLM Reasoning (online = OpenRouter / offline = Ollama)
# ---------------------------------------------------------------------------

def _reason_node(state: AgentState) -> dict[str, Any]:
    if state.get("error"):
        return {
            "structured_report": {
                "contract_overview": state.get("contract_overview", ""),
                "risk_severity_breakdown": {"High": 0, "Medium": 0, "Low": 0},
                "flagged_clauses_and_mitigation": [],
                "disclaimer": DISCLAIMER,
            },
            "markdown_report": (
                f"## Error\n\n{state.get('error')}\n\n"
                f"## Disclaimer\n\n{DISCLAIMER}"
            ),
            "clause_assessments": [],
        }

    mode: str = state.get("mode", "online")
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
                )
            else:
                analysis = analyze_clause_with_openrouter(
                    row["clause_text"],
                    str(row["risk_level"]),
                    float(row["confidence"]),
                    row.get("best_practices", []),
                )
        except Exception as exc:
            analysis = safe_parse_analysis(
                "{}",
                fallback_text=f"LLM step failed: {exc}",
            )
        assessments.append({**row, "analysis": analysis})

    structured = build_structured_report(
        state.get("contract_overview", ""),
        assessments,
    )
    md = render_markdown_report(structured)
    return {
        "clause_assessments": assessments,
        "structured_report": structured,
        "markdown_report": md,
    }


# ---------------------------------------------------------------------------
# Graph builder (cached per mode so each mode gets its own compiled graph)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=2)
def _build_compiled_graph():
    g = StateGraph(AgentState)
    g.add_node("classify", _classify_node)
    g.add_node("research", _research_node)
    g.add_node("reason", _reason_node)
    g.add_edge(START, "classify")
    g.add_edge("classify", "research")
    g.add_edge("research", "reason")
    g.add_edge("reason", END)
    return g.compile()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_hybrid_agent_pipeline(
    raw_text: str,
    confidence_threshold: float = 0.5,
    mode: Mode = "online",
) -> AgentState:
    """
    Run the full LangGraph classify → research → reason pipeline.

    Args:
        raw_text: Full contract text.
        confidence_threshold: Min ML model confidence to include a clause.
        mode: "online"  → OpenRouter (cloud LLM)
              "offline" → Ollama (local LLM, no internet needed)

    Returns:
        Final AgentState with `structured_report` and `markdown_report`.
    """
    graph = _build_compiled_graph()
    return graph.invoke(
        {
            "raw_text": raw_text,
            "confidence_threshold": confidence_threshold,
            "mode": mode,
        }
    )
