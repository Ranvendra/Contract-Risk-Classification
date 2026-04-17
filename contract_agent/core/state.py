"""
Global Agent State Types
"""
from __future__ import annotations

from typing import Any, Literal, TypedDict

Mode = Literal["online", "offline"]

class AgentState(TypedDict, total=False):
    """
    State object passed immutably between LangGraph nodes.
    """
    raw_text:             str
    domain:               str               # detected contract domain
    confidence_threshold: float
    mode:                 Mode              # "online" | "offline"
    contract_overview:    str
    flagged_clauses:      list[dict[str, Any]]   # High + Medium only
    researched:           list[dict[str, Any]]
    clause_assessments:   list[dict[str, Any]]
    structured_report:    dict[str, Any]
    markdown_report:      str
    error:                str | None
