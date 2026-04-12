"""
Shared prompt builder for both OpenRouter and Ollama clients.

All analysis output must conform to the ANALYSIS_SCHEMA below.
Centralising here guarantees identical output structure regardless of mode.
"""
from __future__ import annotations
from typing import Any

# ── Output schema (both clients return this shape) ────────────────────────────
ANALYSIS_SCHEMA = {
    "plain_english_summary": str,       # What the clause actually says, in plain language
    "what_makes_it_risky": str,         # Specific legal danger — concrete, not generic
    "who_bears_the_risk": str,          # "Client", "Vendor", "Both", or "Third Party"
    "severity_rationale": str,          # Why the ML model scored it High/Medium/Low
    "industry_standard_practice": str,  # What a balanced, fair version of this clause looks like
    "negotiation_tips": str,            # 2-3 bullet points: what to push back on
    "safer_rewrite": str,               # A complete, ready-to-use replacement clause
    "action_required": str,             # One clear action: "Negotiate", "Remove", "Accept", "Clarify"
}

SCHEMA_KEYS = list(ANALYSIS_SCHEMA.keys())

# ── Output instruction appended to every prompt ───────────────────────────────
_OUTPUT_INSTRUCTION = (
    "Respond with ONLY a single valid JSON object — no markdown, no explanation before or after. "
    "Use exactly these 8 keys:\n"
    '  "plain_english_summary"       – 2-3 sentences: what this clause says in plain English\n'
    '  "what_makes_it_risky"         – Specific legal danger with real-world consequence (2-4 sentences)\n'
    '  "who_bears_the_risk"          – Exactly one of: "Client", "Vendor", "Both", "Third Party"\n'
    '  "severity_rationale"          – Why this clause was rated [RISK_LABEL] (1-2 sentences)\n'
    '  "industry_standard_practice"  – What a fair, balanced clause on this topic looks like (2-3 sentences)\n'
    '  "negotiation_tips"            – 2-3 specific negotiation points as a numbered list (e.g. "1. ... 2. ...")\n'
    '  "safer_rewrite"               – A complete, ready-to-use replacement clause in formal legal style\n'
    '  "action_required"             – Exactly one of: "Remove Clause", "Negotiate Terms", "Seek Legal Review", "Accept with Caution"\n'
)


def build_system_prompt(risk_label: str, confidence: float) -> str:
    """
    Returns a detailed system prompt for legal clause analysis.
    Used identically by OpenRouter and Ollama clients.
    """
    return (
        "You are a Senior Legal Risk Analyst AI with expertise in commercial contract law. "
        "A machine-learning classifier has flagged the clause below as "
        f"[{risk_label.upper()} RISK] with {confidence:.0%} confidence. "
        "Your job is to produce a professional, client-ready risk assessment — written for "
        "a business executive who is NOT a lawyer. Every explanation must be:\n"
        "  • Specific (name the real legal consequence, not a generic warning)\n"
        "  • Practical (explain what actually happens if things go wrong)\n"
        "  • Actionable (tell the client exactly what to do)\n"
        "  • Balanced (acknowledge if part of the clause is reasonable)\n\n"
        "Rules:\n"
        "  • Do NOT invent citations — only reference the best-practice excerpts provided\n"
        "  • Do NOT use phrases like 'consult a lawyer' as your only advice\n"
        "  • The safer_rewrite must be a complete, formal clause — not a summary\n"
        "  • Keep all values as plain strings (no nested JSON)\n\n"
        + _OUTPUT_INSTRUCTION.replace("[RISK_LABEL]", risk_label.upper())
    )


def build_user_message(clause_text: str, best_practices: list[dict[str, Any]]) -> str:
    """Build the user-turn message with the clause + retrieved KB excerpts."""
    bp_blocks: list[str] = []
    for i, bp in enumerate(best_practices, 1):
        bp_blocks.append(
            f"[EXCERPT {i} | Topic: {bp.get('topic', 'General')}]\n"
            f"Title: {bp.get('title', '')}\n"
            f"Standard: {bp.get('text', '')}"
        )

    bp_section = (
        "\n\n---\n\n".join(bp_blocks)
        if bp_blocks
        else "(No best-practice excerpts available for this clause type.)"
    )

    return (
        f"## CLAUSE TO ANALYSE\n\n{clause_text.strip()}\n\n"
        f"## RETRIEVED LEGAL BEST-PRACTICE EXCERPTS\n\n{bp_section}"
    )


def safe_parse_analysis(raw: str, fallback_text: str = "") -> dict[str, str]:
    """
    Parse JSON from LLM output; guarantee all SCHEMA_KEYS exist.
    Never raises — always returns a usable dict.
    """
    import json, re

    s = raw.strip()
    # Strip <think>…</think> blocks (Qwen3.5 / DeepSeek style)
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()
    # Strip markdown fences
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s).strip()

    try:
        data = json.loads(s)
    except (json.JSONDecodeError, ValueError):
        # Attempt to extract a JSON object using regex
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
            except Exception:
                data = {}
        else:
            data = {}

    # Map old 3-key output → new 8-key schema (backward compat)
    if "legal_concern" in data and "what_makes_it_risky" not in data:
        data["what_makes_it_risky"] = data.pop("legal_concern", "")
        data["industry_standard_practice"] = data.pop("comparison_to_best_practice", "")
        data.setdefault("safer_rewrite", data.pop("safer_rewrite", ""))

    # Ensure all schema keys present
    defaults = {
        "plain_english_summary": fallback_text or "Analysis could not be completed.",
        "what_makes_it_risky": "Analysis could not be completed.",
        "who_bears_the_risk": "Unknown",
        "severity_rationale": "Model flagged this clause based on statistical patterns.",
        "industry_standard_practice": "No best-practice data available.",
        "negotiation_tips": "1. Consult a legal professional before signing.",
        "safer_rewrite": "Clause requires manual legal review.",
        "action_required": "Seek Legal Review",
    }
    for key, default_val in defaults.items():
        if not data.get(key):
            data[key] = default_val

    return {k: str(data.get(k, "")) for k in SCHEMA_KEYS}
