"""
contract_agent/llm/prompting.py — Component Logic
=========================================================

Centralises all prompt construction so both cloud and local LLM clients
produce identical output structures regardless of the provider.

Domain-aware: injects contract domain into system prompt and user message
for higher-precision, domain-specific legal analysis.
"""
from __future__ import annotations

from typing import Any

# ── Output schema ─────────────────────────────────────────────────────────────

ANALYSIS_SCHEMA = {
    "plain_english_summary":      str,
    "what_makes_it_risky":        str,
    "who_bears_the_risk":         str,
    "severity_rationale":         str,
    "industry_standard_practice": str,
    "negotiation_tips":           str,
    "safer_rewrite":              str,
    "action_required":            str,
}

SCHEMA_KEYS = list(ANALYSIS_SCHEMA.keys())

# ── Output instruction ────────────────────────────────────────────────────────

_OUTPUT_INSTRUCTION = (
    "Respond with ONLY a single valid JSON object — no markdown fences, no explanation before or after. "
    "Use exactly these 8 keys:\n"
    '  "plain_english_summary"       – 2-3 sentences: what this clause means in plain English\n'
    '  "what_makes_it_risky"         – Specific legal danger with a real-world consequence (2-4 sentences)\n'
    '  "who_bears_the_risk"          – Exactly one of: "Client", "Vendor", "Both", "Third Party"\n'
    '  "severity_rationale"          – Why this clause was rated [RISK_LABEL] (1-2 sentences)\n'
    '  "industry_standard_practice"  – What a fair, balanced clause looks like per domain best practice (2-3 sentences)\n'
    '  "negotiation_tips"            – 2-3 specific, actionable negotiation points as a numbered list\n'
    '  "safer_rewrite"               – A complete, ready-to-use replacement clause in formal legal style\n'
    '  "action_required"             – Exactly one of: "Remove Clause", "Negotiate Terms", "Seek Legal Review", "Accept with Caution"\n'
)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────

def build_system_prompt(
    risk_label:  str,
    confidence:  float,
    domain:      str = "General",
) -> str:
    """
    Returns a domain-aware system prompt for legal clause analysis.

    Args:
        risk_label:  High | Medium | Low (from ML classifier)
        confidence:  ML classifier confidence score (0–1)
        domain:      Detected contract domain (NDA/Employment/Lease/SaaS/Vendor/General)
    """
    domain_context = (
        f"This is a **{domain} contract**. "
        f"Apply {domain}-specific legal standards, jurisdiction-specific rules, "
        f"and domain best practices in your analysis. "
    ) if domain and domain != "General" else (
        "This is a general commercial contract. "
    )

    return (
        "You are a Senior Legal Risk Analyst AI. "
        f"{domain_context}"
        "A machine-learning classifier has flagged the clause below as "
        f"[{risk_label.upper()} RISK] with {confidence:.0%} confidence. "
        "Your job is to produce a risk assessment written in EXTREMELY SIMPLE, EVERYDAY ENGLISH (7th-8th grade reading level). "
        "Assume the reader is a middle-school student or a beginner with zero legal knowledge. "
        "DO NOT use complex legal jargon, Latin terms, or long, complicated sentences.\n\n"
        "Every explanation must be:\n"
        "  • Simple — use everyday words instead of legal terms (e.g., use 'promise' instead of 'covenant')\n"
        "  • Specific — name the real consequence, like 'you might have to pay extra money'\n"
        "  • Practical — explain what goes wrong in real life clearly\n"
        "  • Actionable — give clear, simple steps to take\n\n"
        "Rules:\n"
        "  • For document headers, exhibit labels, or incomplete sentences (e.g. 'EXHIBIT 10.1', 'EXECUTION COPY'), output a proper English sentence like 'This appears to be a standard document heading or exhibit label rather than a substantive clause.' DO NOT just repeat the raw text.\n"
        "  • DO NOT invent citations — only reference the domain guidelines provided\n"
        "  • DO NOT use 'consult a lawyer' as your only advice — explain the exact risk first\n"
        "  • The safer_rewrite must be a complete, formal clause — not a summary\n"
        "  • Keep all JSON values as plain strings (no nested JSON or markdown inside values)\n"
        "  • Keep responses focused and not excessively long\n\n"
        + _OUTPUT_INSTRUCTION.replace("[RISK_LABEL]", risk_label.upper())
    )


def build_user_message(
    clause_text:    str,
    best_practices: list[dict[str, Any]],
    domain:         str = "General",
) -> str:
    """
    Build the user-turn message with clause text and domain-filtered KB excerpts.
    """
    bp_blocks: list[str] = []
    for i, bp in enumerate(best_practices, 1):
        bp_domain = bp.get("domain", domain)
        bp_blocks.append(
            f"[EXCERPT {i} | Domain: {bp_domain} | Topic: {bp.get('topic', 'General')}]\n"
            f"Title: {bp.get('title', '')}\n"
            f"Standard: {bp.get('text', '')}"
        )

    bp_section = (
        "\n\n---\n\n".join(bp_blocks)
        if bp_blocks
        else "(No domain-specific best-practice excerpts available for this clause.)"
    )

    return (
        f"## CONTRACT DOMAIN: {domain}\n\n"
        f"## CLAUSE TO ANALYSE\n\n{clause_text.strip()}\n\n"
        f"## RETRIEVED {domain.upper()}-SPECIFIC LEGAL BEST-PRACTICE EXCERPTS\n\n{bp_section}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Safe JSON parser
# ─────────────────────────────────────────────────────────────────────────────

def safe_parse_analysis(raw: str, fallback_text: str = "") -> dict[str, str]:
    """
    Parse JSON from LLM output and guarantee all SCHEMA_KEYS are populated.
    Never raises — always returns a usable dict.
    """
    import json
    import re

    s = raw.strip()
    # Strip <think>…</think> blocks (Qwen / DeepSeek reasoning models)
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()
    # Strip markdown code fences
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$",          "", s).strip()

    try:
        data = json.loads(s)
    except (json.JSONDecodeError, ValueError):
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
            except Exception:
                data = {}
        else:
            data = {}

    # Backward-compat: remap old 3-key schema to new 8-key schema
    if "legal_concern" in data and "what_makes_it_risky" not in data:
        data["what_makes_it_risky"]        = data.pop("legal_concern", "")
        data["industry_standard_practice"] = data.pop("comparison_to_best_practice", "")

    # Ensure every schema key is present with a sensible default
    defaults = {
        "plain_english_summary":      fallback_text or "Analysis could not be completed.",
        "what_makes_it_risky":        "Analysis could not be completed.",
        "who_bears_the_risk":         "Unknown",
        "severity_rationale":         "Model flagged this clause based on statistical risk patterns.",
        "industry_standard_practice": "No domain-specific best-practice data available.",
        "negotiation_tips":           "1. Consult a qualified legal professional before signing.",
        "safer_rewrite":              "Clause requires manual legal review before execution.",
        "action_required":            "Seek Legal Review",
    }
    for key, default_val in defaults.items():
        if not data.get(key):
            data[key] = default_val

    return {k: str(data.get(k, "")) for k in SCHEMA_KEYS}
