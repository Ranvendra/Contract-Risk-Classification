"""
contract_agent/core/domain.py — Component Logic
=============================================================

Identifies the contract domain (NDA, Employment, Lease, SaaS, Vendor, General)
using a two-stage strategy:

  Stage 1: Keyword scoring (instant, free, no API calls)
  Stage 2: LLM fast-call fallback only if keyword confidence is low

Returns one of: NDA | Employment | Lease | SaaS | Vendor | General
"""
from __future__ import annotations

import os
import re
from typing import Callable

# ─────────────────────────────────────────────────────────────────────────────
# Supported domains
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_DOMAINS: list[str] = [
    "NDA",
    "Employment",
    "Lease",
    "SaaS",
    "Vendor",
    "General",
]

# ─────────────────────────────────────────────────────────────────────────────
# Keyword signal table
# Each entry: (regex_pattern, score_weight)
# Higher weight = stronger domain signal
# ─────────────────────────────────────────────────────────────────────────────

_DOMAIN_SIGNALS: dict[str, list[tuple[str, int]]] = {
    "NDA": [
        (r"\bnon.?disclosure\b",          5),
        (r"\bnda\b",                       5),
        (r"\bdisclosing\s+party\b",        5),
        (r"\breceiving\s+party\b",         5),
        (r"\bproprietary\s+information\b", 4),
        (r"\btrade\s+secret",              4),
        (r"\bconfidential(?:ity)?\b",      2),
    ],
    "Employment": [
        (r"\bemployee\b",                  4),
        (r"\bemployer\b",                  4),
        (r"\bat.will\b",                   5),
        (r"\bsalary\b",                    4),
        (r"\bnon.compete\b",               5),
        (r"\bseverance\b",                 5),
        (r"\bwages?\b",                    4),
        (r"\bbenefits?\b",                 3),
        (r"\bwork(?:ing)?\s+hours?\b",     3),
        (r"\bontract\s+of\s+employment\b", 5),
        (r"\bpayroll\b",                   4),
    ],
    "Lease": [
        (r"\blandlord\b",                  5),
        (r"\btenant\b",                    5),
        (r"\blease\s+agreement\b",         5),
        (r"\brent(?:al)?\b",               4),
        (r"\bpremises\b",                  4),
        (r"\bsecurity\s+deposit\b",        5),
        (r"\beviction\b",                  5),
        (r"\btenancy\b",                   5),
        (r"\bsubletting?\b",               4),
    ],
    "SaaS": [
        (r"\bsoftware.as.a.service\b",     5),
        (r"\bsaas\b",                      5),
        (r"\bsubscription\b",              4),
        (r"\buptime\b",                    5),
        (r"\bservice\s+level\s+agreement\b", 5),
        (r"\b\bsla\b",                     4),
        (r"\bcloud\s+service\b",           4),
        (r"\bapi\b",                       3),
        (r"\bplatform\s+service\b",        4),
        (r"\bauto.renew\b",                4),
    ],
    "Vendor": [
        (r"\bvendor\b",                    5),
        (r"\bsupplier\b",                  5),
        (r"\bpurchase\s+order\b",          5),
        (r"\bdelivery\b",                  3),
        (r"\bgoods\b",                     4),
        (r"\bmanufactur",                  4),
        (r"\bincoterms?\b",                5),
        (r"\bprocurement\b",               4),
        (r"\bsupply\s+chain\b",            5),
        (r"\bwarranty\s+period\b",         3),
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Core detection logic
# ─────────────────────────────────────────────────────────────────────────────

def _keyword_score(text: str) -> dict[str, int]:
    """Score each domain by counting weighted keyword hits."""
    t = text.lower()
    scores: dict[str, int] = {d: 0 for d in SUPPORTED_DOMAINS}
    for domain, signals in _DOMAIN_SIGNALS.items():
        for pattern, weight in signals:
            hits = len(re.findall(pattern, t))
            scores[domain] += hits * weight
    return scores


def detect_domain(
    text: str,
    llm_fallback: bool = True,
    text_limit: int = 3000,
) -> str:
    """
    Detect the contract domain from text.

    Args:
        text:         Full contract text (only first `text_limit` chars are scanned).
        llm_fallback: If True and keyword confidence is low, attempt a single-shot
                      LLM call for disambiguation.
        text_limit:   Characters from doc start to analyse (default 3000).

    Returns:
        One of: NDA | Employment | Lease | SaaS | Vendor | General
    """
    sample = text[:text_limit]
    scores = _keyword_score(sample)

    best_domain = max(scores, key=lambda d: scores[d] if d != "General" else -1)
    best_score  = scores[best_domain]

    # High confidence — keyword match is unambiguous
    if best_score >= 8:
        return best_domain

    # Medium confidence — use keyword result but warn internally
    if best_score >= 4:
        return best_domain

    # Low confidence — try LLM for disambiguation if allowed
    if llm_fallback and best_score < 4:
        try:
            llm_domain = _llm_detect(text[:2000])
            if llm_domain in SUPPORTED_DOMAINS:
                return llm_domain
        except Exception:
            pass  # silently fall through to General

    return "General"


def _llm_detect(text: str) -> str:
    """
    Single low-cost LLM call to identify the contract domain.
    Uses Groq (fastest/cheapest) → OpenRouter as fallback.
    """
    from openai import OpenAI

    groq_key   = os.environ.get("GROQ_API_KEY",         "").strip()
    or_key_1   = os.environ.get("OPENROUTER_API_KEY_1", "").strip()
    or_key_2   = os.environ.get("OPENROUTER_API_KEY_2", "").strip()
    or_key     = or_key_1 or or_key_2

    if not groq_key and not or_key:
        return "General"

    prompt = (
        "You are a contract classification expert. Based on the contract excerpt below, "
        "identify which ONE domain it belongs to. "
        "Reply with ONLY one word from this exact list: "
        "NDA, Employment, Lease, SaaS, Vendor, General\n\n"
        f"Contract excerpt:\n{text[:1500]}"
    )

    def _call(client: "OpenAI", model: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5,
            timeout=10.0,
        )
        raw = (resp.choices[0].message.content or "General").strip()
        for domain in SUPPORTED_DOMAINS:
            if domain.lower() in raw.lower():
                return domain
        return "General"

    # Try Groq first (fastest)
    if groq_key:
        try:
            client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=groq_key,
            )
            model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
            return _call(client, model)
        except Exception:
            pass

    # Fallback to OpenRouter
    if or_key:
        try:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=or_key,
            )
            model = os.environ.get("OPENROUTER_MODEL", "openai/gpt-oss-120b")
            return _call(client, model)
        except Exception:
            pass

    return "General"


# ─────────────────────────────────────────────────────────────────────────────
# Domain metadata helpers (used by UI)
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_META: dict[str, dict] = {
    "NDA":        {"icon": "", "color": "#818CF8", "label": "Non-Disclosure Agreement"},
    "Employment": {"icon": "", "color": "#F59E0B", "label": "Employment Contract"},
    "Lease":      {"icon": "", "color": "#10B981", "label": "Lease / Real Estate"},
    "SaaS":       {"icon": "", "color": "#3B82F6", "label": "SaaS / Software License"},
    "Vendor":     {"icon": "", "color": "#EC4899", "label": "Vendor / Supply Agreement"},
    "General":    {"icon": "", "color": "#6B7280", "label": "General Commercial"},
}


def get_domain_badge_html(domain: str) -> str:
    """Return an HTML badge string for the detected domain."""
    meta  = DOMAIN_META.get(domain, DOMAIN_META["General"])
    icon  = meta["icon"]
    color = meta["color"]
    label = meta["label"]
    icon_part = f"{icon} " if icon else ""
    return (
        f'<span style="display:inline-flex;align-items:center;gap:6px;'
        f'padding:5px 14px;border-radius:20px;font-size:.82rem;font-weight:700;'
        f'background:{color}22;color:{color};border:1px solid {color}55;">'
        f'{icon_part}{label}</span>'
    )
