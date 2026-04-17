"""
Unified Cloud LLM Client — 3-level fallback chain.

Priority order:
  1. OpenRouter Key 1  (OPENROUTER_API_KEY_1)
  2. OpenRouter Key 2  (OPENROUTER_API_KEY_2)
  3. Groq              (GROQ_API_KEY)

Each provider is tried in order. The first one that succeeds returns the result.
The pipeline never fails as long as at least one key is valid.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from openai import OpenAI

from contract_agent.llm.prompting import (
    build_system_prompt,
    build_user_message,
    safe_parse_analysis,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Internal: single-provider call
# ─────────────────────────────────────────────────────────────────────────────

def _call_openrouter(api_key: str, model: str, system: str, user: str) -> str:
    """Returns raw LLM response string. Raises on any error."""
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.15,
        max_tokens=1800,
        response_format={"type": "json_object"},
        timeout=60.0,
    )
    return completion.choices[0].message.content or "{}"


def _call_groq(api_key: str, model: str, system: str, user: str) -> str:
    """Returns raw LLM response string. Raises on any error."""
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.15,
        max_tokens=1500,
        response_format={"type": "json_object"},
        timeout=30.0,
    )
    return completion.choices[0].message.content or "{}"


def _call_pollinations(model: str, system: str, user: str) -> str:
    """Returns raw LLM response string via Pollinations AI free endpoint."""
    client = OpenAI(
        base_url="https://text.pollinations.ai/openai",
        api_key="dummy",
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.15,
        response_format={"type": "json_object"},
        timeout=30.0,
    )
    return completion.choices[0].message.content or "{}"


# ─────────────────────────────────────────────────────────────────────────────
# Public: main analysis function with 3-level fallback
# ─────────────────────────────────────────────────────────────────────────────

def analyze_clause_with_cloud(
    clause_text: str,
    risk_label: str,
    confidence: float,
    best_practices: list[dict[str, Any]],
    domain: str = "General",
) -> dict[str, str]:
    """
    Analyses a clause using cloud LLMs with automatic failover:
      1. OpenRouter Key 1  → 2. OpenRouter Key 2  → 3. Groq

    Args:
        domain: Detected contract domain for domain-aware prompt construction.

    Returns the standard 8-key analysis dict.
    """
    or_model    = os.environ.get("OPENROUTER_MODEL", "openai/gpt-oss-120b").strip()
    groq_model  = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile").strip()

    or_key_1    = os.environ.get("OPENROUTER_API_KEY_1", "").strip()
    or_key_2    = os.environ.get("OPENROUTER_API_KEY_2", "").strip()
    groq_key    = os.environ.get("GROQ_API_KEY", "").strip()

    system = build_system_prompt(risk_label, confidence, domain=domain)
    user   = build_user_message(clause_text, best_practices, domain=domain)

    # ── Attempt 1: Pollinations (Free & Always Available) ─────────────────────
    try:
        raw = _call_pollinations("openai", system, user)
        logger.info("✅ Pollinations Free API succeeded.")
        return safe_parse_analysis(raw, fallback_text=clause_text[:200])
    except Exception as exc:
        logger.warning(f"⚠️  Pollinations API failed: {exc}")

    # ── Attempt 2: OpenRouter Key 1 ───────────────────────────────────────────
    if or_key_1:
        try:
            raw = _call_openrouter(or_key_1, or_model, system, user)
            logger.info("✅ OpenRouter Key 1 succeeded.")
            return safe_parse_analysis(raw, fallback_text=clause_text[:200])
        except Exception as exc:
            logger.warning(f"⚠️  OpenRouter Key 1 failed: {exc}")

    # ── Attempt 3: OpenRouter Key 2 ───────────────────────────────────────────
    if or_key_2:
        try:
            raw = _call_openrouter(or_key_2, or_model, system, user)
            logger.info("✅ OpenRouter Key 2 succeeded (Key 1 fallback).")
            return safe_parse_analysis(raw, fallback_text=clause_text[:200])
        except Exception as exc:
            logger.warning(f"⚠️  OpenRouter Key 2 failed: {exc}")

    # ── Attempt 4: Deepseek via OpenRouter ────────────────────────────────────
    if or_key_1 or or_key_2:
        try:
            k = or_key_1 or or_key_2
            raw = _call_openrouter(k, "deepseek/deepseek-chat", system, user)
            logger.info("✅ Deepseek via OpenRouter (Emergency Fallback) succeeded.")
            return safe_parse_analysis(raw, fallback_text=clause_text[:200])
        except Exception as exc:
            logger.warning(f"⚠️  Deepseek fallback failed: {exc}")

    # ── Attempt 5: Groq ───────────────────────────────────────────────────────
    if groq_key:
        try:
            raw = _call_groq(groq_key, groq_model, system, user)
            logger.info("✅ Groq succeeded (final fallback).")
            return safe_parse_analysis(raw, fallback_text=clause_text[:200])
        except Exception as exc:
            logger.error(f"❌ Groq fallback failed: {exc}")

    # ── All failed ────────────────────────────────────────────────────────────
    configured = [k for k, v in {
        "OpenRouter Key 1": or_key_1,
        "OpenRouter Key 2": or_key_2,
        "Groq": groq_key,
    }.items() if v]

    error_msg = (
        f"All cloud providers failed. Tried: {', '.join(configured) or 'none configured'}."
        if configured else "No cloud API keys configured in .env."
    )
    logger.error(error_msg)
    return safe_parse_analysis("{}", fallback_text=error_msg)


# ─────────────────────────────────────────────────────────────────────────────
# Health check — used by Streamlit sidebar
# ─────────────────────────────────────────────────────────────────────────────

def check_cloud_health() -> dict[str, bool]:
    """Returns which providers are configured (key present in env)."""
    return {
        "pollinations": True,
        "openrouter": bool(
            os.environ.get("OPENROUTER_API_KEY_1", "").strip()
            or os.environ.get("OPENROUTER_API_KEY_2", "").strip()
        ),
        "groq": bool(os.environ.get("GROQ_API_KEY", "").strip()),
    }
