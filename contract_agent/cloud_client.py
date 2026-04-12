"""
Unified Cloud LLM Client with Fallback Logic.
Primary: OpenRouter (gpt-oss-120b)
Secondary: Groq (llama-3.3-70b-versatile)

This ensures the system stays online even if the primary provider fails.
"""
from __future__ import annotations

import os
import logging
from typing import Any

from openai import OpenAI
from contract_agent._shared_prompt import (
    build_system_prompt,
    build_user_message,
    safe_parse_analysis,
)

logger = logging.getLogger(__name__)

def analyze_clause_with_cloud(
    clause_text: str,
    risk_label: str,
    confidence: float,
    best_practices: list[dict[str, Any]],
) -> dict[str, str]:
    """
    Attempts analysis using OpenRouter. If it fails, falls back to Groq.
    Returns the standard 8-key analysis dictionary.
    """
    
    # ── Attempt 1: OpenRouter ──────────────────────────────────────────────
    or_api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    or_model = os.environ.get("OPENROUTER_MODEL", "openai/gpt-oss-120b").strip()
    
    if or_api_key:
        try:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=or_api_key,
            )
            
            completion = client.chat.completions.create(
                model=or_model,
                messages=[
                    {"role": "system", "content": build_system_prompt(risk_label, confidence)},
                    {"role": "user",   "content": build_user_message(clause_text, best_practices)},
                ],
                temperature=0.15,
                max_tokens=1800,
                response_format={"type": "json_object"},
                timeout=60.0, # 1 minute timeout for primary
            )
            
            raw = completion.choices[0].message.content or "{}"
            return safe_parse_analysis(raw, fallback_text=clause_text[:200])
            
        except Exception as exc:
            logger.warning(f"OpenRouter failed: {exc}. Falling back to Groq...")
    
    # ── Attempt 2: Groq Fallback ───────────────────────────────────────────
    groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()
    groq_model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
    
    if groq_api_key:
        try:
            client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=groq_api_key,
            )
            
            completion = client.chat.completions.create(
                model=groq_model,
                messages=[
                    {"role": "system", "content": build_system_prompt(risk_label, confidence)},
                    {"role": "user",   "content": build_user_message(clause_text, best_practices)},
                ],
                temperature=0.15,
                max_tokens=1500, # Groq usually has slightly smaller context windows for JSON mode
                response_format={"type": "json_object"},
                timeout=30.0, # 30s timeout for fallback
            )
            
            raw = completion.choices[0].message.content or "{}"
            parsed = safe_parse_analysis(raw, fallback_text=clause_text[:200])
            # Add a small note to the summary indicating fallback was used
            if "plain_english_summary" in parsed:
                parsed["plain_english_summary"] += " [Analysis provided by Fallback Provider (Groq)]"
            return parsed
            
        except Exception as exc:
            logger.error(f"Groq fallback failed: {exc}")
    
    # ── Final Fallback: Error string in analysis shape ──────────────────────
    error_msg = "Cloud analysis failed across all providers."
    if not or_api_key and not groq_api_key:
        error_msg = "No Cloud API keys (OpenRouter/Groq) found in .env."
        
    return safe_parse_analysis("{}", fallback_text=error_msg)


def check_cloud_health() -> dict[str, bool]:
    """Simple check to see if keys are configured."""
    return {
        "openrouter": bool(os.environ.get("OPENROUTER_API_KEY", "").strip()),
        "groq": bool(os.environ.get("GROQ_API_KEY", "").strip()),
    }
