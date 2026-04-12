import os
import re
from typing import Any

from openai import OpenAI

from contract_agent._shared_prompt import (
    build_system_prompt,
    build_user_message,
    safe_parse_analysis,
)


def analyze_clause_with_openrouter(
    clause_text: str,
    risk_label: str,
    confidence: float,
    best_practices: list[dict[str, Any]],
) -> dict[str, str]:
    """
    Calls OpenRouter cloud API and returns a structured 8-key analysis dict.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    model = os.environ.get("OPENROUTER_MODEL", "openai/gpt-oss-120b").strip()

    system = build_system_prompt(risk_label, confidence)
    user   = build_user_message(clause_text, best_practices)

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
        temperature=0.15,          # lower = more consistent / less hallucination
        max_tokens=1800,
        response_format={"type": "json_object"},
    )

    raw = completion.choices[0].message.content or "{}"
    return safe_parse_analysis(raw, fallback_text=clause_text[:200])
