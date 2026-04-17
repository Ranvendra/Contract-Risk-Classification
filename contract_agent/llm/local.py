"""
Offline LLM backend via local Ollama.
Same interface as cloud_client — returns the 8-key analysis dict.
Auto-starts the Ollama daemon if it is not already running.
"""
from __future__ import annotations

import os
import subprocess
import time
from typing import Any

import requests

from contract_agent.llm.prompting import (
    build_system_prompt,
    build_user_message,
    safe_parse_analysis,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_base_url() -> str:
    return os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


def _get_model() -> str:
    return os.environ.get("OLLAMA_MODEL", "qwen3.5:2b").strip()


def _is_ollama_running(base_url: str, timeout: float = 2.0) -> bool:
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _start_ollama_daemon(base_url: str) -> bool:
    """Attempt to start Ollama in the background; return True if alive."""
    if _is_ollama_running(base_url):
        return True

    ollama_bin = "/usr/local/bin/ollama"
    if not os.path.exists(ollama_bin):
        ollama_bin = "ollama"

    try:
        subprocess.Popen(
            [ollama_bin, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        for _ in range(20):
            time.sleep(0.5)
            if _is_ollama_running(base_url):
                return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Health check (used by Streamlit sidebar)
# ---------------------------------------------------------------------------

def check_ollama_health() -> dict[str, Any]:
    base_url = _get_base_url()
    model = _get_model()
    auto_started = False

    if not _is_ollama_running(base_url):
        auto_started = _start_ollama_daemon(base_url)

    if not _is_ollama_running(base_url):
        return {
            "reachable": False,
            "model": model,
            "available_models": [],
            "auto_started": False,
        }

    available: list[str] = []
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=5)
        if r.status_code == 200:
            available = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass

    return {
        "reachable": True,
        "model": model,
        "available_models": available,
        "auto_started": auto_started,
    }


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyze_clause_with_ollama(
    clause_text: str,
    risk_label: str,
    confidence: float,
    best_practices: list[dict[str, Any]],
    domain: str = "General",
) -> dict[str, str]:
    """
    Sends the clause to local Ollama for analysis.
    Args:
        domain: Detected contract domain for domain-aware prompting.
    Returns the 8-key analysis dict (same as cloud client).
    """
    base_url = _get_base_url()
    model = _get_model()

    if not _is_ollama_running(base_url):
        started = _start_ollama_daemon(base_url)
        if not started:
            raise RuntimeError(
                f"Ollama is not reachable at {base_url}. "
                "Start it with: ollama serve"
            )

    system = build_system_prompt(risk_label, confidence, domain=domain)
    user   = build_user_message(clause_text, best_practices, domain=domain)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False,
        "options": {
            "temperature": 0.15,
            "num_predict": 2048,
            "top_p": 0.9,
        },
    }

    try:
        response = requests.post(
            f"{base_url}/api/chat",
            json=payload,
            timeout=180,           # local models can be slow on first load
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Ollama request timed out (3 min). "
            f"Is '{model}' pulled? Run: ollama pull {model}"
        )
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(f"Cannot connect to Ollama at {base_url}: {exc}")

    data = response.json()
    raw  = data.get("message", {}).get("content", "") or "{}"
    return safe_parse_analysis(raw, fallback_text=clause_text[:200])
