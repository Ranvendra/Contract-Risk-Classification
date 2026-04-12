"""Quick end-to-end test for both offline (Ollama) and online (OpenRouter) modes."""
import os, json, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from contract_agent.workflow import run_hybrid_agent_pipeline

SAMPLE_CONTRACT = """
CONSULTING SERVICES AGREEMENT

1. Termination
Either party may terminate this Agreement immediately at any time, with or without cause,
with no notice required. Upon termination, Client forfeits all rights to deliverables
completed up to that point and no refund shall be issued.

2. Indemnification
The Consultant shall not be liable for any damages, losses, or claims arising from the
services provided. The Client agrees to indemnify and hold harmless the Consultant from
any and all claims, including attorney fees and legal costs.
"""

def run_test(mode: str):
    print(f"\n{'='*70}")
    print(f"  TESTING MODE: {mode.upper()}")
    print(f"{'='*70}\n")

    result = run_hybrid_agent_pipeline(
        SAMPLE_CONTRACT,
        confidence_threshold=0.5,
        mode=mode,
    )

    if result.get("error"):
        print(f"ERROR: {result['error']}")
        return

    print(result["markdown_report"])

    clauses = result["structured_report"].get("flagged_clauses_and_mitigation", [])
    print(f"\n{'='*70}")
    print(f"  STRUCTURED JSON — Clause 1 of {len(clauses)}")
    print(f"{'='*70}")
    if clauses:
        print(json.dumps(clauses[0], indent=2))

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "offline"
    run_test(mode)
