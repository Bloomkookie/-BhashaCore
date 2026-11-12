"""BhashaCore Extended Chatbot & Model Comparison Runner.

Evaluates both FAQ retrieval and Ollama-based models (qwen2.5:3b / gemma3:4b).
Gracefully logs error status if Ollama is unreachable without crashing.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from config import BASELINE_QUESTIONS_PATH, EXPANDED_QUESTIONS_PATH, OLLAMA_MODEL
from knowledge_base import KnowledgeBase
from ollama_client import check_ollama_status, generate_response

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = BASE_DIR / "model_comparison_results.csv"

# Models actually verified installed on local Ollama:
CANDIDATE_MODELS = ["qwen2.5:3b", "gemma3:4b"]

LANG_KEYS = {
    "English": "question_en",
    "Hindi": "question_hi",
    "Bengali": "question_bn",
    "Punjabi": "question_pa",
    "Assamese": "question_as",
    "Nepali": "question_ne",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BhashaCore model evaluation.")
    parser.add_argument("--models", nargs="+", default=CANDIDATE_MODELS, help="Models to compare.")
    parser.add_argument("--limit", type=int, default=3, help="Number of scenarios to evaluate per model.")
    args = parser.parse_args()

    print("=" * 65)
    print("BHASHACORE - EXTENDED MODEL EVALUATION")
    print("=" * 65)

    kb = KnowledgeBase()
    kb.load_faq_file()
    context = kb.compile_grounding_context()

    status = check_ollama_status()
    print("Ollama Daemon Status:", "AVAILABLE" if status["available"] else "OFFLINE")
    if status["available"]:
        print("Installed Models:", ", ".join(status["models"]))

    with EXPANDED_QUESTIONS_PATH.open("r", encoding="utf-8") as f:
        scenarios = json.load(f)[: args.limit]

    results: List[Dict[str, Any]] = []

    for model_name in args.models:
        print(f"\n--- Testing Model: {model_name} ---")
        for sc in scenarios:
            test_id = sc["id"]
            topic = sc["topic"]
            for lang_name, q_key in LANG_KEYS.items():
                question = sc.get(q_key, "")
                if not question:
                    continue

                if status["available"]:
                    res = generate_response(
                        question=question,
                        context=context,
                        language=lang_name,
                        model=model_name,
                        timeout=60,
                    )
                    answer = res["answer"]
                    latency = res["latency_ms"]
                    run_status = res["status"]
                    err = res.get("error") or ""
                else:
                    answer = ""
                    latency = 0.0
                    run_status = "error"
                    err = "Ollama daemon unreachable (offline test)"

                results.append({
                    "test_id": test_id,
                    "topic": topic,
                    "language": lang_name,
                    "model": model_name,
                    "question": question,
                    "answer": answer,
                    "status": run_status,
                    "latency_ms": latency,
                    "error": err,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                })

    df = pd.DataFrame(results)
    df.to_csv(DEFAULT_OUTPUT, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(df)} comparison records to {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
