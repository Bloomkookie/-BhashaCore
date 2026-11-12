"""BhashaCore: Human Quality Review Template Generator.

Reads model comparison or evaluation results and prepares a human review template
with standardized rubrics for native/fluent speakers to evaluate:
- Factual correctness
- Completeness
- Language fluency / grammar
- Hallucination / invention of ungrounded facts
- Overall pass / fail

Works with either `model_comparison_results.csv` (from compare_models.py)
or `evaluation_results.csv` / `faq_evaluation_results.csv`.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

# Check potential input files in priority order
CANDIDATE_INPUTS = [
    BASE_DIR / "model_comparison_results.csv",
    BASE_DIR / "evaluation_results.csv",
    BASE_DIR / "faq_evaluation_results.csv",
]
OUTPUT_FILE = BASE_DIR / "quality_review_template.csv"


def main():
    input_file = None
    for cand in CANDIDATE_INPUTS:
        if cand.exists():
            input_file = cand
            break

    if not input_file:
        print("=" * 65)
        print("BhashaCore - Quality Review Template Generator")
        print("=" * 65)
        print("ERROR: No evaluation result CSV found.")
        print("Please run one of the evaluation scripts first:")
        print("  - python3 evaluate_faq.py          (Generates faq_evaluation_results.csv)")
        print("  - python3 evaluate_chatbot.py      (Generates evaluation_results.csv via Ollama)")
        print("  - python3 compare_models.py        (Generates model_comparison_results.csv via Ollama)")
        sys.exit(1)

    print(f"Reading results from: {input_file.name}")
    df = pd.read_csv(input_file)

    # Determine available columns
    core_cols = ["test_id", "language", "question", "answer"]
    optional_cols = [
        "model", "topic", "expected_facts", "must_not_invent",
        "expected_behavior", "status", "latency_seconds", "latency_ms"
    ]

    selected_cols = [c for c in (core_cols + optional_cols) if c in df.columns]
    review = df[selected_cols].copy()

    # Add review rubrics
    review["factual_correctness"] = ""
    review["completeness"] = ""
    review["language_quality"] = ""
    review["hallucination"] = ""
    review["overall_pass"] = ""
    review["review_notes"] = ""

    review.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\nQuality review template successfully created: {OUTPUT_FILE.name}")
    print(f"Total answers prepared for human evaluation: {len(review)}")
    print("\nSuggested evaluation rubrics for native/fluent reviewers:")
    print("  factual_correctness: pass / fail / uncertain")
    print("  completeness:        pass / fail / uncertain")
    print("  language_quality:    good / acceptable / poor")
    print("  hallucination:       yes / no / uncertain")
    print("  overall_pass:        yes / no")
    print("\nIMPORTANT: Review answers in their respective target languages.")
    print("Translations are unverified drafts until fluent native speaker sign-off.")


if __name__ == "__main__":
    main()
