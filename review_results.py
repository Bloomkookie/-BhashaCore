
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "evaluation_results.csv"
OUTPUT_FILE = BASE_DIR / "quality_review.csv"

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Missing file: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

# Keep the fields needed to review each answer.
review = df[
    [
        "test_id",
        "topic",
        "language",
        "question",
        "answer",
        "expected_facts",
        "must_not_invent",
        "expected_behavior",
        "status",
    ]
].copy()

# Human-review fields. Fill these in after checking each answer.
review["factual_correctness"] = ""
review["completeness"] = ""
review["language_quality"] = ""
review["hallucination"] = ""
review["overall_pass"] = ""
review["review_notes"] = ""

review.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
)

print(f"Review template created: {OUTPUT_FILE}")
print(f"Answers to review: {len(review)}")
print("\nUse these labels:")
print("factual_correctness: pass / fail / uncertain")
print("completeness: pass / fail / uncertain")
print("language_quality: good / acceptable / poor")
print("hallucination: yes / no / uncertain")
print("overall_pass: yes / no")
print("\nReview the answers in the requested language.")
