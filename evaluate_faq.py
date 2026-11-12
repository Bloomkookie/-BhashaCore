"""BhashaCore Automated FAQ Evaluation Script.

Evaluates the FAQ retrieval engine across all 6 supported languages
(English, Hindi, Bengali, Punjabi, Assamese, and Nepali) against both:
- `evaluation_questions_expanded.json` (10 scenarios x 6 languages = 60 questions)
- `evaluation_questions.json` (3 scenarios x 6 languages = 18 questions)
- An adversarial suite of unsupported / out-of-domain questions to verify
  zero-hallucination and rejection behavior.

Outputs latency, per-language accuracy metrics, and exports a detailed CSV report.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from faq_engine import (
    DEFAULT_FAQ_PATH,
    LANGUAGE_LABELS,
    find_faq,
    get_answer,
    load_faqs,
)

BASE_DIR = Path(__file__).resolve().parent
EXPANDED_QUESTIONS_FILE = BASE_DIR / "evaluation_questions_expanded.json"
BASELINE_QUESTIONS_FILE = BASE_DIR / "evaluation_questions.json"
OUTPUT_CSV_FILE = BASE_DIR / "faq_evaluation_results.csv"

# Out-of-domain / unsupported questions to test anti-hallucination gating
UNSUPPORTED_TEST_CASES = [
    {"id": "ood_01", "lang": "en", "q": "What is the capital of France?"},
    {"id": "ood_02", "lang": "en", "q": "Who is the President of the United States?"},
    {"id": "ood_03", "lang": "hi", "q": "भारत का प्रधानमंत्री कौन है?"},
    {"id": "ood_04", "lang": "bn", "q": "আজকের আবহাওয়া কেমন?"},
    {"id": "ood_05", "lang": "pa", "q": "ਮੈਨੂੰ ਖਾਣਾ ਬਣਾਉਣਾ ਸਿਖਾਓ"},
    {"id": "ood_06", "lang": "as", "q": "ব্ৰহ্মপুত্ৰ নদী ক’ৰ পৰা ওলাইছে?"},
    {"id": "ood_07", "lang": "ne", "q": "नेपालको राजधानी कहाँ छ?"},
    {"id": "ood_08", "lang": "en", "q": "Is hostel accommodation provided with the scholarship?"},
    {"id": "ood_09", "lang": "en", "q": "Can I buy a laptop with this scholarship money?"},
    {"id": "ood_10", "lang": "hi", "q": "क्या इस छात्रवृत्ति के साथ हॉस्टल मिलेगा?"},
]


def run_evaluation(export_csv: bool = True) -> Dict[str, Any]:
    print("=" * 70)
    print("BHASHACORE - MULTILINGUAL FAQ RETRIEVAL EVALUATION")
    print("=" * 70)

    payload = load_faqs(DEFAULT_FAQ_PATH)
    print(f"Loaded FAQ Knowledge Base: {len(payload['faqs'])} topics")

    with EXPANDED_QUESTIONS_FILE.open("r", encoding="utf-8") as f:
        expanded_scenarios = json.load(f)

    rows: List[Dict[str, Any]] = []

    # Per-language stats: {lang_code: {"total": 0, "correct": 0, "total_time": 0.0}}
    stats: Dict[str, Dict[str, Any]] = {
        code: {"name": name, "total": 0, "correct": 0, "total_time": 0.0}
        for name, code in LANGUAGE_LABELS.items()
    }

    # 1. Expanded benchmark evaluation
    print(f"\n[1/3] Running Expanded Benchmark ({len(expanded_scenarios)} scenarios x 6 languages)...")
    for item in expanded_scenarios:
        expected_topic = item["topic"]
        test_id = item["id"]

        for lang_name, code in LANGUAGE_LABELS.items():
            question = item.get(f"question_{code}")
            if not question:
                continue

            t0 = time.perf_counter()
            faq, score = find_faq(question, payload)
            elapsed = time.perf_counter() - t0

            matched_topic = faq["topic"] if faq else "NO_MATCH"
            is_correct = (matched_topic == expected_topic)

            stats[code]["total"] += 1
            if is_correct:
                stats[code]["correct"] += 1
            stats[code]["total_time"] += elapsed

            answer_text = get_answer(faq, code) if faq else ""

            rows.append({
                "benchmark": "expanded",
                "test_id": test_id,
                "language": lang_name,
                "lang_code": code,
                "expected_topic": expected_topic,
                "matched_topic": matched_topic,
                "score": round(score, 3),
                "is_correct": is_correct,
                "latency_ms": round(elapsed * 1000, 2),
                "question": question,
                "answer": answer_text,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            })

    # 2. Baseline benchmark evaluation
    print("\n[2/3] Running Baseline Benchmark...")
    with BASELINE_QUESTIONS_FILE.open("r", encoding="utf-8") as f:
        baseline_scenarios = json.load(f)

    base_correct = 0
    base_total = 0
    for item in baseline_scenarios:
        expected_topic = item["topic"]
        test_id = f"base_{item['id']}"

        for lang_name, code in LANGUAGE_LABELS.items():
            question = item.get(f"question_{code}")
            if not question:
                continue

            t0 = time.perf_counter()
            faq, score = find_faq(question, payload)
            elapsed = time.perf_counter() - t0

            matched_topic = faq["topic"] if faq else "NO_MATCH"
            is_correct = (matched_topic == expected_topic)
            base_total += 1
            if is_correct:
                base_correct += 1

            rows.append({
                "benchmark": "baseline",
                "test_id": test_id,
                "language": lang_name,
                "lang_code": code,
                "expected_topic": expected_topic,
                "matched_topic": matched_topic,
                "score": round(score, 3),
                "is_correct": is_correct,
                "latency_ms": round(elapsed * 1000, 2),
                "question": question,
                "answer": get_answer(faq, code) if faq else "",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            })

    # 3. Anti-hallucination / unsupported questions
    print("\n[3/3] Running Anti-Hallucination Gating Checks...")
    ood_correct = 0
    ood_total = len(UNSUPPORTED_TEST_CASES)
    for ood in UNSUPPORTED_TEST_CASES:
        t0 = time.perf_counter()
        faq, score = find_faq(ood["q"], payload)
        elapsed = time.perf_counter() - t0

        matched_topic = faq["topic"] if faq else "NO_MATCH"
        # For out-of-domain questions, the correct behavior is NO_MATCH
        is_correct = (faq is None)
        if is_correct:
            ood_correct += 1

        rows.append({
            "benchmark": "anti_hallucination",
            "test_id": ood["id"],
            "language": ood["lang"],
            "lang_code": ood["lang"],
            "expected_topic": "NO_MATCH",
            "matched_topic": matched_topic,
            "score": round(score, 3),
            "is_correct": is_correct,
            "latency_ms": round(elapsed * 1000, 2),
            "question": ood["q"],
            "answer": "",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })

    # Print summary metrics table
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY RESULTS")
    print("=" * 70)
    print(f"{'Language':<12} {'Code':<6} {'Tested':<8} {'Passed':<8} {'Accuracy':<10} {'Mean Latency':<12}")
    print("-" * 70)

    total_tested = 0
    total_passed = 0
    total_time = 0.0

    for code, data in stats.items():
        total = data["total"]
        correct = data["correct"]
        acc = (correct / total * 100) if total else 0.0
        mean_ms = (data["total_time"] / total * 1000) if total else 0.0

        total_tested += total
        total_passed += correct
        total_time += data["total_time"]

        print(f"{data['name']:<12} {code:<6} {total:<8} {correct:<8} {acc:>5.1f}%     {mean_ms:>6.2f} ms")

    overall_acc = (total_passed / total_tested * 100) if total_tested else 0.0
    overall_mean_ms = (total_time / total_tested * 1000) if total_tested else 0.0
    print("-" * 70)
    print(f"{'OVERALL':<12} {'all':<6} {total_tested:<8} {total_passed:<8} {overall_acc:>5.1f}%     {overall_mean_ms:>6.2f} ms")
    print("=" * 70)
    print(f"Baseline Benchmark Pass Rate:       {base_correct}/{base_total} ({base_correct/base_total*100:.1f}%)")
    print(f"Anti-Hallucination Rejection Rate: {ood_correct}/{ood_total} ({ood_correct/ood_total*100:.1f}%)")

    if export_csv:
        import csv
        fieldnames = [
            "benchmark", "test_id", "language", "lang_code", "expected_topic",
            "matched_topic", "score", "is_correct", "latency_ms", "question",
            "answer", "timestamp",
        ]
        with OUTPUT_CSV_FILE.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nDetailed evaluation report exported to: {OUTPUT_CSV_FILE}")

    return {
        "total_tested": total_tested,
        "total_passed": total_passed,
        "overall_accuracy": overall_acc,
        "baseline_passed": base_correct,
        "baseline_total": base_total,
        "ood_rejected": ood_correct,
        "ood_total": ood_total,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BhashaCore FAQ retrieval.")
    parser.add_argument("--no-csv", action="store_true", help="Do not write results to CSV.")
    args = parser.parse_args()
    run_evaluation(export_csv=not args.no_csv)
