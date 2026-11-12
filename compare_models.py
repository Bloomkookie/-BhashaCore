import json
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = BASE_DIR / "evaluation_questions_expanded.json"
OUTPUT_FILE = BASE_DIR / "model_comparison_results.csv"

OLLAMA_URL = "http://localhost:11434/api/chat"
# Baseline is already installed. Challenger is a stronger candidate; pull it first.
MODELS = ["qwen2.5:3b", "qwen2.5:7b"]

LANGUAGES = {
    "English": "question_en",
    "Hindi": "question_hi",
    "Bengali": "question_bn",
    "Punjabi": "question_pa",
    "Assamese": "question_as",
    "Nepali": "question_ne",
}

KNOWLEDGE_BASE = """
This is a fictional scholarship scheme created for testing only.
Required documents:
- Identity document
- Proof of enrollment
- Recent academic record
- Bank account details
The application deadline is unspecified.
The scholarship amount is unspecified.
Eligibility criteria are not provided.
The application method is not specified.
Contact information is not provided.
This is fictional test information, not real government policy.
"""

SYSTEM_PROMPT = """
You are BhashaCore, a multilingual public-service chatbot.
Follow these rules:
1. Answer only in the language requested by the user.
2. Use only the supplied knowledge base.
3. Never invent dates, amounts, eligibility rules, links, contact details, or requirements.
4. When information is missing, clearly say it is not specified.
5. If asked for required documents, include all four documents from the knowledge base.
6. Keep answers concise and direct.
7. This scholarship scheme is fictional and only for testing.
"""

def ask_chatbot(question, language, model):
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Available information:\n{KNOWLEDGE_BASE}\n\n"
                    f"Answer in {language}.\nQuestion: {question}"
                ),
            },
        ],
        "options": {"temperature": 0, "num_predict": 180},
    }
    started = time.perf_counter()
    response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    answer = response.json()["message"]["content"].strip()
    return answer, round(elapsed, 2)

def check_ollama_status() -> bool:
    try:
        res = requests.get("http://localhost:11434/api/tags", timeout=3)
        return res.status_code == 200
    except Exception:
        return False

def main():
    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(f"Missing benchmark: {QUESTIONS_FILE}")

    if not check_ollama_status():
        print("!" * 65)
        print("WARNING: Ollama is not reachable at http://localhost:11434")
        print("To run LLM model comparisons:")
        print("  1. Start Ollama: ollama serve")
        print("  2. Pull models:  ollama pull qwen2.5:3b && ollama pull qwen2.5:7b")
        print("Proceeding anyway (API errors will be logged to CSV)...")
        print("!" * 65)

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    rows = []
    total = len(questions) * len(LANGUAGES) * len(MODELS)
    print(f"Cases: {len(questions)} | Languages: {len(LANGUAGES)} | Models: {len(MODELS)}")
    print(f"Planned model responses: {total}")
    print("This can take a while, especially with qwen2.5:7b.")

    for model in MODELS:
        print(f"\n{'=' * 20} MODEL: {model} {'=' * 20}")
        for language, question_key in LANGUAGES.items():
            print(f"\n--- {language} ---")
            for item in questions:
                question = item.get(question_key)
                row = {
                    "model": model,
                    "test_id": item["id"],
                    "topic": item["topic"],
                    "language": language,
                    "question": question or "",
                    "answer": "",
                    "expected_facts": json.dumps(item.get("expected_facts", []), ensure_ascii=False),
                    "must_not_invent": json.dumps(item.get("must_not_invent", []), ensure_ascii=False),
                    "expected_behavior": item.get("expected_behavior", ""),
                    "status": "error",
                    "latency_seconds": None,
                    "error": "",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    # Fill these after human/native-speaker review:
                    "factual_correctness": "",
                    "completeness": "",
                    "language_quality": "",
                    "hallucination": "",
                    "overall_pass": "",
                    "review_notes": "",
                }
                if not question:
                    row["error"] = "Missing translated question"
                else:
                    try:
                        answer, latency = ask_chatbot(question, language, model)
                        row["answer"] = answer
                        row["status"] = "success"
                        row["latency_seconds"] = latency
                        print(f"{item['id']} | {latency}s | {answer[:140]}")
                    except Exception as exc:
                        row["error"] = f"{type(exc).__name__}: {exc}"
                        print(f"{item['id']} | ERROR | {row['error']}")

                rows.append(row)
                pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    df = pd.DataFrame(rows)
    print("\nSaved:", OUTPUT_FILE)
    print("\nAPI status by model and language:")
    print(df.pivot_table(index=["model", "language"], columns="status",
                         values="test_id", aggfunc="count", fill_value=0).to_string())
    print("\nMean latency (successful responses only):")
    print(df[df["status"] == "success"].groupby(["model", "language"])["latency_seconds"]
          .mean().round(2).to_string())
    print("\nIMPORTANT: API success and latency are not answer-quality scores.")
    print("Review answers with fluent speakers and fill the quality columns before reporting quality.")

if __name__ == "__main__":
    main()
