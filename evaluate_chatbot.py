
import json
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:3b"

BASE_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = BASE_DIR / "evaluation_questions.json"
OUTPUT_FILE = BASE_DIR / "evaluation_results.csv"

LANGUAGES = {
    "English": "question_en",
    "Hindi": "question_hi",
    "Bengali": "question_bn",
    "Punjabi": "question_pa",
    "Assamese": "question_as",
    "Nepali": "question_ne",
}

# --------------------------------------------------
# FICTIONAL KNOWLEDGE BASE FOR TESTING
# --------------------------------------------------

KNOWLEDGE_BASE = """
This is a fictional scholarship scheme created for testing only.

Required documents:
- Identity document
- Proof of enrollment
- Recent academic record
- Bank account details

The application deadline is unspecified.
The scholarship amount is unspecified.

This is fictional test information, not real government policy.
"""

SYSTEM_PROMPT = """
You are BhashaCore, a multilingual public-service chatbot.

Rules:
1. Answer in the language explicitly requested by the user.
2. Use only the supplied knowledge base.
3. Never invent missing facts, dates, amounts, or requirements.
4. If information is unavailable, clearly say so in the requested language.
5. For questions about required documents, include all four listed documents.
6. Use simple, clear language.
7. The scholarship information is fictional and for testing only.
"""

# --------------------------------------------------
# ASK THE CHATBOT
# --------------------------------------------------

def ask_chatbot(question, language):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        f"Available information:\n{KNOWLEDGE_BASE}\n\n"
                        f"Answer in {language}.\n"
                        f"Question: {question}"
                    ),
                },
            ],
            "options": {
                "temperature": 0,
                "num_predict": 200,
            },
        },
        timeout=300,
    )

    response.raise_for_status()

    data = response.json()
    return data["message"]["content"].strip()


def check_ollama_status() -> bool:
    try:
        res = requests.get("http://localhost:11434/api/tags", timeout=3)
        return res.status_code == 200
    except Exception:
        return False


# --------------------------------------------------
# MAIN EVALUATION
# --------------------------------------------------

def main():
    if not QUESTIONS_FILE.exists():
        print(f"ERROR: Questions file not found: {QUESTIONS_FILE}")
        return

    if not check_ollama_status():
        print("!" * 65)
        print("WARNING: Ollama is not reachable at http://localhost:11434")
        print("To run Ollama-based chatbot evaluation:")
        print("  1. Start Ollama: ollama serve")
        print("  2. Pull model:   ollama pull qwen2.5:3b")
        print("Note: To evaluate the fast FAQ retrieval app instead, run:")
        print("  python3 evaluate_faq.py")
        print("!" * 65)

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        questions = json.load(file)

    results = []

    total_tests = len(questions) * len(LANGUAGES)

    print("=" * 65)
    print("BHASHACORE - MULTILINGUAL CHATBOT EVALUATION")
    print("=" * 65)
    print(f"Model: {MODEL}")
    print(f"Test cases: {len(questions)}")
    print(f"Languages: {len(LANGUAGES)}")
    print(f"Planned responses: {total_tests}")

    for language, question_key in LANGUAGES.items():
        print(f"\n{'=' * 20} {language} {'=' * 20}")

        for item in questions:
            test_id = item.get("id", "unknown")
            topic = item.get("topic", "unknown")
            question = item.get(question_key)

            if not question:
                print(f"Skipping test {test_id}: missing question")

                results.append({
                    "test_id": test_id,
                    "topic": topic,
                    "language": language,
                    "question": "",
                    "answer": "",
                    "expected_facts": json.dumps(
                        item.get("expected_facts", []),
                        ensure_ascii=False,
                    ),
                    "must_not_invent": json.dumps(
                        item.get("must_not_invent", []),
                        ensure_ascii=False,
                    ),
                    "expected_behavior": item.get(
                        "expected_behavior", ""
                    ),
                    "status": "error",
                    "error": "Missing question for language",
                    "timestamp": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                })
                continue

            print(f"\nTest {test_id}: {topic}")
            print(f"Question: {question}")

            try:
                answer = ask_chatbot(question, language)
                status = "success"
                error = ""

                print(f"Answer: {answer}")

            except requests.exceptions.Timeout:
                answer = ""
                status = "error"
                error = "Request timed out after 300 seconds"
                print(f"ERROR: {error}")

            except requests.exceptions.RequestException as exc:
                answer = ""
                status = "error"
                error = str(exc)
                print(f"ERROR: {error}")

            except (KeyError, ValueError) as exc:
                answer = ""
                status = "error"
                error = f"Invalid Ollama response: {exc}"
                print(f"ERROR: {error}")

            except Exception as exc:
                answer = ""
                status = "error"
                error = f"{type(exc).__name__}: {exc}"
                print(f"ERROR: {error}")

            results.append({
                "test_id": test_id,
                "topic": topic,
                "language": language,
                "question": question,
                "answer": answer,
                "expected_facts": json.dumps(
                    item.get("expected_facts", []),
                    ensure_ascii=False,
                ),
                "must_not_invent": json.dumps(
                    item.get("must_not_invent", []),
                    ensure_ascii=False,
                ),
                "expected_behavior": item.get(
                    "expected_behavior", ""
                ),
                "status": status,
                "error": error,
                "timestamp": datetime.now().isoformat(
                    timespec="seconds"
                ),
            })

            # Save after every test so partial results are preserved.
            pd.DataFrame(results).to_csv(
                OUTPUT_FILE,
                index=False,
                encoding="utf-8-sig",
            )

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("EVALUATION RUN FINISHED")
    print("=" * 65)

    df = pd.DataFrame(results)

    successful = int((df["status"] == "success").sum())
    failed = int((df["status"] == "error").sum())

    print(f"Total test cases: {len(df)}")
    print(f"Successful API responses: {successful}")
    print(f"Failed API responses: {failed}")

    if len(df) > 0:
        print("\nAPI execution status by language:")
        summary = (
            df.groupby(["language", "status"])
            .size()
            .unstack(fill_value=0)
        )
        print(summary.to_string())

        print("\nErrors:")
        errors = df[df["status"] == "error"]

        if errors.empty:
            print("No API execution errors.")
        else:
            print(
                errors[["language", "topic", "error"]]
                .to_string(index=False)
            )

    print(f"\nResults saved to: {OUTPUT_FILE}")
    print(
        "\nNote: API success means a response was received. "
        "It does not mean the answer is correct or fluent."
    )


if __name__ == "__main__":
    main()
