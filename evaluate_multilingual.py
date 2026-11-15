"""Six-language behavioural evaluation for BhashaCore.

Default (no ``--live``) validates the case file and prints the coverage matrix.
With ``--live`` it runs the real orchestrator (Ollama + optional NLLB), checks
routing/provenance/script expectations, and writes every answer for human review.

These checks are proxies, not proof of factual correctness or translation
quality.  Read the saved answers before making per-language quality claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import resource
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from chatbot import Chatbot  # noqa: E402
from language_utils import LANGUAGE_NAMES, language_code, script_purity  # noqa: E402

DEFAULT_CASES = BASE_DIR / "evaluation" / "multilingual_cases.json"
DEFAULT_OUTPUT = BASE_DIR / "evaluation" / "results"
SUPPORTED_ROUTES = {"general", "document", "faq", "translation", "summarize", "clarify"}
SUPPORTED_LANGUAGES = set(LANGUAGE_NAMES)


def load_cases(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    if not cases:
        raise ValueError(f"No cases found in {path}")
    seen = set()
    for case in cases:
        case_id = case.get("id")
        if not case_id or case_id in seen:
            raise ValueError(f"Missing or duplicate case id: {case_id!r}")
        seen.add(case_id)
        if case.get("language") not in SUPPORTED_LANGUAGES:
            raise ValueError(f"{case_id}: unsupported language {case.get('language')!r}")
        expect = case.get("expect") or {}
        if expect.get("route") and expect["route"] not in SUPPORTED_ROUTES:
            raise ValueError(f"{case_id}: unknown route {expect['route']!r}")
        turns = case.get("turns") or [case.get("prompt")]
        if not turns or not all(isinstance(turn, str) and turn.strip() for turn in turns):
            raise ValueError(f"{case_id}: empty prompt")
    return payload


def coverage_matrix(cases: List[Dict[str, Any]]) -> str:
    languages = [code for code in LANGUAGE_NAMES]
    categories = sorted({case["category"] for case in cases})
    counts = Counter((case["category"], case["language"]) for case in cases)
    width = max(len(category) for category in categories) + 2
    header = "category".ljust(width) + "".join(lang.ljust(6) for lang in languages)
    lines = [header, "-" * len(header)]
    for category in categories:
        row = category.ljust(width)
        row += "".join(str(counts.get((category, lang), 0)).ljust(6) for lang in languages)
        lines.append(row)
    lines.append("")
    lines.append(f"{len(cases)} cases, {len(categories)} categories, {len(languages)} languages")
    for case in cases:
        lines.append(
            f"  {case['id']:<20} {case['category']:<18} {case['language']}  "
            f"{(case.get('turns') or [case['prompt']])[-1][:60]}"
        )
    return "\n".join(lines)


def _normalize_digits(text: str) -> str:
    return "".join(
        str(unicodedata.decimal(char)) if unicodedata.decimal(char, None) is not None else char
        for char in text
    )


def _contains(answer: str, candidates: Iterable[str]) -> bool:
    lowered = _normalize_digits(answer).casefold()
    return any(_normalize_digits(candidate).casefold() in lowered for candidate in candidates)


def evaluate_expectations(
    case: Dict[str, Any], response: Any, target_lang: str
) -> Dict[str, Any]:
    expect = case.get("expect") or {}
    checks: Dict[str, bool] = {}
    details: Dict[str, Any] = {}

    if expect.get("route"):
        checks["route"] = response.route == expect["route"]
        details["route"] = response.route
    if "evidence_found" in expect:
        checks["evidence_found"] = response.evidence_found == expect["evidence_found"]
        details["evidence_found"] = response.evidence_found
    if "min_citations" in expect:
        count = len(response.citations)
        checks["citations"] = count >= int(expect["min_citations"])
        details["citation_count"] = count
    if expect.get("contains_any"):
        checks["keywords"] = _contains(response.answer, expect["contains_any"])
    if expect.get("forbid_any"):
        checks["forbidden"] = not _contains(response.answer, expect["forbid_any"])
    if expect.get("min_script_purity") is not None:
        purity = script_purity(response.answer, target_lang)
        checks["script_purity"] = purity >= float(expect["min_script_purity"])
        details["script_purity"] = round(purity, 3)
    checks["non_empty"] = bool(response.answer.strip())
    return {"passed": all(checks.values()), "checks": checks, "details": details}


def run_case(bot: Chatbot, case: Dict[str, Any]) -> Dict[str, Any]:
    turns = case.get("turns") or [case["prompt"]]
    started = time.perf_counter()
    response = None
    error = None
    bot.clear_history()
    for turn in turns:
        try:
            response = bot.respond(turn, selected_language="Auto")
        except Exception as exc:  # noqa: BLE001 - record and continue the suite
            error = f"{type(exc).__name__}: {exc}"
            break
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)

    if response is None:
        return {
            "id": case["id"],
            "category": case["category"],
            "language": case["language"],
            "turns": turns,
            "passed": False,
            "error": error or "no response",
            "latency_ms": elapsed_ms,
        }

    evaluation = evaluate_expectations(case, response, response.target_lang)
    record = {
        "id": case["id"],
        "category": case["category"],
        "language": case["language"],
        "turns": turns,
        "answer": response.answer,
        "route": response.route,
        "source_type": response.source_type,
        "target_lang": response.target_lang,
        "detected_lang": response.detected_lang,
        "detection_confidence": response.detection_confidence,
        "evidence_found": response.evidence_found,
        "citation_count": len(response.citations),
        "citations": [citation.label() for citation in response.citations],
        "model": response.model,
        "backend": response.backend,
        "translated": response.translated,
        "used_translation_assist": response.used_translation_assist,
        "script_purity": round(script_purity(response.answer, response.target_lang), 3),
        "latency_ms": response.latency_ms,
        "wall_latency_ms": elapsed_ms,
        "notes": response.notes,
        "error": response.error,
        "passed": evaluation["passed"],
        "checks": evaluation["checks"],
        "check_details": evaluation["details"],
    }
    return record


def peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return round(value / divisor, 1)


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_language: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"cases": 0, "passed": 0, "latency_ms": [], "purity": []}
    )
    by_category: Dict[str, Dict[str, int]] = defaultdict(lambda: {"cases": 0, "passed": 0})
    for record in records:
        lang = record["language"]
        by_language[lang]["cases"] += 1
        by_language[lang]["passed"] += int(bool(record.get("passed")))
        if record.get("latency_ms") is not None:
            by_language[lang]["latency_ms"].append(record["latency_ms"])
        if record.get("script_purity") is not None:
            by_language[lang]["purity"].append(record["script_purity"])
        category = record["category"]
        by_category[category]["cases"] += 1
        by_category[category]["passed"] += int(bool(record.get("passed")))

    language_summary = {}
    for lang, row in by_language.items():
        latencies = row["latency_ms"]
        purities = row["purity"]
        language_summary[lang] = {
            "cases": row["cases"],
            "passed": row["passed"],
            "mean_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
            "mean_script_purity": round(sum(purities) / len(purities), 3) if purities else None,
        }
    return {
        "cases": len(records),
        "passed": sum(1 for record in records if record.get("passed")),
        "by_language": language_summary,
        "by_category": dict(by_category),
        "peak_rss_mb": peak_rss_mb(),
    }


def print_summary(summary: Dict[str, Any], records: List[Dict[str, Any]]) -> None:
    print("\n== Summary ==")
    print(f"Cases: {summary['cases']}  Passed: {summary['passed']}  Peak RSS: {summary['peak_rss_mb']} MB")
    print(f"{'lang':<6}{'cases':<7}{'passed':<8}{'mean s':<9}{'purity':<8}")
    for lang in LANGUAGE_NAMES:
        row = summary["by_language"].get(lang)
        if not row:
            continue
        mean_s = (row["mean_latency_ms"] or 0) / 1000.0
        purity = row["mean_script_purity"]
        print(
            f"{lang:<6}{row['cases']:<7}{row['passed']:<8}{mean_s:<9.1f}"
            f"{(purity if purity is not None else 0):<8.3f}"
        )
    failures = [record for record in records if not record.get("passed")]
    if failures:
        print("\nFailed checks:")
        for record in failures:
            failed = [name for name, ok in (record.get("checks") or {}).items() if not ok]
            print(f"  {record['id']}: {failed or record.get('error')}")


def save_records(records: List[Dict[str, Any]], prefix: Path, model: str) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in model)
    jsonl_path = prefix.with_name(f"{prefix.name}_{slug}.jsonl")
    csv_path = prefix.with_name(f"{prefix.name}_{slug}.csv")
    summary_path = prefix.with_name(f"{prefix.name}_{slug}_summary.json")

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    columns = [
        "id", "category", "language", "passed", "route", "source_type", "evidence_found",
        "citation_count", "script_purity", "latency_ms", "wall_latency_ms", "model",
        "backend", "translated", "answer", "notes", "error",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["notes"] = " | ".join(record.get("notes") or [])
            writer.writerow(row)

    summary = summarize(records)
    summary["model"] = model
    summary["prompt_review"] = "draft_unverified"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {jsonl_path}\nSaved: {csv_path}\nSaved: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--live", action="store_true", help="run the real chatbot")
    parser.add_argument("--model", default=None, help="override the Ollama chat model")
    parser.add_argument("--language", default=None, help="comma-separated language codes")
    parser.add_argument("--category", default=None, help="comma-separated categories")
    parser.add_argument("--per-category", type=int, default=None, help="limit cases per category")
    parser.add_argument("--strict", action="store_true", help="exit non-zero if any check fails")
    return parser.parse_args()


def filter_cases(cases: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    selected = cases
    if args.language:
        wanted = {item.strip() for item in args.language.split(",") if item.strip()}
        selected = [case for case in selected if case["language"] in wanted]
    if args.category:
        wanted = {item.strip() for item in args.category.split(",") if item.strip()}
        selected = [case for case in selected if case["category"] in wanted]
    if args.per_category:
        counts: Counter = Counter()
        limited = []
        for case in selected:
            if counts[case["category"]] >= args.per_category:
                continue
            counts[case["category"]] += 1
            limited.append(case)
        selected = limited
    return selected


def main() -> int:
    args = parse_args()
    payload = load_cases(args.cases)
    cases = filter_cases(payload["cases"], args)
    if not cases:
        print("No cases matched the filters.")
        return 1

    print(coverage_matrix(cases))
    if not args.live:
        print("\nDry run only. Add --live to call Ollama/NLLB. This file's translations are draft and unreviewed.")
        return 0

    bot = Chatbot(model=args.model)
    fixture = payload.get("metadata", {}).get("fixtures", {}).get("library")
    if fixture:
        fixture_path = BASE_DIR / "evaluation" / fixture
        result = bot.add_document(fixture_path.name, fixture_path.read_bytes())
        if result.error:
            print(f"Fixture ingestion failed: {result.error}")
            return 1
        print(f"Fixture loaded: {fixture_path.name} ({len(result.chunks)} chunk(s))")
    status = bot.refresh_status()
    print(
        f"Model: {status.get('active_model')} | embedding: {status.get('embedding_model')} "
        f"({status.get('embedding_available')}) | translator: {status.get('translator_backend')}"
    )

    records: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['id']} ...", end=" ", flush=True)
        record = run_case(bot, case)
        records.append(record)
        flag = "ok" if record.get("passed") else "FAIL"
        latency = (record.get("latency_ms") or 0) / 1000.0
        print(f"{flag} ({latency:.1f}s)")

    total_s = time.perf_counter() - started
    print_summary(summarize(records), records)
    print(f"\nWall time: {total_s / 60.0:.1f} min")
    save_records(records, args.output, args.model or status.get("active_model") or "default")
    if args.strict and any(not record.get("passed") for record in records):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
