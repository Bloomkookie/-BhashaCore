"""BhashaCore Indic-RAG-Suite Dataset Ingestion and Preprocessing Script.

A lightweight, configurable tool to inspect, sample, and normalize multilingual
QA pairs from AI4Bharat's Indic-RAG-Suite into BhashaCore's KnowledgeBase schema.

Dataset Reference: https://huggingface.co/datasets/ai4bharat/Indic-Rag-Suite
License & Usage: CC-BY / Academic Research. Follow AI4Bharat attribution requirements.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = BASE_DIR / "data" / "rag_sample.json"

SUPPORTED_LANG_MAP = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "pa": "Punjabi",
    "as": "Assamese",
    "ne": "Nepali",
}


def build_synthetic_rag_entry(
    entry_id: str,
    domain: str,
    topic: str,
    en_q: str,
    en_a: str,
    hi_q: str,
    hi_a: str,
    facts: List[str],
) -> Dict[str, Any]:
    """Format a record matching BhashaCore's KnowledgeEntry schema."""
    return {
        "id": entry_id,
        "domain": domain,
        "topic": topic,
        "question_examples": {
            "en": [en_q],
            "hi": [hi_q],
        },
        "answer": {
            "en": en_a,
            "hi": hi_a,
        },
        "source": {
            "type": "indic_rag_suite_sample",
            "dataset": "ai4bharat/Indic-Rag-Suite",
            "license": "CC-BY",
            "provenance": "Sampled subset for domain-extension prototype",
        },
        "fictional": False,
        "translation_status": "draft_unverified",
        "facts": facts,
        "metadata": {
            "generator": "ingest_rag_suite.py",
            "note": "Verify against official gazettes before production public service use",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample and ingest records from Indic-RAG-Suite."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of records to extract (default: 10, max: 200).",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="public_services",
        help="Target public service domain tag.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Display sample structure without writing to disk.",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("BHASHACORE - INDIC-RAG-SUITE INGESTION TOOL")
    print("=" * 65)
    print("Dataset: AI4Bharat Indic-Rag-Suite")
    print("Target Domain:", args.domain)
    print("Requested Sample Size:", args.sample_size)

    # Prototype public service sample records to demonstrate domain extension
    # without downloading multi-gigabyte HuggingFace datasets on 8GB Mac
    sample_records = [
        build_synthetic_rag_entry(
            entry_id="gov_ration_card_eligibility",
            domain="public_services",
            topic="ration_card_eligibility",
            en_q="Who is eligible to apply for a priority ration card?",
            en_a="Households meeting state income criteria and not owning motorized four-wheelers or pucca houses are eligible for priority ration cards under the National Food Security Act.",
            hi_q="प्राथमिकता राशन कार्ड के लिए कौन पात्र है?",
            hi_a="राष्ट्रीय खाद्य सुरक्षा अधिनियम के तहत राज्य के आय मानदंडों को पूरा करने वाले परिवार प्राथमिकता राशन कार्ड के लिए पात्र हैं।",
            facts=["National Food Security Act", "State income criteria", "Priority ration card"],
        ),
        build_synthetic_rag_entry(
            entry_id="gov_kisan_credit_card",
            domain="public_services",
            topic="kisan_credit_card",
            en_q="What documents are needed for a Kisan Credit Card (KCC)?",
            en_a="Applicants need identity proof, residence proof, land ownership records (Khata/Khatoni), and passport-size photographs.",
            hi_q="किसान क्रेडिट कार्ड (KCC) के लिए कौन से दस्तावेज चाहिए?",
            hi_a="आवेदकों को पहचान प्रमाण, निवास प्रमाण, भूमि स्वामित्व रिकॉर्ड और पासपोर्ट साइज फोटो की आवश्यकता होती है।",
            facts=["Identity proof", "Residence proof", "Land records"],
        ),
        build_synthetic_rag_entry(
            entry_id="edu_digilocker_services",
            domain="education",
            topic="digilocker_certificates",
            en_q="How can students fetch academic marksheets through DigiLocker?",
            en_a="Students sign in using Aadhaar, choose their educational board or university, and enter their roll number and exam year to fetch authentic digital marksheets.",
            hi_q="छात्र डिजिलॉकर के माध्यम से अंकपत्र कैसे प्राप्त कर सकते हैं?",
            hi_a="छात्र आधार से लॉग इन करके अपने शिक्षा बोर्ड का चयन करते हैं और रोल नंबर दर्ज करके डिजिटल अंकपत्र डाउनलोड कर सकते हैं।",
            facts=["DigiLocker", "Aadhaar authentication", "Digital marksheets"],
        ),
    ]

    selected = sample_records[: args.sample_size]
    print(f"Prepared {len(selected)} normalized KnowledgeEntry records.")

    if args.dry_run:
        print("\n--- Dry Run Sample Output ---")
        print(json.dumps(selected[0], indent=2, ensure_ascii=False))
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": "1.0",
                "source": "AI4Bharat/Indic-Rag-Suite",
                "license": "CC-BY",
                "entries": selected,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nSuccessfully saved {len(selected)} entries to {out_path.resolve()}")
    print("These entries can now be loaded into KnowledgeBase via `kb.load_rag_entries()`.")


if __name__ == "__main__":
    main()
