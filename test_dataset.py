"""BhashaCore: Multilingual dataset inspection and verification script.

Inspects the AI4Bharat IN22-Gen multilingual sample across all six supported languages:
English, Hindi, Bengali, Punjabi, Assamese, and Nepali.

Works both offline using the bundled `multilingual_sample.csv` and online
via HuggingFace datasets if network access is available.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

SAMPLE_FILE = Path(__file__).resolve().parent / "multilingual_sample.csv"
LANGUAGES = ["English", "Hindi", "Bengali", "Punjabi", "Assamese", "Nepali"]


def verify_local_sample() -> bool:
    print("=" * 60)
    print("BhashaCore - Multilingual Dataset Inspection")
    print("=" * 60)

    if not SAMPLE_FILE.exists():
        print(f"Local sample file not found: {SAMPLE_FILE}")
        return False

    print(f"Reading local dataset sample: {SAMPLE_FILE.name}")
    df = pd.read_csv(SAMPLE_FILE, encoding="utf-8-sig")

    print(f"Total sample rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    missing = [lang for lang in LANGUAGES if lang not in df.columns]
    if missing:
        print(f"ERROR: Missing language columns: {missing}")
        return False

    print("\nLanguage completeness check:")
    for lang in LANGUAGES:
        valid = (df[lang].fillna("").astype(str).str.strip() != "").sum()
        pct = (valid / len(df)) * 100
        print(f"  {lang:<10}: {valid}/{len(df)} non-empty ({pct:.1f}%)")

    print("\nFirst row sample across all 6 languages:")
    first = df.iloc[0]
    for lang in LANGUAGES:
        val = str(first[lang])
        preview = (val[:75] + "...") if len(val) > 75 else val
        print(f"  {lang:<10}: {preview}")

    print("\nDataset verification passed successfully!")
    return True


def try_online_dataset():
    """Attempt loading live dataset from HuggingFace if network is available."""
    try:
        from datasets import load_dataset
        print("\nAttempting to connect to HuggingFace for ai4bharat/IN22-Gen...")
        ds = load_dataset("ai4bharat/IN22-Gen", split="test")
        print(f"Successfully loaded IN22-Gen from HuggingFace ({len(ds)} examples).")
    except Exception as exc:
        print(f"\nNote: Online HuggingFace connection not available ({type(exc).__name__}).")
        print("Using bundled 'multilingual_sample.csv' (offline verification).")


if __name__ == "__main__":
    success = verify_local_sample()
    try_online_dataset()
