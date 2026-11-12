
from datasets import load_dataset
import pandas as pd

# BhashaCore: multilingual dataset preparation
# Languages: English, Hindi, Bengali, Punjabi, Assamese, Nepali

DATASET_NAME = "ai4bharat/IN22-Gen"
OUTPUT_FILE = "multilingual_sample.csv"
SAMPLE_SIZE = 100

LANGUAGE_COLUMNS = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Bengali": "ben_Beng",
    "Punjabi": "pan_Guru",
    "Assamese": "asm_Beng",
    "Nepali": "npi_Deva",
}


def main():
    print("=" * 55)
    print("BhashaCore - Multilingual Dataset Preparation")
    print("=" * 55)

    print("\nLoading AI4Bharat IN22-Gen dataset...")
    dataset = load_dataset(DATASET_NAME)

    data = dataset["test"]
    print(f"Dataset loaded: {len(data)} examples available")

    # Check that every requested language column exists
    available_columns = data.column_names
    missing = [
        col for col in LANGUAGE_COLUMNS.values()
        if col not in available_columns
    ]

    if missing:
        raise ValueError(
            f"Missing language columns in dataset: {missing}"
        )

    # Select a small sample for initial inspection
    sample_size = min(SAMPLE_SIZE, len(data))
    sample = data.select(range(sample_size))

    # Export selected languages and their source domains
    columns = list(LANGUAGE_COLUMNS.values()) + ["domain"]
    df = sample.to_pandas()[columns]

    # Rename columns to readable language names
    readable_names = {
        column: language
        for language, column in LANGUAGE_COLUMNS.items()
    }
    readable_names["domain"] = "Domain"
    df = df.rename(columns=readable_names)

    # Save as UTF-8 CSV for Indic scripts
    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nDataset exported successfully!")
    print(f"Rows saved: {len(df)}")
    print(f"Languages: {len(LANGUAGE_COLUMNS)}")
    print(f"Output file: {OUTPUT_FILE}")
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 3 examples:")
    print(df.head(3).to_string(index=False))

    print("\nLanguage data completeness:")
    for language in LANGUAGE_COLUMNS:
        non_empty = df[language].fillna("").astype(str).str.strip()
        valid_count = (non_empty != "").sum()
        print(f"{language}: {valid_count}/{len(df)} non-empty")

    print("\nNext step: build the chatbot evaluation benchmark.")


if __name__ == "__main__":
    main()
