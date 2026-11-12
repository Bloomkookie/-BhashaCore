# BhashaCore 🌐

**BhashaCore** is a reliable, lightweight multilingual FAQ chatbot designed for public-service scenarios. It supports six Indian languages: **English, Hindi, Bengali, Punjabi, Assamese, and Nepali**.

BhashaCore uses an intent-driven similarity engine (`faq_engine.py`) with Indic Unicode-preserving normalization, content-word scoring, and confidence gating. It matches user inquiries directly against curated multilingual FAQ entries in `faq_data.json`, delivering deterministic responses with zero hallucinations and without requiring expensive model training or running local LLMs.

---

> [!WARNING]
> ### Prototype & Test Data Notice
> 1. **Fictional Test Fixture:** The scholarship scheme used in this prototype is **completely fictional** and intended strictly for technical evaluation. It is **not** real government policy or public-service guidance.
> 2. **Draft Translations:** All non-English responses (Hindi, Bengali, Punjabi, Assamese, Nepali) are **starter editing drafts** (`draft_unverified`) and have **not** yet been independently certified by native fluent speakers. Before any public deployment, each answer must undergo review by qualified native speakers.

---

## Architecture Overview

```
BhashaCore/
├── app.py                           # Interactive Streamlit chatbot UI
├── faq_engine.py                    # Core retrieval, Unicode normalization & similarity engine
├── faq_data.json                    # Multilingual FAQ knowledge base (6 topics, 6 languages)
├── evaluate_faq.py                  # Automated benchmark evaluation runner
├── requirements.txt                 # Minimal dependencies to run the chatbot (Streamlit)
├── requirements-research.txt        # Optional dependencies for research & dataset scripts
├── .streamlit/config.toml           # Streamlit headless & telemetry-free configuration
├── tests/
│   ├── test_faq_engine.py           # 31 unit & regression tests across all 6 languages
│   └── test_app_integration.py      # Streamlit app integration & backward-compatibility tests
├── evaluation_questions.json        # Baseline test scenarios (18 questions)
├── evaluation_questions_expanded.json # Expanded benchmark (10 scenarios x 6 languages = 60 questions)
├── multilingual_sample.csv          # 100-row sample from AI4Bharat IN22-Gen (culture domain)
├── prepare_dataset.py               # IN22-Gen dataset extraction script
├── test_dataset.py                  # Dataset integrity and completeness verification
├── evaluate_chatbot.py              # Optional Ollama LLM evaluation script
├── compare_models.py                # Optional Ollama multi-model benchmark (3B vs 7B)
├── create_quality_review.py         # Human evaluation template generator
└── review_results.py                # Review template creator for evaluation results
```

### Key Technical Enhancements

1. **Indic Unicode Normalization:**
   Standard regex `[^\w\s]` strips Unicode combining marks (matras, viramas, halants), shredding Indic words into broken single characters (e.g. `छात्रवृत्ति` became `['छ', 'त', 'रव', 'त', 'त']`). `faq_engine.py` implements NFC Unicode normalization preserving letters (`L`), combining marks (`M`), and numbers (`N`).

2. **Domain-Specific Stopword & Intent Isolation:**
   Filtering out general grammatical stopwords and common domain terms (`scholarship`, `छात्रवृत्ति`, `বৃত্তি`, `ਵਜ਼ੀਫ਼ਾ`) isolates genuine intent words (e.g., `documents`, `deadline`, `amount`, `eligibility`, `contact`), preventing boilerplate text from triggering false-positive matches.

3. **Anti-Hallucination & Ambiguity Guardrails:**
   - Unsupported / out-of-domain questions (weather, general knowledge, unsupported policies) return `NO_MATCH` with a polite refusal to guess.
   - Ambiguous queries where two topics have indistinguishable scores (margin < 0.04) or single ambiguous words prompt the user to rephrase rather than guessing.
   - Missing translations trigger an explicit fallback stating that an answer is pending fluent review in that language.

---

## Getting Started on macOS

### 1. Prerequisites
- macOS 12+ (Apple Silicon or Intel)
- Python 3.9, 3.10, 3.11, or 3.12

Check your Python version:
```bash
python3 --version
```

### 2. Set Up Virtual Environment

Open Terminal and navigate to the project root:
```bash
cd ~/Downloads/BhashaCore

# Create an isolated virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```

### 3. Install Minimal Dependencies to Run the App

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

*(Optional: If you intend to run dataset scripts or Ollama benchmarks, install `pip install -r requirements-research.txt`).*

### 4. Launch the Application

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## Testing & Quality Verification

### Run the Automated Unit Test Suite

Run the built-in test suite (no external test runner needed):
```bash
python3 -m unittest discover -s tests -v
```

**Test Coverage (31 automated tests):**
- Latin and Indic normalization across Hindi, Bengali, Punjabi, Assamese, and Nepali
- Preservation of vowel signs, matras, viramas, and conjuncts
- Morphological stem matching for Indic inflections
- Semantic similarity scoring and stopword filtering
- Schema validation of `faq_data.json` across all six languages
- 100% precision on the 60-question expanded benchmark
- Anti-hallucination rejection of out-of-domain and unsupported questions
- Ambiguous query rejection
- Missing translation fallback handling
- Streamlit application module integration and export compatibility

### Run the Multilingual FAQ Benchmark Tool

Evaluate retrieval accuracy, confidence scores, and per-query latency across all 60 test questions:
```bash
python3 evaluate_faq.py
```

Sample output:
```
======================================================================
EVALUATION SUMMARY RESULTS
======================================================================
Language     Code   Tested   Passed   Accuracy   Mean Latency
----------------------------------------------------------------------
English      en     10       10       100.0%      12.11 ms
Hindi        hi     10       10       100.0%      13.52 ms
Bengali      bn     10       10       100.0%      12.40 ms
Punjabi      pa     10       10       100.0%      13.14 ms
Assamese     as     10       10       100.0%      12.50 ms
Nepali       ne     10       10       100.0%      11.60 ms
----------------------------------------------------------------------
OVERALL      all    60       60       100.0%      12.55 ms
======================================================================
Baseline Benchmark Pass Rate:       18/18 (100.0%)
Anti-Hallucination Rejection Rate: 10/10 (100.0%)
```

---

## Research & Evaluation Workflows (Optional)

These scripts are optional research tools and are kept separate from the minimal chatbot runtime:

### 1. Dataset Verification
Verify the bundled 100-row AI4Bharat IN22-Gen sample:
```bash
python3 test_dataset.py
```

### 2. Dataset Preparation (Online)
To re-download and export a sample from HuggingFace (requires internet access):
```bash
python3 prepare_dataset.py
```

### 3. Ollama LLM Evaluation (Optional)
If you wish to benchmark against local LLMs:
1. Ensure Ollama is installed and running:
   ```bash
   ollama serve
   ```
2. Pull the evaluation models:
   ```bash
   ollama pull qwen2.5:3b
   ollama pull qwen2.5:7b
   ```
3. Run the evaluation:
   ```bash
   python3 evaluate_chatbot.py
   python3 compare_models.py
   ```

### 4. Human Quality Review Template
Generate a review sheet for native fluent speakers to score factual correctness, completeness, and language fluency:
```bash
python3 create_quality_review.py
```
This generates `quality_review_template.csv` ready for human evaluation.

---

## Troubleshooting on macOS

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `PermissionError: credentials.toml` | Streamlit attempted to write to `~/.streamlit` on first launch | A pre-configured `.streamlit/config.toml` is provided in the repository with `gatherUsageStats = false`. Set `STREAMLIT_BROWSER_GATHER_USAGE_STATS=false` if needed. |
| `ModuleNotFoundError: No module named 'streamlit'` | Virtual environment is not activated | Activate the virtual environment with `source .venv/bin/activate` before running commands. |
| Corrupted or missing FAQ file | `faq_data.json` deleted or moved | Ensure `faq_data.json` is located in the same directory as `app.py`. |
| Indic characters appear as squares/tofu in terminal | Terminal lacks Indic Unicode fonts | Use the modern macOS Terminal or iTerm2 with fonts such as *Kohinoor*, *Bangla MN*, *Gurmukhi MN*, or *Devanagari Sangam MN*. The web browser interface renders all scripts natively. |

---

## Language Quality Review Checklist

Before using non-English answers in any public demonstration:
1. **Hindi (`hi`):** Review by native Hindi speaker for natural grammar, appropriate formal tone (*आप*), and clarity.
2. **Bengali (`bn`):** Check formal verb endings and document vocabulary (*নথি*, *পরিচয়পত্র*).
3. **Punjabi (`pa`):** Verify Gurmukhi spellings, auxiliary verbs (*ਹਨ*, *ਲੋੜੀਂਦੇ*), and vocabulary (*ਵਜ਼ੀਫ਼ਾ*, *ਦਸਤਾਵੇਜ਼*).
4. **Assamese (`as`):** Confirm unique Assamese characters (*ৰ*, *ৱ*) and case inflections (*-ৰ*, *-লৈ*).
5. **Nepali (`ne`):** Verify polite imperative forms (*दिनुहोस्*, *चाहिन्छ*) and administrative terminology (*कागजात*, *अभिलेख*).
