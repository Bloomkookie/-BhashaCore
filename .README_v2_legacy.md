# BhashaCore: Multilingual FAQ Chatbot

A reliable, fast, zero-hallucination multilingual public-service FAQ chatbot supporting **English, Hindi, Bengali, Punjabi, Assamese, and Nepali**.

BhashaCore is engineered for high accuracy, low latency, and zero-overhead local deployment on standard consumer hardware (including Apple Silicon MacBook Air with 8 GB RAM).

---

## 🚀 Key Highlights & Architecture (v2.0)

1. **Lightweight Deterministic Core:**
   - Pure Python standard library retrieval via `faq_engine.py` and `knowledge_base.py`.
   - **~12 ms retrieval latency** on Apple Silicon with 100% test accuracy on evaluation benchmarks.
   - Preserves Indic Unicode combining marks (matras, viramas, halants) across Devanagari, Bengali, and Gurmukhi scripts.
   - Stopword and domain-term filtering prevents false-positive topic matches.

2. **Extensible Multi-Domain Knowledge Layer:**
   - Multi-domain indexing (`education`, `public_services`, and custom government schemes).
   - Ingests curated local FAQs and sampled subsets from `ai4bharat/Indic-Rag-Suite`.
   - Explicit provenance metadata tracking: source reference, dataset origin, and translation review status.

3. **Hybrid Fallback & Anti-Hallucination Guardrails:**
   - Strict confidence thresholding (`threshold=0.40`) and ambiguity gating (`margin=0.04`).
   - If an unsupported query is asked, the engine safely falls back rather than inventing factual answers.
   - When configured and available, Ollama (`qwen2.5:3b` / `gemma3:4b`) provides grounded answering strictly constrained to the knowledge base context.

4. **Streamlit Interactive UI:**
   - Dynamic language switching for all 6 languages.
   - Real-time system status sidebar (Ollama daemon status, active knowledge domain records).
   - Provenance badges indicating topic confidence (`🟢 High`, `🟡 Moderate`, `🟠 Threshold`) and draft translation notices.

---

## 🛠️ Installation & Setup (macOS)

### 1. Prerequisites
- macOS (tested on macOS Sonoma / Sequoia)
- Python 3.9+ (System Python `/usr/bin/python3` or Homebrew Python)

### 2. Environment Setup
```bash
cd BhashaCore

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install application dependencies
pip install -r requirements.txt
```

### 3. Launch the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Automated Testing & Evaluation

### Run Full Unit Test Suite (44 Tests)
```bash
python3 -m unittest discover -s tests -v
```
All 44 automated tests verify:
- Unicode Indic normalization and matra preservation
- Multilingual tokenization and soft stem matching
- Knowledge base multi-domain loading & schema validation
- Deterministic retrieval across all 6 languages
- Out-of-domain rejection and ambiguity guardrails
- Ollama client offline fault-tolerance
- Translation service fallback contracts

### Run Retrieval Benchmark
```bash
python3 evaluate_faq.py
```
Runs the 60-question expanded benchmark, 18-question baseline, and 10 adversarial out-of-domain queries, exporting results to `faq_evaluation_results.csv`.

---

## 🤖 Optional Ollama Integration

BhashaCore is fully functional **without** an LLM. If you wish to enable generative responses for unassisted queries:

```bash
# Start Ollama daemon
ollama serve

# In a separate terminal, pull recommended models
ollama pull qwen2.5:3b
# Or lightweight alternative:
ollama pull gemma3:4b
```

Configure parameters via environment variables if desired:
```bash
export BHASHACORE_MODEL="qwen2.5:3b"
export BHASHACORE_OLLAMA_URL="http://localhost:11434"
export BHASHACORE_THRESHOLD="0.40"
```

---

## 📊 Evaluation & Quality Verification

| Language | Code | Benchmark Accuracy | Mean Retrieval Latency | Translation Status |
| :--- | :--- | :--- | :--- | :--- |
| **English** | `en` | 100.0% | 12.3 ms | Verified Reference |
| **Hindi** | `hi` | 100.0% | 13.4 ms | Draft (Unverified) |
| **Bengali** | `bn` | 100.0% | 12.4 ms | Draft (Unverified) |
| **Punjabi** | `pa` | 100.0% | 13.1 ms | Draft (Unverified) |
| **Assamese** | `as` | 100.0% | 12.6 ms | Draft (Unverified) |
| **Nepali** | `ne` | 100.0% | 11.5 ms | Draft (Unverified) |

> ⚠️ **Important Testing Notice:** Non-English translations in `faq_data.json` are draft fixtures created for technical benchmarking and must undergo human review by native fluent speakers before public deployment. Test scholarship data is fictional and not official government policy.

---

## 📂 Project Directory Structure

```text
BhashaCore/
├── app.py                     # Streamlit web application with sidebar diagnostics
├── config.py                  # Central configuration & environment variables
├── knowledge_base.py          # Multi-domain knowledge layer & provenance tracker
├── faq_engine.py              # Pure-Python Unicode-aware retrieval engine
├── ollama_client.py           # Fault-tolerant local LLM client
├── translation_service.py     # IndicTrans2 interface & fallback translator
├── ingest_rag_suite.py        # Preprocessing script for AI4Bharat Indic-RAG-Suite
├── evaluate_faq.py            # Automated multi-language evaluation runner
├── compare_models.py          # LLM comparison & benchmarking tool
├── faq_data.json              # Curated multi-domain multilingual FAQ knowledge base
├── data/
│   ├── README.md
│   └── rag_sample.json        # Ingested public-service sample records
├── tests/
│   ├── __init__.py
│   ├── test_app_integration.py
│   ├── test_faq_engine.py
│   └── test_v2_upgrades.py    # Unit tests for v2.0 modules
├── requirements.txt           # Minimal runtime dependencies
└── requirements-research.txt  # Optional research & evaluation dependencies
```
