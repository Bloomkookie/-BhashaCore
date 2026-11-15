# BhashaCore — Six-Language Conversational Assistant

A local, research-prototype chatbot for **English, Hindi, Bengali, Punjabi,
Assamese, and Nepali** that combines a conversational LLM with deterministic FAQ
retrieval, multilingual document RAG, translation, and summarization.

This is an experimental prototype. It is not a certified translation or
government-information service, it has not been reviewed by native speakers, and
general-knowledge answers come from a small local model rather than a verified
source.

---

## What it can do

| Capability | How it works | Provenance shown in UI |
| :--- | :--- | :--- |
| General conversation, science, math, programming | Ollama chat model (`gemma3:4b` by default) | "General knowledge · model-generated, not source-verified" |
| Uploaded-document QA (PDF/TXT/MD/JSON/CSV) | Chunking + hybrid lexical/embedding retrieval + grounded prompt + citation markers | "Document-grounded" plus a per-passage source panel |
| Curated FAQ demo | Deterministic retrieval over `faq_data.json` | "Curated FAQ · demo knowledge base" |
| Translation | NLLB-200 when installed, otherwise the chat model | "Machine translation · generated text" |
| Summarization | Model summary of the supplied/previous text | "Model summary · verify important facts" |
| Follow-up questions | Session-local history plus document context resolution | Same as the resolved route |
| Ambiguous input | A clarification question instead of a guess | "Clarification requested" |
| Missing evidence | An explicit refusal; no invented facts and no source card | "No supporting evidence was found" |

Answers are never presented as verified: every response carries a provenance
label and notes. Refusals never get a fabricated citation, and retrieved
passages that the model did not cite are labeled "retrieved, not cited".

---

## Architecture

```text
Streamlit UI (app.py)
  └── Chatbot orchestrator (chatbot.py)
        ├── Language detection / target selection (language_utils.py)
        ├── Intent router (router.py)
        │     general | document | translation | summarize | clarify
        ├── General answers
        │     ├── curated FAQ lookup (knowledge_base.py + faq_engine.py)
        │     └── Ollama chat model (llm_client.py -> gemma3:4b / qwen2.5:3b)
        ├── Document RAG (documents.py)
        │     ├── ingestion: PDF/TXT/MD/JSON/CSV, page/section metadata, SHA-1 dedupe
        │     ├── retrieval: 0.45 lexical coverage + 0.55 bge-m3 cosine
        │     └── grounded prompt, [n] citations, refusal gate
        ├── Machine translation (translation_service.py)
        │     ├── NLLB-200-distilled-600M (local, en↔Indic, Indic→English→Indic pivot)
        │     └── translation-assisted generation for pa / as / ne
        └── Session memory (last 8 messages, 6000 chars, session-local only)
```

Key design choices:

- **One orchestrator, not a multi-agent system.** Routing is transparent
  keyword/pattern logic; every decision is inspectable and testable offline.
- **Provenance is explicit.** The model is told to answer only from retrieved
  passages for document questions, and the UI separates cited from
  merely-retrieved passages.
- **Prior assistant messages are history, not evidence.** Document answers always
  re-retrieve; a previous answer is never treated as a verified source.
- **Uploaded text is untrusted data.** The system prompts instruct the model to
  ignore instructions found inside documents. This is a mitigation, not a
  guarantee; treat hostile uploads as a residual risk.
- **Memory-safe model swapping.** Ollama is unloaded before NLLB loads and NLLB
  is unloaded afterwards, so the two models do not need to coexist in RAM.

### Language strategy (measured, not assumed)

| Language | Generation path | Why |
| :--- | :--- | :--- |
| English (`en`) | Direct generation | Baseline |
| Hindi (`hi`) | Direct generation | `gemma3:4b` produced usable native-script answers in testing |
| Bengali (`bn`) | Direct generation | Same |
| Punjabi (`pa`) | English generation → NLLB translation | Direct generation was less reliable in early manual checks |
| Assamese (`as`) | English generation → NLLB translation | Same |
| Nepali (`ne`) | English generation → NLLB translation | Same |

Machine-translated answers carry a note saying the text is machine-generated and
has not been reviewed by a person. Stored FAQ translations are marked as
unverified drafts.

Translation safeguards: currency codes (`INR`, `USD`, `Rs.`) are replaced by
opaque tokens during translation and restored afterwards, because NLLB
otherwise transliterates them; decoding is greedy with a four-gram repetition
block. This fixed observed `INR → ইঞ্চি আৰ এছ` and looping-output defects, but
does not make long code-bearing answers translate well.

---

## Models and data actually used

| Component | Artifact | Size | Licence | Status |
| :--- | :--- | :--- | :--- | :--- |
| Chat model | `gemma3:4b` (Ollama) | ~3.4 GB | Gemma Terms of Use | Installed and used by default |
| Fallback chat model | `qwen2.5:3b` (Ollama) | ~1.9 GB | Apache-2.0 | Installed, optional |
| Embeddings | `bge-m3` (Ollama) | ~1.2 GB | MIT (Ollama distribution) | Installed and used for hybrid retrieval |
| Translation | `facebook/nllb-200-distilled-600M` | ~2.4 GB | **CC-BY-NC-4.0 (non-commercial)** | Installed in `models/`, used for pa/as/ne |
| FAQ demo | `faq_data.json`, fictional scholarship | small | project fixture | Used as the curated demo |
| Public-service sample | `data/rag_sample.json`, derived from AI4Bharat Indic-RAG-Suite | small | CC-BY attribution retained in file | Loaded into the FAQ knowledge base when present |

Not integrated: **IndicTrans2** weights are gated on Hugging Face and require an
account with accepted access. The integration code exists but no weights were
downloaded. **IN22-Gen / Indic-RAG-Suite full datasets** were not downloaded;
only the small bundled sample is used. Nothing is downloaded automatically.

---

## Install and launch

Requirements: macOS (tested on Apple Silicon), Python 3.9+, Ollama, and roughly
8 GB of usable RAM when running one model at a time.

```bash
cd BhashaCore

# Reuse the existing environment (do not recreate it unnecessarily)
source .venv/bin/activate
pip install -r requirements.txt        # streamlit, requests, numpy, pypdf

# Check which optional models are present
python setup_models.py

# Fetch only what is missing (explicit opt-in, several GB)
python setup_models.py download all

# Launch
streamlit run app.py
```

Open `http://localhost:8501`. If the port is taken, use
`streamlit run app.py --server.port 8502`.

Notes:

- To use translation-assisted languages, install the optional stack:
  `pip install -r requirements-research.txt` (torch, transformers,
  sentencepiece, huggingface_hub).
- The Streamlit process caches the `Chatbot` object. **Restart the server after
  editing Python code**; a browser rerun is not enough.
- Model configuration is environment-driven, for example:

```bash
export BHASHACORE_MODEL="qwen2.5:3b"        # alternate chat model
export BHASHACORE_TRANSLATION="false"       # disable NLLB entirely
export BHASHACORE_EMBEDDINGS="false"        # lexical-only retrieval
```

---

## How to test each language

In the UI, use language **Auto** for detection or pick a language explicitly.
Good smoke questions:

| Language | Try | Expected behavior |
| :--- | :--- | :--- |
| English | `Why is the sky blue?` | General answer, provenance "General knowledge" |
| Hindi | `प्रकाश संश्लेषण सरल हिंदी में समझाइए।` | Direct native-script answer |
| Bengali | `ভারতের রাজধানী কী?` | Direct native-script answer |
| Punjabi | `ਭਾਰਤ ਦੀ ਰਾਜਧਾਨੀ ਕੀ ਹੈ?` | English generation, machine-translated Punjabi, translation note |
| Assamese | `ভাৰতৰ ৰাজধানী কি?` | English generation, machine-translated Assamese |
| Nepali | `भारतको राजधानी के हो?` | English generation, machine-translated Nepali |
| Any | Upload `evaluation/fixtures/library_policy.txt`, then ask `According to the uploaded file, what is the membership fee?` | Grounded answer with `[1]` and a source card |
| Any | Ask about the library director in the same document | Explicit refusal, no source card |

Detection caveat: Bengali and Assamese share a script. A very short input such as
`কি?` has no distinguishing words and is detected as Bengali; choose Assamese in
the selector when detection cannot know. Romanized Indic input is reported as
low-confidence English-like text rather than silently guessed.

### Automated tests

```bash
python -m unittest discover -s tests -q
```

The offline suite (99 tests) covers language detection and script purity,
routing in six languages, FAQ retrieval, document ingestion and dedupe, citation
handling, refusal behavior, follow-up context, translation-assisted generation,
model failures, streaming, and history trimming. It stubs the LLM, translator,
and embeddings, so it runs without Ollama or NLLB.

### Multilingual behavioral evaluation

```bash
# Validate the case file and print the coverage matrix (no model calls)
python evaluate_multilingual.py

# Run the real chatbot (Ollama + optional NLLB)
python evaluate_multilingual.py --live
python evaluate_multilingual.py --live --language hi,bn --per-category 2
```

The draft suite in `evaluation/multilingual_cases.json` has 70 cases across 12
categories (general, science, programming, paraphrases, translation, summary,
FAQ, document QA, missing evidence, ambiguity, multi-turn, mixed language).
Every answer is saved to `evaluation/results_<model>.jsonl/.csv` for human review.
Automated checks cover route, evidence flag, citation count, keyword presence,
and script purity. **They are proxies, not proof of factual or translation
quality.**

---

## Evaluation results and resource use

See `evaluation/results_gemma3-4b_summary.json`, `.csv`, and `.jsonl` for the
latest run. Summary:

| Language | Cases | Passed | Mean turn latency | Mean script purity |
| :--- | :--- | :--- | :--- | :--- |
| English | 12 | 12 | 16.0 s | 1.000 |
| Hindi | 13 | 13 | 18.1 s | 0.944 |
| Bengali | 12 | 12 | 15.9 s | 0.958 |
| Punjabi | 11 | 11 | 37.8 s | 0.971 |
| Assamese | 11 | 11 | 36.4 s | 0.983 |
| Nepali | 11 | 11 | 36.8 s | 0.964 |
| **Total** | **70** | **70** | — | — |

Run conditions: `gemma3:4b`, bge-m3 hybrid retrieval, NLLB greedy decoding,
70 cases in 31.0 minutes. "Passed" means the automated route/provenance/keyword/
script checks passed — it does **not** mean the answers were human-reviewed or
that translation quality is good. Reading the saved answers still shows real
defects: Assamese/Punjabi/Nepali code examples are mangled by translation, and
some translated answers are awkward or over-assertive.

Resource observations:

- Peak process RSS during the six-language suite was about 3.0 GB, including
  torch/NLLB inside the evaluator.
- Ollama reports the chat model using roughly 2.3 GB of GPU memory.
- English/Hindi/Bengali direct answers typically take tens of seconds per turn on
  CPU-only hardware; Punjabi/Assamese/Nepali translation-assisted turns can take
  one to two minutes because the chat model and NLLB are swapped in and out.
- FAQ, clarification, and templated refusal paths return immediately without
  calling a model.

---

## Known limitations

1. **No native-speaker review.** Prompts, FAQ translations, and evaluation case
   translations are draft text. Machine translations are labeled as such.
2. **Automatic checks are shallow.** Route/keyword/script checks cannot verify
   factual correctness, naturalness, or citation entailment. Read the saved
   answers before making language-quality claims.
3. **General answers are unverified.** The model can be wrong, especially on
   niche facts. Provenance makes this explicit but does not fix it.
4. **Indic detection is heuristic.** Bengali/Assamese share a script; romanized
   input is not reliably detectable. Manual selection is the reliable path.
5. **Translation-assisted answers depend on NLLB.** Quality varies by language
   and domain, and Indic→Indic requests pivot through English.
6. **Citation checking is structural.** The system validates that cited numbers
   exist among retrieved passages; it does not prove that a passage supports the
   specific claim.
7. **PDF support is text-only.** Scanned/image PDFs are not OCR'd.
8. **Prompt-injection risk remains.** Retrieved text is treated as untrusted by
   instruction, but a determined hostile document could still influence output.
9. **Model swapping is slow.** Translation-assist round trips unload and reload
   models to fit in memory.
10. **The FAQ is fictional.** The scholarship demo data is not real policy.

---

## File map

```text
BhashaCore/
├── app.py                        Streamlit UI, diagnostics, provenance panels
├── chatbot.py                    Orchestrator, memory, provenance, refusals
├── router.py                     Intent routing in six languages
├── language_utils.py             Detection, script purity, romanised heuristics
├── prompts.py                    Prompts and localized refusal/clarification text
├── llm_client.py                 Ollama chat client, unload helper, streaming
├── translation_service.py        NLLB / IndicTrans2 interface and fallback
├── documents.py                  Ingestion, chunking, hybrid retrieval, citations
├── knowledge_base.py             Curated FAQ index and provenance metadata
├── faq_engine.py                 Unicode-aware lexical retrieval
├── config.py                     Environment-driven configuration
├── setup_models.py               Explicit model check/download helper
├── evaluate_multilingual.py      Six-language behavioural evaluation runner
├── evaluation/
│   ├── multilingual_cases.json   70-case draft suite
│   ├── fixtures/library_policy.txt
│   └── results_*                 Saved answers, metrics and summaries
├── tests/                        99 offline unit tests
├── data/rag_sample.json          Bundled Indic-RAG-Suite-derived sample
├── faq_data.json                 Fictional scholarship FAQ demo
├── requirements.txt              Runtime dependencies
└── requirements-research.txt     Optional translation/research dependencies
```

Legacy research scripts kept for reference: `evaluate_faq.py` (FAQ benchmark),
`compare_models.py` (older LLM comparison), `evaluate_chatbot.py` (older
single-domain runner), `prepare_dataset.py`, `ingest_rag_suite.py`,
`test_dataset.py`, `create_quality_review.py`, `review_results.py`. They use
older result formats and are not part of the current test path.

---

## Prioritized next steps

1. Native-speaker review of FAQ translations, UI copy, and evaluation prompts.
2. Human grading of the saved evaluation answers (correctness and fluency),
   replacing keyword proxies for quality claims.
3. Cache NLLB between translation steps, or run translation in a separate
   process, to cut the 1–2 minute assist latency.
4. Add citation-entailment checks so `[n]` is verified against the claim.
5. Obtain IndicTrans2 access and benchmark it against NLLB for pa/as/ne.
6. Improve shared-script and romanized detection, or make the language selector
   the enforced default for short inputs.
7. OCR support for scanned PDFs.
8. Harden prompt-injection defenses and add a document trust warning in the UI.
