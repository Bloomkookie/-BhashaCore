"""BhashaCore Configuration Module.

Centralized configuration with environment-variable overrides for the
conversational LLM, multilingual retrieval, translation, and file ingestion.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Set

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

# ---------------------------------------------------------------------------
# Conversational LLM (Ollama)
# gemma3:4b was measured to be substantially better than qwen2.5:3b on the
# five Indic target languages; qwen2.5:3b remains a lighter fallback.
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL: str = os.environ.get("BHASHACORE_OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.environ.get("BHASHACORE_MODEL", "gemma3:4b")
OLLAMA_FALLBACK_MODELS = ["gemma3:4b", "qwen2.5:3b"]
OLLAMA_TIMEOUT: int = int(os.environ.get("BHASHACORE_TIMEOUT", "180"))
OLLAMA_TEMPERATURE: float = float(os.environ.get("BHASHACORE_TEMPERATURE", "0.2"))
OLLAMA_NUM_PREDICT: int = int(os.environ.get("BHASHACORE_NUM_PREDICT", "512"))
OLLAMA_NUM_CTX: int = int(os.environ.get("BHASHACORE_NUM_CTX", "4096"))
OLLAMA_KEEP_ALIVE: str = os.environ.get("BHASHACORE_KEEP_ALIVE", "10m")

# ---------------------------------------------------------------------------
# FAQ retrieval (deterministic, curated demo knowledge)
# ---------------------------------------------------------------------------
FAQ_MATCH_THRESHOLD: float = float(os.environ.get("BHASHACORE_THRESHOLD", "0.40"))
FAQ_AMBIGUITY_MARGIN: float = float(os.environ.get("BHASHACORE_AMBIGUITY_MARGIN", "0.04"))
FAQ_DIRECT_MIN_SCORE: float = float(os.environ.get("BHASHACORE_FAQ_DIRECT_SCORE", "0.55"))

# ---------------------------------------------------------------------------
# Document RAG
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = int(os.environ.get("BHASHACORE_CHUNK_SIZE", "900"))
CHUNK_OVERLAP: int = int(os.environ.get("BHASHACORE_CHUNK_OVERLAP", "150"))
RETRIEVAL_TOP_K: int = int(os.environ.get("BHASHACORE_TOP_K", "4"))
RETRIEVAL_MIN_LEXICAL: float = float(os.environ.get("BHASHACORE_MIN_LEXICAL", "0.06"))
RETRIEVAL_MIN_SEMANTIC: float = float(os.environ.get("BHASHACORE_MIN_SEMANTIC", "0.45"))
MAX_CONTEXT_CHARS: int = int(os.environ.get("BHASHACORE_CONTEXT_CHARS", "7000"))
MAX_DOCUMENT_CHARS: int = int(os.environ.get("BHASHACORE_MAX_DOC_CHARS", "2000000"))
ALLOWED_EXTENSIONS: Set[str] = {".pdf", ".txt", ".md", ".markdown", ".json", ".csv"}

EMBEDDING_ENABLED: bool = os.environ.get("BHASHACORE_EMBEDDINGS", "true").lower() in ("1", "true", "yes")
EMBEDDING_MODEL: str = os.environ.get("BHASHACORE_EMBED_MODEL", "bge-m3")
EMBEDDING_TIMEOUT: int = int(os.environ.get("BHASHACORE_EMBED_TIMEOUT", "180"))
EMBEDDING_BATCH_SIZE: int = int(os.environ.get("BHASHACORE_EMBED_BATCH", "8"))

# ---------------------------------------------------------------------------
# Translation
# NLLB-200 distilled 600M is the default local translator.  IndicTrans2 is
# preferred automatically if its local directory is present.
# ---------------------------------------------------------------------------
TRANSLATION_ENABLED: bool = os.environ.get("BHASHACORE_TRANSLATION", "true").lower() in ("1", "true", "yes")
NLLB_MODEL_DIR: Path = Path(
    os.environ.get("BHASHACORE_NLLB_DIR", str(MODELS_DIR / "nllb-200-distilled-600M"))
)
INDICTRANS2_EN_INDIC_DIR: Path = Path(
    os.environ.get(
        "BHASHACORE_INDICTRANS_EN_INDIC_DIR",
        str(MODELS_DIR / "indictrans2-en-indic-dist-200M"),
    )
)
INDICTRANS2_INDIC_EN_DIR: Path = Path(
    os.environ.get(
        "BHASHACORE_INDICTRANS_INDIC_EN_DIR",
        str(MODELS_DIR / "indictrans2-indic-en-dist-200M"),
    )
)
TRANSLATION_MAX_INPUT_CHARS: int = int(os.environ.get("BHASHACORE_TRANSLATION_CHARS", "400"))
# Greedy decoding: measured twice as fast as 4-beam search on CPU with no
# observed quality difference on our sample; override if you prefer beam search.
TRANSLATION_NUM_BEAMS: int = int(os.environ.get("BHASHACORE_TRANSLATION_BEAMS", "1"))

# Languages whose direct generation measured poorly with small local models.
# For these, BhashaCore generates in English and machine-translates the result.
QUALITY_TRANSLATION_ASSIST_LANGS: Set[str] = {"pa", "as", "ne"}
SCRIPT_PURITY_MIN: float = float(os.environ.get("BHASHACORE_SCRIPT_PURITY", "0.55"))

# ---------------------------------------------------------------------------
# Conversation memory
# ---------------------------------------------------------------------------
HISTORY_MAX_MESSAGES: int = int(os.environ.get("BHASHACORE_HISTORY_MESSAGES", "8"))
HISTORY_MAX_CHARS: int = int(os.environ.get("BHASHACORE_HISTORY_CHARS", "6000"))

# Supported languages mapping: UI label -> ISO 639-1 internal code
SUPPORTED_LANGUAGES: Dict[str, str] = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Punjabi": "pa",
    "Assamese": "as",
    "Nepali": "ne",
}

LANGUAGE_CODES: Dict[str, str] = {v: k for k, v in SUPPORTED_LANGUAGES.items()}

# File paths
DEFAULT_FAQ_PATH = BASE_DIR / "faq_data.json"
EXPANDED_QUESTIONS_PATH = BASE_DIR / "evaluation_questions_expanded.json"
BASELINE_QUESTIONS_PATH = BASE_DIR / "evaluation_questions.json"


def get_config_summary() -> Dict[str, Any]:
    """Return an overview dictionary of active configuration parameters."""
    return {
        "ollama_url": OLLAMA_BASE_URL,
        "configured_model": OLLAMA_MODEL,
        "fallback_models": list(OLLAMA_FALLBACK_MODELS),
        "timeout_seconds": OLLAMA_TIMEOUT,
        "temperature": OLLAMA_TEMPERATURE,
        "num_ctx": OLLAMA_NUM_CTX,
        "match_threshold": FAQ_MATCH_THRESHOLD,
        "ambiguity_margin": FAQ_AMBIGUITY_MARGIN,
        "faq_direct_min_score": FAQ_DIRECT_MIN_SCORE,
        "translation_enabled": TRANSLATION_ENABLED,
        "embeddings_enabled": EMBEDDING_ENABLED,
        "embedding_model": EMBEDDING_MODEL,
        "retrieval_top_k": RETRIEVAL_TOP_K,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "supported_languages_count": len(SUPPORTED_LANGUAGES),
        "history_max_messages": HISTORY_MAX_MESSAGES,
        "translation_assist_languages": sorted(QUALITY_TRANSLATION_ASSIST_LANGS),
    }
