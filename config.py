"""BhashaCore Configuration Module.

Centralized configuration management with environment variable overrides
for LLM endpoints, retrieval thresholds, timeouts, and file paths.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"

# Ollama / LLM configuration
OLLAMA_BASE_URL: str = os.environ.get("BHASHACORE_OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.environ.get("BHASHACORE_MODEL", "qwen2.5:3b")
OLLAMA_TIMEOUT: int = int(os.environ.get("BHASHACORE_TIMEOUT", "120"))
OLLAMA_TEMPERATURE: float = float(os.environ.get("BHASHACORE_TEMPERATURE", "0.1"))
OLLAMA_NUM_PREDICT: int = int(os.environ.get("BHASHACORE_NUM_PREDICT", "300"))

# FAQ & Retrieval thresholds
FAQ_MATCH_THRESHOLD: float = float(os.environ.get("BHASHACORE_THRESHOLD", "0.40"))
FAQ_AMBIGUITY_MARGIN: float = float(os.environ.get("BHASHACORE_AMBIGUITY_MARGIN", "0.04"))

# Translation settings
TRANSLATION_ENABLED: bool = os.environ.get("BHASHACORE_TRANSLATION", "false").lower() in ("1", "true", "yes")

# Supported languages mapping: UI label -> ISO 639-1 / BhashaCore internal code
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
        "timeout_seconds": OLLAMA_TIMEOUT,
        "temperature": OLLAMA_TEMPERATURE,
        "match_threshold": FAQ_MATCH_THRESHOLD,
        "ambiguity_margin": FAQ_AMBIGUITY_MARGIN,
        "translation_enabled": TRANSLATION_ENABLED,
        "supported_languages_count": len(SUPPORTED_LANGUAGES),
    }
