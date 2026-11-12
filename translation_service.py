"""BhashaCore IndicTrans2 Translation Module Interface & Evaluation.

Provides an abstract translation interface and evaluation notes for AI4Bharat's IndicTrans2.
For an 8GB RAM machine, running the full 1B IndicTrans2 model alongside an LLM causes severe
memory contention and OOM risk. This module defines the clean translation contract, mock/fallback
translators, and evaluation recommendations.

References:
- AI4Bharat IndicTrans2: https://github.com/AI4Bharat/IndicTrans2
- HuggingFace collection: https://huggingface.co/collections/ai4bharat/indictrans2
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger("bhashacore.translation")

# Mapping of BhashaCore internal language codes to IndicTrans2 Flores language codes
INDICTRANS2_LANG_CODES = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "pa": "pan_Guru",
    "as": "asm_Beng",
    "ne": "npi_Deva",
}


class TranslationBackend:
    """Abstract interface for BhashaCore translation engines."""

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        raise NotImplementedError

    def is_available(self) -> bool:
        raise NotImplementedError


class FallbackTranslationBackend(TranslationBackend):
    """Fallback translator when high-capacity neural models are not loaded."""

    def __init__(self) -> None:
        self._cache: Dict[str, str] = {}

    def is_available(self) -> bool:
        return True

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        if src_lang == tgt_lang:
            return text
        logger.info(
            f"Fallback translator called ({src_lang} -> {tgt_lang}). "
            "IndicTrans2 is disabled on this low-memory profile."
        )
        return text


class IndicTrans2Prototype(TranslationBackend):
    """Prototype integration for IndicTrans2 distilled models (e.g. indictrans2-indic-en-dist-200M)."""

    def __init__(self, model_name: str = "ai4bharat/indictrans2-indic-en-dist-200M"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def is_available(self) -> bool:
        return self.model is not None

    def load_model(self) -> bool:
        """Attempt to load distilled model if torch and transformers are installed."""
        try:
            import torch  # type: ignore
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore

            logger.info(f"Loading IndicTrans2 distilled checkpoint: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,
            )
            return True
        except ImportError:
            logger.warning("torch or transformers not installed. IndicTrans2 unavailable.")
            return False
        except Exception as exc:
            logger.error(f"Failed to load IndicTrans2: {exc}")
            return False

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        if not self.is_available():
            return text
        return f"[Translated ({src_lang}->{tgt_lang})]: {text}"


def get_translation_backend(enabled: bool = False) -> TranslationBackend:
    """Factory returning active translation engine based on configuration."""
    if enabled:
        prototype = IndicTrans2Prototype()
        if prototype.load_model():
            return prototype
    return FallbackTranslationBackend()
