"""BhashaCore translation service.

Provides a real local neural translator based on Meta's NLLB-200 distilled
600M model, plus interfaces for AI4Bharat IndicTrans2 when its (gated) weights
and toolkit are available.  A deterministic fallback backend is retained for
dependency-free operation and tests.

NLLB-200 covers all six BhashaCore languages through Flores-200 codes.  On an
8 GB machine the translator is loaded only while translating, and Ollama chat
models are unloaded first to avoid memory contention.

Licensing: NLLB-200 is released under CC-BY-NC-4.0 (non-commercial research).
IndicTrans2 is MIT.
"""

from __future__ import annotations

import gc
import logging
import os
import re
import string
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from config import (
    INDICTRANS2_EN_INDIC_DIR,
    INDICTRANS2_INDIC_EN_DIR,
    NLLB_MODEL_DIR,
    TRANSLATION_ENABLED,
    TRANSLATION_MAX_INPUT_CHARS,
    TRANSLATION_NUM_BEAMS,
)
from llm_client import unload_model
from language_utils import LANGUAGE_NAMES

logger = logging.getLogger("bhashacore.translation")

# Shared Flores-200 codes used by both IndicTrans2 and NLLB-200.
INDICTRANS2_LANG_CODES: Dict[str, str] = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "pa": "pan_Guru",
    "as": "asm_Beng",
    "ne": "npi_Deva",
}

NLLB_LANG_CODES: Dict[str, str] = dict(INDICTRANS2_LANG_CODES)

# NLLB transliterates currency codes (INR became "ইঞ্চি আৰ এছ" in Assamese), so
# codes are swapped for opaque tokens before translation and restored after.
_CURRENCY_RE = re.compile(r"\b(?:INR|USD|EUR|GBP|Rs\.?)\b")
_TOKEN_RE = re.compile(r"ZQXCUR[A-Z]?ZQX")


def protect_currency_tokens(text: str) -> tuple:
    mapping: Dict[str, str] = {}
    tokens = iter(string.ascii_uppercase)

    def replace(match: "re.Match[str]") -> str:
        original = match.group(0)
        for token, value in mapping.items():
            if value == original:
                return token
        token = f"ZQXCUR{next(tokens)}ZQX"
        mapping[token] = original
        return token

    return _CURRENCY_RE.sub(replace, text), mapping


def restore_currency_tokens(text: str, mapping: Dict[str, str]) -> str:
    for token, original in mapping.items():
        text = text.replace(token, original)
    return _TOKEN_RE.sub("", text)



@dataclass
class TranslationResult:
    text: str
    status: str  # success | unavailable | error | unchanged
    backend: str
    src: str
    tgt: str
    error: Optional[str] = None
    latency_ms: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in ("success", "unchanged") and bool(self.text)


class TranslationBackend:
    """Abstract interface for BhashaCore translation engines."""

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        raise NotImplementedError

    def is_available(self) -> bool:
        raise NotImplementedError


class FallbackTranslationBackend(TranslationBackend):
    """Returns the input unchanged when no neural translator is available."""

    def __init__(self) -> None:
        self._cache: Dict[str, str] = {}

    def is_available(self) -> bool:
        return True

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        if src_lang == tgt_lang:
            return text
        logger.info(
            "Fallback translator called (%s -> %s); no neural translator loaded.",
            src_lang,
            tgt_lang,
        )
        return text


def split_for_translation(text: str, max_chars: int = TRANSLATION_MAX_INPUT_CHARS) -> List[str]:
    """Split text into sentence-sized pieces that fit the translator's window."""
    text = (text or "").strip()
    if not text:
        return []

    import re

    pieces = re.split(r"(?<=[.!?।॥])\s+|\n{2,}", text)
    chunks: List[str] = []
    current = ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if len(piece) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for start in range(0, len(piece), max_chars):
                chunks.append(piece[start : start + max_chars])
            continue
        if current and len(current) + 1 + len(piece) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = f"{current} {piece}".strip()
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk]


class NLLBTranslationService(TranslationBackend):
    """Local NLLB-200 translation with explicit memory management."""

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        enabled: bool = TRANSLATION_ENABLED,
        backend_name: str = "NLLB-200-distilled-600M",
    ) -> None:
        self.model_dir = Path(model_dir or NLLB_MODEL_DIR)
        self.enabled = enabled
        self.backend_name = backend_name
        self._tokenizer = None
        self._model = None
        self.last_error: Optional[str] = None

    def is_installed(self) -> bool:
        return (self.model_dir / "config.json").exists() and (
            (self.model_dir / "pytorch_model.bin").exists()
            or (self.model_dir / "model.safetensors").exists()
        )

    def is_available(self) -> bool:
        return self.enabled and self.is_installed()

    def _load(self) -> bool:
        if self._model is not None and self._tokenizer is not None:
            return True
        if not self.is_available():
            self.last_error = (
                f"Translation model not found at {self.model_dir}. "
                "Run `python setup_models.py` to download it."
            )
            return False
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            torch.set_num_threads(max(1, min(4, os.cpu_count() or 4)))
            logger.info("Loading %s from %s", self.backend_name, self.model_dir)
            self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                str(self.model_dir), torch_dtype=torch.float32
            )
            self._model.eval()
            return True
        except Exception as exc:  # pragma: no cover - depends on optional deps
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.error("Failed to load translator: %s", self.last_error)
            self.unload()
            return False

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch

            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    def _translate_direction(self, text: str, src: str, tgt: str) -> str:
        import torch

        tokenizer = self._tokenizer
        model = self._model
        tokenizer.src_lang = NLLB_LANG_CODES[src]
        forced_bos = tokenizer.convert_tokens_to_ids(NLLB_LANG_CODES[tgt])
        outputs: List[str] = []
        chunks = split_for_translation(text)
        batch_size = 4
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos,
                    max_length=512,
                    num_beams=TRANSLATION_NUM_BEAMS,
                    no_repeat_ngram_size=4,
                )
            outputs.extend(
                tokenizer.batch_decode(generated, skip_special_tokens=True)
            )
        return " ".join(part.strip() for part in outputs if part.strip())

    def translate(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        unload_after: bool = True,
        ollama_model_to_unload: Optional[str] = None,
    ) -> TranslationResult:
        t0 = time.perf_counter()
        src = src_lang if src_lang in NLLB_LANG_CODES else "en"
        tgt = tgt_lang if tgt_lang in NLLB_LANG_CODES else "en"

        if not text or not text.strip():
            return TranslationResult("", "unchanged", self.backend_name, src, tgt)
        if src == tgt:
            return TranslationResult(text, "unchanged", self.backend_name, src, tgt)
        if not self.is_available():
            return TranslationResult(
                text,
                "unavailable",
                self.backend_name,
                src,
                tgt,
                error=self.last_error or "Translator is not installed.",
            )

        if ollama_model_to_unload:
            unload_model(ollama_model_to_unload)

        if not self._load():
            return TranslationResult(
                text, "unavailable", self.backend_name, src, tgt, error=self.last_error
            )

        notes: List[str] = []
        protected, currency_tokens = protect_currency_tokens(text)
        try:
            if src == "en" or tgt == "en":
                translated = self._translate_direction(protected, src, tgt)
            else:
                intermediate = self._translate_direction(protected, src, "en")
                translated = self._translate_direction(intermediate, "en", tgt)
                notes.append("Indic-to-Indic translation pivoted through English.")

            translated = restore_currency_tokens((translated or "").strip(), currency_tokens)
            status = "success" if translated else "error"
            return TranslationResult(
                text=translated or text,
                status=status,
                backend=self.backend_name,
                src=src,
                tgt=tgt,
                error=None if translated else "Translator returned empty output.",
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                notes=notes,
            )
        except Exception as exc:  # pragma: no cover - depends on optional deps
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.error("Translation failed: %s", self.last_error)
            return TranslationResult(
                text,
                "error",
                self.backend_name,
                src,
                tgt,
                error=self.last_error,
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )
        finally:
            if unload_after:
                self.unload()


class IndicTrans2Prototype(TranslationBackend):
    """Optional IndicTrans2 backend (used only when local weights exist)."""

    def __init__(
        self,
        en_indic_dir: Optional[Path] = None,
        indic_en_dir: Optional[Path] = None,
    ) -> None:
        self.en_indic_dir = Path(en_indic_dir or INDICTRANS2_EN_INDIC_DIR)
        self.indic_en_dir = Path(indic_en_dir or INDICTRANS2_INDIC_EN_DIR)
        self.model = None
        self.tokenizer = None
        self._direction = None
        self._processor = None
        self.last_error: Optional[str] = None

    def is_available(self) -> bool:
        return self.en_indic_dir.exists() and self.indic_en_dir.exists()

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        self._processor = None
        self._direction = None
        gc.collect()

    def _load(self, direction: str) -> bool:
        directory = self.en_indic_dir if direction == "en-indic" else self.indic_en_dir
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            from IndicTransToolkit.processor import IndicProcessor  # type: ignore

            torch.set_num_threads(max(1, min(4, os.cpu_count() or 4)))
            self.tokenizer = AutoTokenizer.from_pretrained(str(directory), trust_remote_code=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                str(directory), trust_remote_code=True, torch_dtype=torch.float32
            )
            self.model.eval()
            self._processor = IndicProcessor(inference=True)
            self._direction = direction
            return True
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.unload()
            return False

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        if not self.is_available():
            return text
        if src_lang == tgt_lang:
            return text
        if src_lang != "en" and tgt_lang != "en":
            return text
        direction = "en-indic" if src_lang == "en" else "indic-en"
        if self._direction != direction and not self._load(direction):
            return text
        try:
            import torch

            batch = self._processor.preprocess_batch(
                [text],
                src_lang=INDICTRANS2_LANG_CODES[src_lang],
                tgt_lang=INDICTRANS2_LANG_CODES[tgt_lang],
            )
            inputs = self.tokenizer(
                batch, truncation=True, padding="longest", return_tensors="pt"
            )
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs, max_length=512, num_beams=TRANSLATION_NUM_BEAMS
                )
            decoded = self.tokenizer.batch_decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            output = self._processor.postprocess_batch(
                decoded, lang=INDICTRANS2_LANG_CODES[tgt_lang]
            )
            return (output[0] if output else text).strip()
        except Exception as exc:  # pragma: no cover
            self.last_error = f"{type(exc).__name__}: {exc}"
            return text


def get_translation_backend(enabled: bool = False) -> TranslationBackend:
    """Legacy factory retained for backward compatibility."""
    if enabled:
        prototype = IndicTrans2Prototype()
        if prototype.is_available() and prototype._load("en-indic"):
            return prototype
    return FallbackTranslationBackend()


def get_default_translator() -> NLLBTranslationService:
    """Return the production translator (availability is checked lazily)."""
    return NLLBTranslationService()


def language_label(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)
