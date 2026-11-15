"""Advanced Ollama chat/embedding client used by the conversational pipeline.

The legacy ``ollama_client`` module is preserved for backward compatibility.
This module adds model resolution, non-streaming/streaming chat, multilingual
embeddings, and memory-aware model unloading.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from typing import Any, Dict, Iterable, List, Optional

import requests

from config import (
    EMBEDDING_MODEL,
    EMBEDDING_TIMEOUT,
    OLLAMA_BASE_URL,
    OLLAMA_FALLBACK_MODELS,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    OLLAMA_TEMPERATURE,
    OLLAMA_TIMEOUT,
)
from ollama_client import check_ollama_status

logger = logging.getLogger("bhashacore.llm")


def model_matches(installed: str, target: str) -> bool:
    if installed == target:
        return True
    installed_base = installed.split(":")[0]
    target_base = target.split(":")[0]
    if ":" not in target and installed_base == target_base:
        return True
    if ":" not in installed and installed_base == target_base:
        return True
    return False


def resolve_model(
    preferred: Optional[str] = None,
    base_url: Optional[str] = None,
    status: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Pick the preferred model if installed, otherwise the configured fallback."""
    preferred_model = preferred or OLLAMA_MODEL
    current = status if status is not None else check_ollama_status(base_url=base_url)
    installed: List[str] = list(current.get("models", []))
    result: Dict[str, Any] = {
        "model": preferred_model,
        "available": False,
        "installed": installed,
        "fell_back": False,
        "error": current.get("error"),
    }
    if not current.get("available"):
        return result

    for name in installed:
        if model_matches(name, preferred_model):
            result.update(model=name, available=True)
            return result

    for candidate in OLLAMA_FALLBACK_MODELS:
        for name in installed:
            if model_matches(name, candidate):
                result.update(model=name, available=True, fell_back=True)
                return result

    if installed:
        result.update(model=installed[0], available=True, fell_back=True)
        return result

    result["error"] = "No Ollama models are installed. Run `ollama pull gemma3:4b`."
    return result


def _payload(
    messages: List[Dict[str, str]],
    model: str,
    temperature: Optional[float],
    num_predict: Optional[int],
    stream: bool,
    keep_alive: Optional[str],
    num_ctx: Optional[int],
) -> Dict[str, Any]:
    return {
        "model": model,
        "stream": stream,
        "messages": messages,
        "keep_alive": keep_alive or OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": OLLAMA_TEMPERATURE if temperature is None else temperature,
            "num_predict": OLLAMA_NUM_PREDICT if num_predict is None else num_predict,
            "num_ctx": OLLAMA_NUM_CTX if num_ctx is None else num_ctx,
        },
    }


def chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    num_predict: Optional[int] = None,
    timeout: Optional[int] = None,
    base_url: Optional[str] = None,
    keep_alive: Optional[str] = None,
    num_ctx: Optional[int] = None,
) -> Dict[str, Any]:
    target_model = model or OLLAMA_MODEL
    url = (base_url or OLLAMA_BASE_URL).rstrip("/")
    timeout_sec = timeout or OLLAMA_TIMEOUT
    result: Dict[str, Any] = {
        "answer": "",
        "model": target_model,
        "latency_ms": 0.0,
        "error": None,
        "status": "error",
        "prompt_eval_count": None,
        "eval_count": None,
    }

    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{url}/api/chat",
            json=_payload(
                messages, target_model, temperature, num_predict, False, keep_alive, num_ctx
            ),
            timeout=timeout_sec,
        )
        result["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        if resp.status_code == 200:
            data = resp.json()
            content = (data.get("message") or {}).get("content", "").strip()
            result["answer"] = content
            result["status"] = "success" if content else "error"
            result["prompt_eval_count"] = data.get("prompt_eval_count")
            result["eval_count"] = data.get("eval_count")
            if not content:
                result["error"] = "Ollama returned an empty response."
        elif resp.status_code == 404:
            result["error"] = (
                f"Model '{target_model}' not found in Ollama. "
                f"Pull it via `ollama pull {target_model}`."
            )
        else:
            result["error"] = f"Ollama error {resp.status_code}: {resp.text[:200]}"
    except requests.exceptions.ConnectionError:
        result["error"] = "Ollama connection refused. Start the daemon via `ollama serve`."
    except requests.exceptions.Timeout:
        result["error"] = f"Ollama generation timed out after {timeout_sec}s."
    except Exception as exc:  # pragma: no cover - defensive
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def stream_chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    num_predict: Optional[int] = None,
    timeout: Optional[int] = None,
    base_url: Optional[str] = None,
    keep_alive: Optional[str] = None,
    num_ctx: Optional[int] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Iterable[str]:
    """Yield text chunks; populate ``meta`` with final status and latency."""
    target_model = model or OLLAMA_MODEL
    url = (base_url or OLLAMA_BASE_URL).rstrip("/")
    timeout_sec = timeout or OLLAMA_TIMEOUT
    holder = meta if meta is not None else {}
    holder.update(
        {"answer": "", "model": target_model, "error": None, "status": "error", "latency_ms": 0.0}
    )
    chunks: List[str] = []
    t0 = time.perf_counter()

    try:
        with requests.post(
            f"{url}/api/chat",
            json=_payload(
                messages, target_model, temperature, num_predict, True, keep_alive, num_ctx
            ),
            timeout=(10, timeout_sec),
            stream=True,
        ) as resp:
            if resp.status_code != 200:
                holder["error"] = f"Ollama error {resp.status_code}: {resp.text[:200]}"
                return
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if data.get("error"):
                    holder["error"] = str(data["error"])
                    break
                piece = (data.get("message") or {}).get("content", "")
                if piece:
                    chunks.append(piece)
                    yield piece
                if data.get("done"):
                    holder["prompt_eval_count"] = data.get("prompt_eval_count")
                    holder["eval_count"] = data.get("eval_count")
                    break
    except requests.exceptions.ConnectionError:
        holder["error"] = "Ollama connection refused. Start the daemon via `ollama serve`."
    except requests.exceptions.Timeout:
        holder["error"] = f"Ollama streaming timed out after {timeout_sec}s."
    except Exception as exc:  # pragma: no cover - defensive
        holder["error"] = f"{type(exc).__name__}: {exc}"

    holder["answer"] = "".join(chunks).strip()
    holder["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
    holder["status"] = "success" if holder["answer"] and not holder["error"] else "error"


def embed_texts(
    texts: List[str],
    model: str = EMBEDDING_MODEL,
    base_url: Optional[str] = None,
    timeout: Optional[int] = None,
    keep_alive: str = "0s",
) -> Dict[str, Any]:
    """Compute multilingual embeddings through Ollama's /api/embed endpoint."""
    url = (base_url or OLLAMA_BASE_URL).rstrip("/")
    result: Dict[str, Any] = {
        "embeddings": [],
        "model": model,
        "error": None,
        "status": "error",
        "latency_ms": 0.0,
    }
    if not texts:
        result["status"] = "success"
        return result

    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{url}/api/embed",
            json={"model": model, "input": list(texts), "truncate": True, "keep_alive": keep_alive},
            timeout=timeout or EMBEDDING_TIMEOUT,
        )
        result["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        if resp.status_code == 200:
            data = resp.json()
            vectors = data.get("embeddings") or []
            if not vectors and data.get("embedding"):
                vectors = [data["embedding"]]
            if len(vectors) == len(texts):
                result["embeddings"] = vectors
                result["status"] = "success"
            else:
                result["error"] = "Embedding count did not match input count."
        elif resp.status_code == 404:
            result["error"] = f"Embedding model '{model}' not found. Run `ollama pull {model}`."
        else:
            result["error"] = f"Ollama embed error {resp.status_code}: {resp.text[:200]}"
    except requests.exceptions.ConnectionError:
        result["error"] = "Ollama connection refused while computing embeddings."
    except requests.exceptions.Timeout:
        result["error"] = "Embedding request timed out."
    except Exception as exc:  # pragma: no cover - defensive
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def embedding_model_available(model: str = EMBEDDING_MODEL, base_url: Optional[str] = None) -> bool:
    status = check_ollama_status(base_url=base_url)
    if not status.get("available"):
        return False
    return any(model_matches(name, model) for name in status.get("models", []))


def unload_model(model: str, base_url: Optional[str] = None) -> bool:
    """Ask Ollama to unload a model so the local translator can use the RAM."""
    try:
        proc = subprocess.run(
            ["ollama", "stop", model], capture_output=True, text=True, timeout=20
        )
        if proc.returncode == 0:
            return True
    except Exception:
        pass

    url = (base_url or OLLAMA_BASE_URL).rstrip("/")
    try:
        resp = requests.post(
            f"{url}/api/generate",
            json={"model": model, "prompt": "", "keep_alive": 0, "stream": False},
            timeout=20,
        )
        return resp.status_code == 200
    except Exception:
        return False


class LLMService:
    """Thin object wrapper around the Ollama client for dependency injection."""

    def __init__(self, model: Optional[str] = None) -> None:
        self.preferred_model = model or OLLAMA_MODEL
        self.resolved_model = self.preferred_model
        self.last_error: Optional[str] = None
        self._status: Optional[Dict[str, Any]] = None

    def refresh(self) -> Dict[str, Any]:
        self._status = check_ollama_status()
        resolved = resolve_model(self.preferred_model, status=self._status)
        self.resolved_model = resolved["model"]
        self.last_error = resolved.get("error")
        resolved["embedding_model_available"] = any(
            model_matches(name, EMBEDDING_MODEL) for name in self._status.get("models", [])
        )
        return resolved

    def available(self) -> bool:
        if self._status is None:
            self.refresh()
        return bool(self._status and self._status.get("available") and self.resolved_model)

    def chat(self, messages, **kwargs) -> Dict[str, Any]:
        return chat(messages, model=self.resolved_model, **kwargs)

    def stream_chat(self, messages, meta=None, **kwargs) -> Iterable[str]:
        return stream_chat(messages, model=self.resolved_model, meta=meta, **kwargs)

    def unload(self) -> bool:
        return unload_model(self.resolved_model)
