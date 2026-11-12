"""BhashaCore Ollama Client Module.

Provides robust, fault-tolerant interactions with local Ollama instances.
Gracefully handles offline daemons, missing models, connection timeouts,
and non-responsive hosts without crashing the main application.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import requests

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_NUM_PREDICT,
    OLLAMA_TEMPERATURE,
    OLLAMA_TIMEOUT,
)

logger = logging.getLogger("bhashacore.ollama")

SYSTEM_PROMPT = """You are BhashaCore, an accurate and transparent multilingual public-service assistant.
Follow these rules strictly:
1. Ground your answer ONLY in the provided verified context.
2. If the context does not contain the answer, explicitly state that the information is unavailable or unspecified.
3. NEVER invent dates, deadlines, monetary amounts, eligibility criteria, links, or contact numbers.
4. Answer strictly in the requested language ({language}).
5. Note: Test scholarship data is fictional and for evaluation purposes only.
"""


def check_ollama_status(base_url: Optional[str] = None) -> Dict[str, Any]:
    """Inspect local Ollama daemon status and return installed models."""
    url = (base_url or OLLAMA_BASE_URL).rstrip("/")
    result: Dict[str, Any] = {
        "available": False,
        "url": url,
        "models": [],
        "error": None,
    }
    try:
        response = requests.get(f"{url}/api/tags", timeout=3)
        if response.status_code == 200:
            data = response.json()
            models_list = [m.get("name", "") for m in data.get("models", [])]
            result["available"] = True
            result["models"] = models_list
        else:
            result["error"] = f"Unexpected HTTP status {response.status_code}"
    except requests.exceptions.ConnectionError:
        result["error"] = "Could not connect to Ollama daemon (is `ollama serve` running?)"
    except requests.exceptions.Timeout:
        result["error"] = "Connection to Ollama daemon timed out."
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def check_model_available(model_name: str, base_url: Optional[str] = None) -> bool:
    """Check if a specific model tag is currently downloaded in Ollama."""
    status = check_ollama_status(base_url=base_url)
    if not status["available"]:
        return False
    models = status["models"]
    # Check exact match or base name without :tag
    return any(m == model_name or m.startswith(f"{model_name}:") for m in models)


def generate_response(
    question: str,
    context: str,
    language: str = "English",
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: Optional[int] = None,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """Generate grounded answer via Ollama chat API.
    
    Returns a dictionary with keys:
        - answer: str
        - model: str
        - latency_ms: float
        - error: Optional[str]
        - status: "success" | "error"
    """
    target_model = model or OLLAMA_MODEL
    url = (base_url or OLLAMA_BASE_URL).rstrip("/")
    timeout_sec = timeout or OLLAMA_TIMEOUT
    temp = temperature if temperature is not None else OLLAMA_TEMPERATURE

    result: Dict[str, Any] = {
        "answer": "",
        "model": target_model,
        "latency_ms": 0.0,
        "error": None,
        "status": "error",
    }

    formatted_system = SYSTEM_PROMPT.format(language=language)
    user_prompt = (
        f"Available Context:\n{context}\n\n"
        f"User Question: {question}\n\n"
        f"Provide a concise, factual answer in {language} based solely on the context."
    )

    payload = {
        "model": target_model,
        "stream": False,
        "messages": [
            {"role": "system", "content": formatted_system},
            {"role": "user", "content": user_prompt},
        ],
        "options": {
            "temperature": temp,
            "num_predict": OLLAMA_NUM_PREDICT,
        },
    }

    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{url}/api/chat",
            json=payload,
            timeout=timeout_sec,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        result["latency_ms"] = round(elapsed_ms, 2)

        if resp.status_code == 200:
            data = resp.json()
            content = data.get("message", {}).get("content", "").strip()
            result["answer"] = content
            result["status"] = "success"
        elif resp.status_code == 404:
            result["error"] = f"Model '{target_model}' not found in Ollama. Pull it via `ollama pull {target_model}`."
        else:
            result["error"] = f"Ollama error {resp.status_code}: {resp.text[:200]}"
    except requests.exceptions.ConnectionError:
        result["error"] = "Ollama connection refused. Start daemon via `ollama serve`."
    except requests.exceptions.Timeout:
        result["error"] = f"Ollama generation timed out after {timeout_sec}s."
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result
