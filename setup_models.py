"""Check or download BhashaCore's optional local models.

Nothing is downloaded unless you pass --download.  The chatbot works without
the optional stack (FAQ + English-only answers), but six-language generation
needs an Ollama chat model and multilingual embeddings, and translation-assisted
languages (pa/as/ne) need NLLB-200.

Models and licences:
  gemma3:4b                     Ollama library (Gemma Terms of Use)   ~3.4 GB
  qwen2.5:3b                    Apache-2.0                           ~1.9 GB
  bge-m3                        MIT (via Ollama)                     ~1.2 GB
  facebook/nllb-200-distilled-600M  CC-BY-NC-4.0 (non-commercial)    ~2.4 GB
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from config import EMBEDDING_MODEL, MODELS_DIR, NLLB_MODEL_DIR, OLLAMA_BASE_URL, OLLAMA_MODEL

NLLB_REPO = "facebook/nllb-200-distilled-600M"
NLLB_FILES = [
    "config.json",
    "generation_config.json",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors",
]


def ollama_models() -> set:
    import requests

    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - report, do not crash
        print(f"Ollama unreachable at {OLLAMA_BASE_URL}: {exc}")
        return set()
    return {model["name"] for model in response.json().get("models", [])}


def nllb_installed() -> bool:
    return (NLLB_MODEL_DIR / "config.json").exists() and (
        (NLLB_MODEL_DIR / "model.safetensors").exists()
        or (NLLB_MODEL_DIR / "pytorch_model.bin").exists()
    )


def report() -> None:
    installed = ollama_models()
    print(f"Ollama at {OLLAMA_BASE_URL}")
    for model, licence in (
        (OLLAMA_MODEL, "configured chat model"),
        ("qwen2.5:3b", "fallback chat model"),
        (EMBEDDING_MODEL, "multilingual embeddings"),
    ):
        present = any(
            name == model or name.startswith(model + ":") for name in installed
        )
        print(f"  {'OK ' if present else 'MISSING'} {model:<16} {licence}")
    state = "OK " if nllb_installed() else "MISSING"
    print(f"  {state} NLLB-200          {NLLB_MODEL_DIR} (CC-BY-NC-4.0, non-commercial)")


def pull(model: str) -> None:
    if shutil.which("ollama") is None:
        raise SystemExit("The `ollama` CLI is not on PATH; install Ollama first.")
    print(f"Pulling {model} ...")
    subprocess.run(["ollama", "pull", model], check=True)


def download_nllb() -> None:
    from huggingface_hub import snapshot_download

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {NLLB_REPO} to {NLLB_MODEL_DIR} (~2.4 GB) ...")
    snapshot_download(
        repo_id=NLLB_REPO,
        local_dir=str(NLLB_MODEL_DIR),
        allow_patterns=NLLB_FILES,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "download",
        nargs="?",
        choices=["all", "llm", "embed", "nllb"],
        help="what to download; omit to only check what is installed",
    )
    args = parser.parse_args()

    if args.download is None:
        report()
        print("\nRun `python setup_models.py download all` to install everything.")
        return 0

    if args.download in ("all", "llm"):
        pull(OLLAMA_MODEL)
        pull("qwen2.5:3b")
    if args.download in ("all", "embed"):
        pull(EMBEDDING_MODEL)
    if args.download in ("all", "nllb") and not nllb_installed():
        download_nllb()

    print()
    report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
