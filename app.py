"""BhashaCore Streamlit application.

A conversational multilingual assistant supporting English, Hindi, Bengali,
Punjabi, Assamese, and Nepali.  The UI exposes general conversation, curated
FAQ lookup, document-grounded QA with citations, translation, summarization,
session memory, and honest diagnostics.

The deterministic FAQ helpers remain importable from this module for backward
compatibility with the original test suite.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from chatbot import Chatbot, ChatResponse
from config import (
    DEFAULT_FAQ_PATH,
    FAQ_AMBIGUITY_MARGIN,
    FAQ_MATCH_THRESHOLD,
    LANGUAGE_CODES,
    SUPPORTED_LANGUAGES,
)
from faq_engine import (
    DEFAULT_AMBIGUITY_MARGIN,
    DEFAULT_MATCH_THRESHOLD,
    LANGUAGE_LABELS,
    find_faq,
    format_fallback_message,
    get_answer,
    load_faqs,
    normalize,
    similarity,
    tokens,
)
from knowledge_base import KnowledgeBase
from language_utils import language_name
from router import ROUTE_LABELS

# Backward-compatible module exports
APP_DIR = Path(__file__).resolve().parent
FAQ_PATH = DEFAULT_FAQ_PATH
MATCH_THRESHOLD = DEFAULT_MATCH_THRESHOLD

__all__ = [
    "APP_DIR",
    "FAQ_PATH",
    "LANGUAGE_LABELS",
    "LANGUAGE_CODES",
    "MATCH_THRESHOLD",
    "load_faqs",
    "normalize",
    "tokens",
    "similarity",
    "find_faq",
    "get_answer",
    "format_fallback_message",
    "main",
]

AUTO_LANGUAGE = "Auto-detect"
LANGUAGE_OPTIONS = [AUTO_LANGUAGE] + list(SUPPORTED_LANGUAGES)

SOURCE_LABELS = {
    "general": "General knowledge · model-generated, not source-verified",
    "document": "Document-grounded · answers restricted to retrieved passages",
    "faq": "Curated FAQ · demo knowledge base",
    "translation": "Machine translation · generated text",
    "summary": "Model summary · verify important facts",
    "clarification": "Clarification requested",
    "error": "Error",
}

EXAMPLE_PROMPTS = [
    ("General & science", "Explain photosynthesis in simple English."),
    ("General & science", "प्रकाश संश्लेषण सरल हिंदी में समझाइए।"),
    ("Programming", "What is a Python function? Give a short example."),
    ("Programming", "SQL में duplicate rows कैसे find करें?"),
    ("Translation", "Translate to Hindi: The district office is closed on Sunday."),
    ("Translation", "Translate to Assamese: Please bring your income certificate."),
    ("Government demo", "What documents are required for the scholarship?"),
    ("Government demo", "छात्रवृत्ति के लिए आवेदन कैसे करें?"),
]


@st.cache_resource
def get_initialized_knowledge_base() -> KnowledgeBase:
    """Load curated FAQs and the optional AI4Bharat sample records once."""
    kb = KnowledgeBase()
    kb.load_faq_file(FAQ_PATH)
    rag_file = APP_DIR / "data" / "rag_sample.json"
    if rag_file.exists():
        try:
            with rag_file.open("r", encoding="utf-8") as handle:
                entries = json.load(handle).get("entries", [])
                kb.load_rag_entries(entries)
        except Exception:
            pass
    return kb


def _get_bot() -> Chatbot:
    if "bot" not in st.session_state:
        st.session_state.bot = Chatbot(kb=get_initialized_knowledge_base())
        st.session_state.messages = []
        st.session_state.upload_signature = ()
        st.session_state.upload_results = []
    return st.session_state.bot


def _language_value(option: str) -> str:
    return "Auto" if option == AUTO_LANGUAGE else option


def _append_user(prompt: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})


def _append_response(response: ChatResponse) -> None:
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.answer,
            "route": response.route,
            "source_type": response.source_type,
            "target_lang": response.target_lang,
            "detected_lang": response.detected_lang,
            "detection_confidence": response.detection_confidence,
            "model": response.model,
            "latency_ms": response.latency_ms,
            "citations": response.citations,
            "notes": response.notes,
            "error": response.error,
            "evidence_found": response.evidence_found,
            "retrieval_method": response.retrieval_method,
            "backend": response.backend,
            "translated": response.translated,
        }
    )


def _render_meta(message: Dict[str, Any]) -> None:
    route = message.get("route", "")
    target_lang = message.get("target_lang")
    details: List[str] = []
    if route:
        details.append(ROUTE_LABELS.get(route, route))
    if target_lang:
        details.append(f"answer language: {language_name(target_lang)}")
    if message.get("model"):
        details.append(f"model: {message['model']}")
    if message.get("latency_ms"):
        details.append(f"latency: {message['latency_ms'] / 1000.0:.1f}s")
    if message.get("retrieval_method"):
        details.append(f"retrieval: {message['retrieval_method']}")
    if message.get("backend"):
        details.append(f"translation: {message['backend']}")
    if details:
        st.caption(" · ".join(details))

    if message.get("source_type") in SOURCE_LABELS:
        st.caption(f"**Source:** {SOURCE_LABELS[message['source_type']]}")
    if message.get("evidence_found") is False:
        st.caption("No supporting evidence was found in the uploaded documents.")
    if (
        message.get("detected_lang")
        and message.get("detection_confidence", 1.0) < 0.6
        and message.get("target_lang") == message.get("detected_lang")
    ):
        st.caption(
            "Automatic language detection was uncertain for this input. "
            "Select the language manually if the answer looks wrong."
        )

    citations = message.get("citations") or []
    if citations:
        cited_count = sum(1 for citation in citations if citation.cited)
        if cited_count:
            title = f"Sources ({cited_count})"
            if cited_count < len(citations):
                title += f" · {len(citations) - cited_count} retrieved but not cited"
        else:
            title = f"Retrieved passages ({len(citations)}, not cited by the model)"
        with st.expander(title, expanded=True):
            for citation in citations:
                marker = "" if citation.cited else " _(retrieved, not cited)_"
                st.markdown(f"**{citation.label()}**{marker}")
                if citation.excerpt:
                    st.caption(citation.excerpt)

    notes = message.get("notes") or []
    if notes:
        with st.expander("Notes & limitations"):
            for note in notes:
                st.markdown(f"- {note}")

    with st.expander("Copy answer"):
        st.code(message.get("content", ""), language="text", wrap_lines=True)


def _run_turn(bot: Chatbot, prompt: str, language_option: str) -> None:
    _append_user(prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        holder: Dict[str, Any] = {}
        pieces: List[str] = []
        try:
            for piece in bot.respond_stream(
                prompt,
                selected_language=_language_value(language_option),
                holder=holder,
            ):
                pieces.append(piece)
                placeholder.markdown("".join(pieces))
        except Exception as exc:  # pragma: no cover - UI safety net
            placeholder.error(f"Unexpected error: {type(exc).__name__}: {exc}")
            _append_response(
                ChatResponse(
                    answer=f"Unexpected error: {type(exc).__name__}: {exc}",
                    route="error",
                    target_lang="en",
                    detected_lang="en",
                    detection_confidence=0.0,
                    source_type="error",
                    error=str(exc),
                )
            )
            return

        response: Optional[ChatResponse] = holder.get("response")
        if response is None:
            placeholder.error("The assistant produced no response.")
            return

        placeholder.markdown(response.answer)
        _render_meta(
            {
                "content": response.answer,
                "route": response.route,
                "source_type": response.source_type,
                "target_lang": response.target_lang,
                "detected_lang": response.detected_lang,
                "detection_confidence": response.detection_confidence,
                "model": response.model,
                "latency_ms": response.latency_ms,
                "citations": response.citations,
                "notes": response.notes,
                "evidence_found": response.evidence_found,
                "retrieval_method": response.retrieval_method,
                "backend": response.backend,
            }
        )
        _append_response(response)


def _render_sidebar(bot: Chatbot) -> None:
    st.sidebar.title("BhashaCore")
    st.sidebar.caption("Six-language conversational assistant with grounded document QA.")

    st.sidebar.subheader("Language")
    st.sidebar.caption(
        "Auto-detect uses Unicode script plus function-word evidence. "
        "Short Assamese/Bengali or Hindi/Nepali text can be ambiguous."
    )

    st.sidebar.divider()
    st.sidebar.subheader("Documents")
    uploads = st.sidebar.file_uploader(
        "Upload PDF, TXT, MD, JSON, or CSV",
        type=["pdf", "txt", "md", "markdown", "json", "csv"],
        accept_multiple_files=True,
        help="Text is chunked and indexed locally. Scanned/image-only PDFs are not supported.",
    )
    signature = tuple(
        (upload.name, len(upload.getvalue()), hashlib.sha1(upload.getvalue()).hexdigest()[:12])
        for upload in (uploads or [])
    )
    current_signature = tuple(st.session_state.get("upload_signature", ()))
    new_files = [
        upload
        for upload in (uploads or [])
        if (upload.name, len(upload.getvalue()), hashlib.sha1(upload.getvalue()).hexdigest()[:12])
        not in current_signature
    ]
    if new_files:
        results = []
        for upload in new_files:
            results.append((upload.name, bot.add_document(upload.name, upload.getvalue())))
        st.session_state.upload_signature = signature
        st.session_state.upload_results = results
        st.rerun()

    results = st.session_state.get("upload_results") or []
    for name, result in results:
        if result.error:
            st.sidebar.error(f"{name}: {result.error}")
        else:
            warnings = " ".join(result.warnings or [])
            text = f"{name}: {len(result.chunks)} chunk(s) indexed."
            if warnings:
                text += f" {warnings}"
            st.sidebar.success(text)

    stats = bot.documents.stats()
    st.sidebar.caption(
        f"{stats['documents']} document(s), {stats['chunks']} chunk(s), "
        f"languages: {', '.join(stats['languages']) or 'none'}"
    )
    if st.sidebar.button("Clear documents", use_container_width=True):
        bot.clear_documents()
        st.session_state.upload_results = []
        st.session_state.upload_signature = ()
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("System diagnostics")
    if st.sidebar.button("Refresh diagnostics", use_container_width=True):
        bot.refresh_status()
        st.rerun()

    status = bot.status()
    if status.get("ollama_available"):
        st.sidebar.success(f"Ollama online · {status.get('active_model')}")
    else:
        st.sidebar.warning("Ollama offline · general answers unavailable")
        if status.get("ollama_error"):
            st.sidebar.caption(status["ollama_error"])
    st.sidebar.caption(
        f"Embedding model: {status.get('embedding_model')} "
        f"({'available' if status.get('embedding_available') else 'unavailable'})"
    )
    st.sidebar.caption(
        f"Translator: {status.get('translator_backend')} "
        f"({'installed' if status.get('translator_installed') else 'not installed'})"
    )
    st.sidebar.caption(
        f"Knowledge base entries: {len(bot.kb.get_all_entries())} · "
        f"session messages: {len(bot.history)}"
    )

    st.sidebar.divider()
    st.sidebar.caption(
        "NLLB-200 is CC-BY-NC-4.0 (non-commercial). "
        "Scholarship records are fictional demo fixtures."
    )


def _render_examples(bot: Chatbot) -> None:
    with st.expander("Example prompts", expanded=not st.session_state.messages):
        st.caption(
            "Questions work in any supported language. "
            "Use the selector above for a fixed answer language."
        )
        columns = st.columns(2)
        for index, (category, prompt) in enumerate(EXAMPLE_PROMPTS):
            column = columns[index % 2]
            if column.button(
                f"{category}: {prompt}",
                key=f"example_{index}",
                use_container_width=True,
            ):
                st.session_state.pending_prompt = prompt
                st.rerun()
        if bot.documents.documents:
            document_prompts = [
                "Summarize the uploaded document.",
                "According to the document, what are the key requirements?",
            ]
            for offset, prompt in enumerate(document_prompts):
                if st.button(prompt, key=f"doc_example_{offset}", use_container_width=True):
                    st.session_state.pending_prompt = prompt
                    st.rerun()


def main() -> None:
    """Run the BhashaCore Streamlit chat application."""
    st.set_page_config(
        page_title="BhashaCore · Multilingual AI Assistant",
        page_icon="🌐",
        layout="centered",
    )

    bot = _get_bot()

    language_option = st.selectbox(
        "Answer language",
        options=LANGUAGE_OPTIONS,
        index=0,
        help=(
            "Auto-detect follows the language of your message. "
            "Select a language to force the answer language."
        ),
    )
    _render_sidebar(bot)

    st.title("🌐 BhashaCore")
    st.caption(
        "Conversational assistant for English, Hindi, Bengali, Punjabi, Assamese, and Nepali. "
        "Answers distinguish general model knowledge, curated FAQ data, and document-grounded evidence."
    )
    st.info(
        "Scholarship FAQ content is fictional demo data. Machine translations are unverified. "
        "Always confirm deadlines, fees, eligibility, and legal requirements against an official source."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                _render_meta(message)

    pending = st.session_state.pop("pending_prompt", None)
    if pending:
        _run_turn(bot, pending, language_option)

    _render_examples(bot)

    question = st.chat_input("Ask anything, upload a document, or request a translation...")
    if question:
        _run_turn(bot, question, language_option)

    action_columns = st.columns(2)
    with action_columns[0]:
        if st.button("Clear conversation", use_container_width=True):
            bot.clear_history()
            st.session_state.messages = []
            st.rerun()
    with action_columns[1]:
        st.caption(
            "Session memory only. Previous assistant statements are not treated as verified evidence."
        )


if __name__ == "__main__":
    main()
