"""BhashaCore Streamlit Web Application.

A lightweight, reliable multilingual FAQ chatbot for public-service scenarios,
supporting English, Hindi, Bengali, Punjabi, Assamese, and Nepali.

Integrates:
- Fast deterministic FAQ retrieval via KnowledgeBase & faq_engine
- Optional LLM grounding via Ollama (qwen2.5:3b / gemma3:4b) with anti-hallucination guardrails
- Full provenance attribution, confidence badges, and draft translation disclaimers
- Diagnostics panel for runtime model and knowledge-base status
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

from config import (
    DEFAULT_FAQ_PATH,
    FAQ_AMBIGUITY_MARGIN,
    FAQ_MATCH_THRESHOLD,
    LANGUAGE_CODES,
    SUPPORTED_LANGUAGES,
    get_config_summary,
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
from ollama_client import check_ollama_status, generate_response

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


@st.cache_resource
def get_initialized_knowledge_base() -> KnowledgeBase:
    """Initialize and cache KnowledgeBase with curated FAQs and available samples."""
    kb = KnowledgeBase()
    kb.load_faq_file(FAQ_PATH)

    # Ingest RAG sample if present in data/
    rag_file = APP_DIR / "data" / "rag_sample.json"
    if rag_file.exists():
        try:
            with rag_file.open("r", encoding="utf-8") as f:
                content = json.load(f)
                entries = content.get("entries", [])
                kb.load_rag_entries(entries)
        except Exception:
            pass

    return kb


def render_sidebar(kb: KnowledgeBase, ollama_status: Dict[str, Any]) -> None:
    """Render system diagnostics, model availability, and knowledge stats in the sidebar."""
    st.sidebar.title("⚙️ System Status")

    # Knowledge Base Overview
    st.sidebar.subheader("Knowledge Layer")
    stats = kb.get_stats()
    st.sidebar.markdown(f"**Total QA Records:** `{stats['total_entries']}`")
    st.sidebar.markdown(f"**Domains Active:** `{', '.join(kb.get_domains())}`")
    st.sidebar.markdown(f"**Languages Loaded:** `{len(stats['supported_languages'])}`")

    st.sidebar.divider()

    # Ollama / LLM Status
    st.sidebar.subheader("Ollama LLM Engine")
    cfg = get_config_summary()
    st.sidebar.markdown(f"**Configured Model:** `{cfg['configured_model']}`")

    if ollama_status["available"]:
        st.sidebar.success("🟢 Ollama Daemon: Online")
        models = ollama_status.get("models", [])
        if models:
            st.sidebar.caption(f"Installed Models: {', '.join(models)}")
    else:
        st.sidebar.warning("🟡 Ollama Daemon: Offline / Sandboxed")
        st.sidebar.caption("FAQ retrieval operates independently without requiring an LLM.")

    st.sidebar.divider()
    st.sidebar.caption("BhashaCore v2.0 · SIH Multilingual AI Assistant")


def main() -> None:
    """Run the BhashaCore Streamlit interactive web application."""
    st.set_page_config(
        page_title="BhashaCore · Multilingual FAQ Chatbot",
        page_icon="🌐",
        layout="centered",
    )

    kb = get_initialized_knowledge_base()
    ollama_status = check_ollama_status()

    # Render system sidebar
    render_sidebar(kb, ollama_status)

    st.title("🌐 BhashaCore")
    st.caption(
        "Multilingual Public-Service FAQ Chatbot · Supporting English, "
        "Hindi, Bengali, Punjabi, Assamese, and Nepali"
    )

    st.warning(
        "⚠️ **Prototype Notice & Disclaimer:** This application includes **fictional scholarship test data** "
        "for system benchmarking. Non-English translations are **unverified drafts** pending native-speaker "
        "validation. Never present or rely on these responses as official government policy."
    )

    # Response Language Selector
    language_name = st.selectbox(
        "Choose your response language:",
        options=list(SUPPORTED_LANGUAGES.keys()),
        index=0,
        help="Select the language in which you want to receive answers.",
    )
    lang_code = SUPPORTED_LANGUAGES[language_name]

    # Session State Management
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "active_language" not in st.session_state:
        st.session_state.active_language = language_name

    # Reset chat history if user switches response language
    if language_name != st.session_state.active_language:
        st.session_state.messages = []
        st.session_state.active_language = language_name

    # Render Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "provenance" in message:
                prov = message["provenance"]
                st.caption(prov)

    # User Chat Input
    question = st.chat_input("Ask a public-service or scholarship question...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching multilingual knowledge base..."):
                entry, score, prov = kb.search(question)

                provenance_caption = ""
                if entry is not None:
                    # Verified stored answer in knowledge base
                    topic_id = entry.topic
                    domain = entry.domain
                    raw_answer = entry.answer.get(lang_code)

                    if not raw_answer:
                        answer = format_fallback_message("missing_translation")
                        provenance_caption = f"Topic `{topic_id}` · Domain: `{domain}` (translation pending in {language_name})"
                    else:
                        answer = raw_answer
                        conf_badge = "🟢 High" if score >= 0.70 else ("🟡 Moderate" if score >= 0.50 else "🟠 Threshold")
                        fictional_badge = " | 🧪 Fictional Test Fixture" if entry.fictional else " | 🏛️ Public Service Data"
                        status_badge = " | ⚠️ Draft translation" if entry.translation_status != "verified" else " | ✅ Verified"

                        provenance_caption = (
                            f"**Source Attribution:** Topic `{topic_id}` ({domain}) · "
                            f"Confidence: `{score:.2f}` ({conf_badge}){fictional_badge}{status_badge}"
                        )
                else:
                    # Fallback path: Check if Ollama is available to provide grounded answer,
                    # otherwise return strict anti-hallucination fallback
                    if ollama_status["available"]:
                        context = kb.compile_grounding_context()
                        llm_res = generate_response(
                            question=question,
                            context=context,
                            language=language_name,
                        )
                        if llm_res["status"] == "success" and llm_res["answer"]:
                            answer = llm_res["answer"]
                            provenance_caption = (
                                f"🤖 **AI-Generated Answer** (Model: `{llm_res['model']}`) · "
                                f"Latency: `{llm_res['latency_ms']}ms` · Context Grounded"
                            )
                        else:
                            answer = format_fallback_message("no_match")
                            provenance_caption = f"No match (Best similarity: `{score:.2f}`) · Ollama generation unassisted"
                    else:
                        answer = format_fallback_message("no_match")
                        provenance_caption = f"No confident match (Best similarity: `{score:.2f}`) · Anti-hallucination gate active"

                st.markdown(answer)
                if provenance_caption:
                    st.caption(provenance_caption)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "provenance": provenance_caption,
            }
        )

    # Interactive Action Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("Show sample questions", use_container_width=True):
            st.session_state.show_samples = not st.session_state.get("show_samples", False)

    # Sample Questions Drawer
    if st.session_state.get("show_samples", False):
        st.subheader(f"Sample Questions ({language_name})")
        for ent in kb.get_all_entries():
            examples = ent.question_examples.get(lang_code, [])
            if examples:
                st.markdown(f"- **[{ent.domain}]** {examples[0]}")

    # Architecture Explanation Expander
    with st.expander("ℹ️ How BhashaCore v2.0 retrieval works"):
        st.markdown(
            """
            - **Extensible Knowledge Base:** Ingests domain-specific FAQs and public service QA records.
            - **Unicode-Preserving Normalization:** Retains Indic script letters, combining marks
              (matras, viramas), and numbers across Devanagari, Bengali, Gurmukhi, and Latin.
            - **Intent-Driven Scoring:** Evaluates content-token overlap and soft stem matching.
            - **Confidence Gating & Provenance:** Clearly indicates source, test data status,
              and translation verification level.
            - **Hybrid Architecture:** Deterministic zero-overhead FAQ search with optional
              context-grounded Ollama LLM answering when enabled.
            """
        )


if __name__ == "__main__":
    main()
