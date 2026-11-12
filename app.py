"""BhashaCore Streamlit Web Application.

A lightweight, reliable multilingual FAQ chatbot for public-service scenarios,
supporting English, Hindi, Bengali, Punjabi, Assamese, and Nepali.

Powered by `faq_engine.py` using pure Python standard library for deterministic,
zero-hallucination FAQ retrieval from `faq_data.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import streamlit as st

from faq_engine import (
    DEFAULT_AMBIGUITY_MARGIN,
    DEFAULT_FAQ_PATH,
    DEFAULT_MATCH_THRESHOLD,
    LANGUAGE_CODES,
    LANGUAGE_LABELS,
    find_faq,
    format_fallback_message,
    get_answer,
    load_faqs,
    normalize,
    similarity,
    tokens,
)

# Preserve backward-compatible module attributes
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


def main() -> None:
    """Run the BhashaCore Streamlit interactive web application."""
    st.set_page_config(
        page_title="BhashaCore · Multilingual FAQ Chatbot",
        page_icon="🌐",
        layout="centered",
    )

    st.title("🌐 BhashaCore")
    st.caption(
        "Multilingual Public-Service FAQ Chatbot · Supporting English, "
        "Hindi, Bengali, Punjabi, Assamese, and Nepali"
    )

    st.warning(
        "**Prototype Notice & Disclaimer:** This application uses a **fictional scholarship** "
        "created strictly for technical testing. Non-English answers are **unverified draft "
        "translations** that have not yet been validated by native fluent speakers. "
        "Do not present or rely on these answers as real government guidance."
    )

    try:
        faq_data = load_faqs(FAQ_PATH)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        st.error(
            f"**Error loading FAQ knowledge base:** {exc}\n\n"
            f"Ensure `faq_data.json` exists in the application root directory."
        )
        st.stop()

    # Response Language Selector
    language_name = st.selectbox(
        "Choose your response language:",
        options=list(LANGUAGE_LABELS.keys()),
        index=0,
        help="Select the language in which you want to receive answers.",
    )
    lang_code = LANGUAGE_LABELS[language_name]

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

    # User Chat Input
    question = st.chat_input("Ask a question about the test scholarship...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the multilingual FAQ..."):
                faq, score = find_faq(question, faq_data, threshold=MATCH_THRESHOLD)

                if faq is None:
                    answer = format_fallback_message("no_match")
                    st.caption(f"No confident match (best similarity: {score:.2f}).")
                else:
                    topic_id = faq.get("topic", faq.get("id", "unknown"))
                    answer = get_answer(faq, lang_code)

                    if not answer:
                        answer = format_fallback_message("missing_translation")
                        st.caption(f"Matched topic: `{topic_id}` ({score:.2f})")
                    else:
                        st.caption(f"Matched FAQ topic: `{topic_id}` (confidence: {score:.2f})")
                        if faq.get("translation_status") != "verified":
                            st.caption(
                                "Translation status: **draft — unverified by native speakers**."
                            )

                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

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
        for faq in faq_data.get("faqs", []):
            examples = faq.get("question_examples", {}).get(lang_code, [])
            if examples:
                st.markdown(f"- {examples[0]}")

    # Architecture Explanation Expander
    with st.expander("ℹ️ How BhashaCore retrieval works"):
        st.markdown(
            """
            - **Multilingual Knowledge Base:** Loads curated FAQ topics from `faq_data.json`.
            - **Unicode-Preserving Normalization:** Retains Indic script letters, combining marks
              (matras, viramas), and numbers across Devanagari, Bengali, Gurmukhi, and Latin.
            - **Intent-Driven Scoring:** Matches queries across all 6 languages using content-token
              overlap and sequence similarity, preventing stopword false-positives.
            - **Safe Fallback & Anti-Hallucination:** If confidence is below the threshold or ambiguous,
              the system declines to answer rather than guessing or fabricating facts.
            - **Zero Heavy ML Dependencies:** Completely offline, deterministic, and instant.
            """
        )


if __name__ == "__main__":
    main()
