"""BhashaCore conversational orchestrator.

Combines general conversation, curated FAQ lookup, document-grounded retrieval,
translation, summarization, clarification, and session memory.  The orchestrator
keeps provenance explicit: general answers, FAQ answers, document answers, and
machine translations are reported as different source types.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from config import (
    EMBEDDING_MODEL,
    FAQ_DIRECT_MIN_SCORE,
    HISTORY_MAX_CHARS,
    HISTORY_MAX_MESSAGES,
    MAX_CONTEXT_CHARS,
    QUALITY_TRANSLATION_ASSIST_LANGS,
    SCRIPT_PURITY_MIN,
)
from documents import Citation, DocumentStore, RetrievalResult, RetrievedChunk
from faq_engine import content_tokens, format_fallback_message
from knowledge_base import KnowledgeBase, KnowledgeEntry
from language_utils import (
    LANGUAGE_NAMES,
    LanguageDetection,
    detect_language,
    language_code,
    language_name,
    script_purity,
)
from llm_client import LLMService
from prompts import (
    format_context,
    general_system,
    grounded_system,
    message,
    query_translation_system,
    summary_system,
    translation_system,
)
from router import (
    ROUTE_CLARIFY,
    ROUTE_DOCUMENT,
    ROUTE_FAQ,
    ROUTE_GENERAL,
    ROUTE_SUMMARIZE,
    ROUTE_TRANSLATION,
    RouteDecision,
    route as route_turn,
)
from translation_service import NLLBTranslationService, TranslationResult

_CITATION_RE = re.compile(r"\[(\d{1,2})\]")

_MISSING_EVIDENCE_PHRASES = (
    "couldn't find", "could not find", "cannot find", "can't find", "unable to find",
    "not found", "not mentioned", "does not mention", "doesn't mention",
    "do not mention", "don't mention", "does not contain", "doesn't contain",
    "does not specify", "doesn't specify", "not specified", "no mention",
    "no information", "no relevant information", "not provided", "not available",
    "जानकारी नहीं", "उपलब्ध नहीं", "उल्लेख नहीं", "नहीं मिला", "नहीं मिली", "नहीं मिल",
    "नहीं दिया", "नहीं दी", "नहीं बताया", "निर्दिष्ट नहीं",
    "পাইনি", "পাওয়া যায়নি", "উল্লেখ নেই", "তথ্য নেই", "উল্লেখ করা হয়নি", "দেওয়া হয়নি", "উল্লেখ করা হয় না",
    "পোৱা নাই", "উল্লেখ নাই", "তথ্য নাই", "দিয়া নাই", "উল্লেখ কৰা হোৱা নাই",
    "ਨਹੀਂ ਲੱਭ", "ਨਹੀਂ ਮਿਲ", "ਜਾਣਕਾਰੀ ਨਹੀਂ", "ਜ਼ਿਕਰ ਨਹੀਂ", "ਜਿਕਰ ਨਹੀਂ",
    "ਨਹੀਂ ਦਿੱਤਾ", "ਨਹੀਂ ਦਿੱਤੀ", "ਨਹੀਂ ਦੱਸਿਆ",
    "भेटिएन", "भेट्न सकिन", "जानकारी छैन", "उल्लेख छैन", "उपलब्ध छैन",
    "दिइएको छैन", "उल्लेख गरिएको छैन",
)


@dataclass
class ChatResponse:
    answer: str
    route: str
    target_lang: str
    detected_lang: str
    detection_confidence: float
    model: Optional[str] = None
    latency_ms: float = 0.0
    citations: List[Citation] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    error: Optional[str] = None
    source_type: str = "general"
    evidence_found: Optional[bool] = None
    retrieval_method: Optional[str] = None
    backend: Optional[str] = None
    translated: bool = False
    used_translation_assist: bool = False


@dataclass
class _PreparedTurn:
    message: str
    selected_language: str
    target: str
    detection: LanguageDetection
    decision: RouteDecision
    started: float
    faq: Optional[Tuple[KnowledgeEntry, float, Dict[str, Any]]] = None
    retrieval_query: Optional[str] = None
    retrieval: Optional[RetrievalResult] = None
    translation_payload: Optional[str] = None
    summary_source: Optional[str] = None
    summary_from_documents: bool = False
    assist: bool = False
    messages: List[Dict[str, str]] = field(default_factory=list)
    streamable: bool = False
    notes: List[str] = field(default_factory=list)


class Chatbot:
    """Stateful conversational agent for one browser session."""

    def __init__(
        self,
        kb: Optional[KnowledgeBase] = None,
        doc_store: Optional[DocumentStore] = None,
        llm: Optional[Any] = None,
        translator: Optional[Any] = None,
        model: Optional[str] = None,
        translation_assist_langs: Optional[Iterable[str]] = None,
    ) -> None:
        self.kb = kb if kb is not None else self._default_kb()
        self.documents = doc_store if doc_store is not None else DocumentStore()
        self.llm = llm if llm is not None else LLMService(model)
        self.translator = translator if translator is not None else NLLBTranslationService()
        self.translation_assist_langs = set(
            translation_assist_langs
            if translation_assist_langs is not None
            else QUALITY_TRANSLATION_ASSIST_LANGS
        )
        self.history: List[Dict[str, Any]] = []
        self.last_route: Optional[str] = None
        self._status: Optional[Dict[str, Any]] = None

    @staticmethod
    def _default_kb() -> KnowledgeBase:
        kb = KnowledgeBase()
        kb.load_faq_file()
        return kb

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def respond(
        self,
        message_text: str,
        selected_language: str = "Auto",
    ) -> ChatResponse:
        context = self._prepare(message_text, selected_language)
        response = self._execute(context)
        self._commit(context, response)
        return response

    def respond_stream(
        self,
        message_text: str,
        selected_language: str = "Auto",
        holder: Optional[Dict[str, Any]] = None,
    ) -> Iterator[str]:
        """Yield answer chunks live, then store the final ChatResponse in ``holder``."""
        try:
            context = self._prepare(message_text, selected_language)
        except Exception as exc:  # pragma: no cover - defensive
            response = ChatResponse(
                answer=message("clarification", "en"),
                route=ROUTE_CLARIFY,
                target_lang="en",
                detected_lang="en",
                detection_confidence=0.0,
                notes=[f"Internal error: {type(exc).__name__}: {exc}"],
                error=str(exc),
                source_type="error",
            )
            self.history.append({"role": "user", "content": message_text, "route": ROUTE_CLARIFY, "lang": "en"})
            self.history.append({"role": "assistant", "content": response.answer, "route": ROUTE_CLARIFY, "lang": "en"})
            if holder is not None:
                holder["response"] = response
            yield response.answer
            return

        response: Optional[ChatResponse] = None
        if context.streamable:
            meta: Dict[str, Any] = {}
            pieces: List[str] = []
            for piece in self.llm.stream_chat(context.messages, meta=meta):
                pieces.append(piece)
                yield piece
            answer = "".join(pieces).strip()
            if meta.get("status") == "success" and answer:
                response = self._general_response(
                    context,
                    answer,
                    model=meta.get("model"),
                    notes=self._general_notes(context),
                )
            else:
                error = meta.get("error") or "The model returned an empty response."
                if answer:
                    response = self._general_response(
                        context,
                        answer,
                        model=meta.get("model"),
                        notes=self._general_notes(context) + [f"Streaming ended early: {error}"],
                        error=error,
                    )
                else:
                    response = self._content_response(
                        context,
                        message("model_unavailable", context.target),
                        source_type="error",
                        error=error,
                        notes=[f"Ollama error: {error}"],
                    )
                    yield response.answer
        else:
            response = self._execute(context)
            yield response.answer

        if response is None:
            response = self._content_response(
                context,
                message("model_unavailable", context.target),
                source_type="error",
                error="No response was produced.",
            )
            yield response.answer

        self._commit(context, response)
        if holder is not None:
            holder["response"] = response

    def clear_history(self) -> None:
        self.history.clear()
        self.last_route = None

    def add_document(self, name: str, data: bytes):
        result = self.documents.add_document(name, data)
        self._status = None
        return result

    def clear_documents(self) -> None:
        self.documents.clear()
        self._status = None

    def refresh_status(self) -> Dict[str, Any]:
        self._ensure_llm()
        resolved: Dict[str, Any] = {}
        if hasattr(self.llm, "refresh"):
            try:
                resolved = self.llm.refresh()
            except Exception as exc:  # pragma: no cover - defensive
                resolved = {"error": f"{type(exc).__name__}: {exc}"}
        translator_installed = True
        if hasattr(self.translator, "is_installed"):
            translator_installed = bool(self.translator.is_installed())
        self._status = {
            "ollama_available": bool(resolved.get("available")),
            "configured_model": getattr(self.llm, "preferred_model", None),
            "active_model": resolved.get("model"),
            "installed_models": resolved.get("installed", []),
            "fell_back": resolved.get("fell_back", False),
            "ollama_error": resolved.get("error"),
            "embedding_model": EMBEDDING_MODEL,
            "embedding_available": bool(resolved.get("embedding_model_available")),
            "translator_backend": getattr(self.translator, "backend_name", "fallback"),
            "translator_installed": translator_installed,
            "translator_available": self._translator_available(),
            "documents": self.documents.stats(),
            "history_messages": len(self.history),
        }
        return self._status

    def status(self) -> Dict[str, Any]:
        if self._status is None:
            self.refresh_status()
        return self._status or {}

    def last_assistant_message(self) -> Optional[str]:
        for item in reversed(self.history):
            if item.get("role") == "assistant":
                return item.get("content")
        return None

    def last_user_message(self) -> Optional[str]:
        for item in reversed(self.history):
            if item.get("role") == "user":
                return item.get("content")
        return None

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------
    def _prepare(self, message_text: str, selected_language: str) -> _PreparedTurn:
        self._ensure_llm()
        started = time.perf_counter()
        text = (message_text or "").strip()
        detection = detect_language(text)
        target = self._resolve_target(text, selected_language, detection)
        decision = route_turn(
            text,
            has_documents=bool(self.documents.documents),
            has_history=bool(self.history),
            last_route=self.last_route,
        )
        if decision.requested_lang:
            target = decision.requested_lang

        context = _PreparedTurn(
            message=text,
            selected_language=selected_language,
            target=target,
            detection=detection,
            decision=decision,
            started=started,
        )

        if decision.route == ROUTE_GENERAL:
            self._prepare_general(context)
        elif decision.route == ROUTE_DOCUMENT:
            self._prepare_document(context)
        elif decision.route == ROUTE_TRANSLATION:
            context.translation_payload = decision.translation_text or self.last_assistant_message()
            if decision.target_lang:
                context.target = decision.target_lang
        elif decision.route == ROUTE_SUMMARIZE:
            self._prepare_summary(context)
        return context

    def _resolve_target(
        self, text: str, selected_language: str, detection: LanguageDetection
    ) -> str:
        selected = (selected_language or "").strip()
        if not selected or selected.lower() in ("auto", "auto detect", "detect"):
            return detection.code or "en"
        return language_code(selected) or "en"

    def _prepare_general(self, context: _PreparedTurn) -> None:
        decision = context.decision
        entry, score, provenance = self.kb.search(context.message)
        if (
            entry is not None
            and score >= FAQ_DIRECT_MIN_SCORE
            and (decision.domain_specific or score >= 0.75)
        ):
            context.faq = (entry, score, provenance)

        context.assist = self._should_translate_assist(context.target)
        system = general_system(
            context.target, time_sensitive=decision.time_sensitive, english=context.assist
        )
        if context.assist:
            system += (
                f"\nThe final answer will be machine-translated into "
                f"{language_name(context.target)}. Keep the content faithful and literal."
            )
        if decision.time_sensitive:
            context.notes.append(message("time_sensitive_note", context.target))
        if decision.domain_specific and context.faq is None:
            context.notes.append(
                "No curated knowledge-base entry matched; this answer is general model knowledge."
            )
        context.messages = self._build_messages(system, context.message)
        context.streamable = (
            context.faq is None
            and not context.assist
            and self._llm_available()
        )

    def _prepare_document(self, context: _PreparedTurn) -> None:
        if not self.documents.documents:
            return
        query = context.message
        if context.decision.is_followup:
            last_user = self.last_user_message()
            if last_user and len(content_tokens(context.message)) <= 4:
                query = f"{last_user} {context.message}"

        extra_queries: List[str] = []
        stats = self.documents.stats()
        if not stats.get("semantic_index") and context.detection.code != "en":
            document_languages = {chunk.lang for chunk in self.documents.chunks}
            if "en" in document_languages and self._llm_available():
                translated = self._translate_query_to_english(query)
                if translated:
                    extra_queries.append(translated)
                    context.notes.append(
                        "Lexical retrieval was assisted by an English translation of the query."
                    )

        context.retrieval_query = query
        context.retrieval = self.documents.search(query, extra_queries=extra_queries)
        context.assist = self._should_translate_assist(context.target)
        if context.decision.time_sensitive:
            context.notes.append(message("time_sensitive_note", context.target))

    def _prepare_summary(self, context: _PreparedTurn) -> None:
        if context.decision.document_reference and self.documents.documents:
            context.summary_source = self.documents.summary_context(MAX_CONTEXT_CHARS)
            context.summary_from_documents = True
        else:
            previous = self.last_assistant_message()
            if previous:
                context.summary_source = previous
            elif ":" in context.message:
                candidate = context.message.split(":", 1)[1].strip()
                if len(candidate) >= 40:
                    context.summary_source = candidate
        context.assist = self._should_translate_assist(context.target)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def _execute(self, context: _PreparedTurn) -> ChatResponse:
        try:
            if context.decision.route == ROUTE_CLARIFY:
                return self._handle_clarify(context)
            if context.decision.route == ROUTE_TRANSLATION:
                return self._handle_translation(context)
            if context.decision.route == ROUTE_SUMMARIZE:
                return self._handle_summary(context)
            if context.decision.route == ROUTE_DOCUMENT:
                return self._handle_document(context)
            return self._handle_general(context)
        except Exception as exc:  # pragma: no cover - defensive
            return self._content_response(
                context,
                message("model_unavailable", context.target),
                source_type="error",
                error=f"{type(exc).__name__}: {exc}",
                notes=[f"Internal error: {type(exc).__name__}: {exc}"],
            )

    def _handle_clarify(self, context: _PreparedTurn) -> ChatResponse:
        return self._content_response(
            context,
            message("clarification", context.target),
            source_type="clarification",
        )

    def _handle_general(self, context: _PreparedTurn) -> ChatResponse:
        if context.faq is not None:
            return self._faq_response(context)

        if not self._llm_available():
            return self._content_response(
                context,
                message("model_unavailable", context.target),
                source_type="error",
                error=getattr(self.llm, "last_error", None) or "Ollama is unavailable.",
                notes=self._general_notes(context),
            )

        if context.assist:
            english_context = context
            english_messages = self._build_messages(
                general_system(
                    context.target,
                    time_sensitive=context.decision.time_sensitive,
                    english=True,
                )
                + f"\nThe final answer will be machine-translated into {language_name(context.target)}.",
                context.message,
            )
            result = self.llm.chat(english_messages)
            if result.get("status") != "success" or not result.get("answer"):
                return self._content_response(
                    context,
                    message("model_unavailable", context.target),
                    source_type="error",
                    error=result.get("error"),
                    notes=[f"Ollama error: {result.get('error')}"],
                )
            translation = self._translate(
                result["answer"], "en", context.target, model=getattr(self.llm, "resolved_model", None)
            )
            if translation is not None and translation.ok and translation.status == "success":
                notes = self._general_notes(context) + [
                    self._translation_note(context.target, translation.backend)
                ]
                return self._general_response(
                    context,
                    translation.text,
                    model=result.get("model"),
                    notes=notes,
                    assist=True,
                    backend=translation.backend,
                    translated=True,
                )
            # Translator failed: fall back to direct generation in the target language.
            fallback = self.llm.chat(
                self._build_messages(
                    general_system(context.target, time_sensitive=context.decision.time_sensitive),
                    context.message,
                )
            )
            notes = self._general_notes(context) + [
                "Machine translation was unavailable; this answer was generated directly and may contain mixed-script text."
            ]
            return self._general_response(
                context,
                fallback.get("answer") or message("model_unavailable", context.target),
                model=fallback.get("model"),
                notes=notes,
                error=fallback.get("error"),
            )

        result = self.llm.chat(context.messages)
        if result.get("status") != "success" or not result.get("answer"):
            return self._content_response(
                context,
                message("model_unavailable", context.target),
                source_type="error",
                error=result.get("error"),
                notes=[f"Ollama error: {result.get('error')}"],
            )

        answer = result["answer"]
        notes = self._general_notes(context)
        repaired, repair_note = self._repair_script(answer, context.target, getattr(self.llm, "resolved_model", None))
        if repair_note:
            notes.append(repair_note)
            answer = repaired
        return self._general_response(context, answer, model=result.get("model"), notes=notes)

    def _faq_response(self, context: _PreparedTurn) -> ChatResponse:
        assert context.faq is not None
        entry, score, _ = context.faq
        stored = entry.answer.get(context.target)
        notes: List[str] = []
        if entry.fictional:
            notes.append(message("fictional_note", context.target))
        if entry.translation_status != "verified" and context.target != "en":
            notes.append("The stored FAQ translation is an unverified draft.")
        if context.decision.time_sensitive:
            notes.append(message("time_sensitive_note", context.target))
        notes.append(f"Curated FAQ topic: {entry.topic} (match score {score:.2f}, demo fixture).")

        if stored:
            return self._content_response(
                context,
                stored,
                route=ROUTE_FAQ,
                source_type="faq",
                notes=notes,
                evidence_found=True,
            )

        if not self._llm_available():
            return self._content_response(
                context,
                format_fallback_message("missing_translation"),
                route=ROUTE_FAQ,
                source_type="faq",
                notes=notes + ["No stored answer exists in the selected language."],
                evidence_found=True,
            )

        faq_text = self._faq_context(entry)
        excerpt = RetrievedChunk(
            chunk=SimpleNamespace(
                doc_name=f"Curated FAQ: {entry.topic}",
                page=None,
                section=entry.domain,
                text=faq_text,
                chunk_id=entry.id,
            )
        )
        system = grounded_system(context.target, english=context.assist)
        user = (
            f"Question: {context.message}\n\n{format_context([excerpt])}\n\n"
            "Answer only from the curated FAQ excerpt above."
        )
        messages = [{"role": "system", "content": system}] + self._llm_history() + [
            {"role": "user", "content": user}
        ]
        result = self.llm.chat(messages)
        if result.get("status") != "success" or not result.get("answer"):
            return self._content_response(
                context,
                format_fallback_message("missing_translation"),
                route=ROUTE_FAQ,
                source_type="faq",
                notes=notes + [f"Ollama error: {result.get('error')}"],
                error=result.get("error"),
                evidence_found=True,
            )
        return self._content_response(
            context,
            result["answer"],
            route=ROUTE_FAQ,
            source_type="faq",
            model=result.get("model"),
            notes=notes,
            evidence_found=True,
        )

    def _handle_document(self, context: _PreparedTurn) -> ChatResponse:
        if not self.documents.documents:
            return self._content_response(
                context,
                message("no_documents", context.target),
                source_type="document",
                evidence_found=False,
            )

        retrieval = context.retrieval
        if retrieval is None or not retrieval.evidence_found or not retrieval.chunks:
            notes = list(context.notes)
            if retrieval is not None:
                notes.append(
                    f"Best retrieval scores: lexical {retrieval.best_lexical:.2f}, "
                    f"semantic {retrieval.best_semantic:.2f}."
                )
                notes.extend(retrieval.warnings)
            return self._content_response(
                context,
                message("missing_evidence", context.target),
                source_type="document",
                evidence_found=False,
                retrieval_method=retrieval.method if retrieval else None,
                notes=notes,
            )

        context_block = format_context(retrieval.chunks)
        if len(context_block) > MAX_CONTEXT_CHARS:
            trimmed: List[RetrievedChunk] = []
            used = 0
            for item in retrieval.chunks:
                if used + len(item.text) > MAX_CONTEXT_CHARS:
                    break
                trimmed.append(item)
                used += len(item.text)
            context_block = format_context(trimmed)
            retrieval = RetrievalResult(
                query=retrieval.query,
                chunks=trimmed,
                evidence_found=True,
                method=retrieval.method,
                warnings=retrieval.warnings,
                latency_ms=retrieval.latency_ms,
                best_lexical=retrieval.best_lexical,
                best_semantic=retrieval.best_semantic,
            )

        if not self._llm_available():
            return self._extractive_document_response(context, retrieval)

        user = (
            f"Question: {context.message}\n\n{context_block}\n\n"
            "Answer only from the excerpts above. Cite each claim with [n]. "
            "If the excerpts do not contain the answer, say so clearly."
        )
        system = grounded_system(context.target, english=context.assist)
        messages = [{"role": "system", "content": system}] + self._llm_history() + [
            {"role": "user", "content": user}
        ]
        result = self.llm.chat(messages)
        if result.get("status") != "success" or not result.get("answer"):
            return self._extractive_document_response(
                context,
                retrieval,
                error=result.get("error"),
            )

        answer = result["answer"]
        cited = set(self._parse_citations(answer))
        notes = list(context.notes) + list(retrieval.warnings)
        backend = None
        translated = False

        if context.assist:
            translation = self._translate(
                answer, "en", context.target, model=getattr(self.llm, "resolved_model", None)
            )
            if translation is not None and translation.ok and translation.status == "success":
                translated_citations = set(self._parse_citations(translation.text))
                answer = translation.text
                if translated_citations:
                    cited = translated_citations
                notes.append(self._translation_note(context.target, translation.backend))
                backend = translation.backend
                translated = True
            else:
                notes.append(
                    "Machine translation failed; the answer is shown in English and may not match the selected language."
                )

        invalid_citations = sorted(
            index for index in cited if not 1 <= index <= len(retrieval.chunks)
        )
        if invalid_citations:
            notes.append(
                "The model cited passage numbers that were not retrieved: "
                + ", ".join(f"[{index}]" for index in invalid_citations)
                + "."
            )
        cited = {index for index in cited if 1 <= index <= len(retrieval.chunks)}
        notes.append(
            f"Grounded in {len(retrieval.chunks)} retrieved chunk(s); retrieval: {retrieval.method}."
        )

        if self._looks_like_missing_evidence(answer):
            answer = re.sub(r"\s+([.,!?।])", r"\1", _CITATION_RE.sub("", answer)).strip()
            notes.append(
                "The answer reports that the requested information is absent from the "
                "retrieved passages, so no source is cited for it."
            )
            return self._content_response(
                context,
                answer,
                source_type="document",
                model=result.get("model"),
                notes=notes,
                citations=[],
                evidence_found=False,
                retrieval_method=retrieval.method,
                backend=backend,
                translated=translated,
            )

        citations = self.documents.build_citations(retrieval.chunks, cited)
        if not cited:
            notes.append(
                "The model did not attach citation markers; the passages below were "
                "retrieved but are not cited by the answer."
            )
        return self._content_response(
            context,
            answer,
            source_type="document",
            model=result.get("model"),
            notes=notes,
            citations=citations,
            evidence_found=bool(cited),
            retrieval_method=retrieval.method,
            backend=backend,
            translated=translated,
        )

    def _extractive_document_response(
        self,
        context: _PreparedTurn,
        retrieval: RetrievalResult,
        error: Optional[str] = None,
    ) -> ChatResponse:
        lines = [message("model_unavailable", context.target), ""]
        cited = set()
        for index, item in enumerate(retrieval.chunks[:3], start=1):
            excerpt = item.text.strip().replace("\n", " ")
            lines.append(f"[{index}] {item.doc_name}"
                         + (f", p. {item.page}" if item.page else "")
                         + f": {excerpt[:400]}{'…' if len(excerpt) > 400 else ''}")
            cited.add(index)
        citations = self.documents.build_citations(retrieval.chunks, cited)
        notes = list(context.notes) + [
            "The language model was unavailable, so retrieved passages were shown verbatim instead."
        ]
        if error:
            notes.append(f"Ollama error: {error}")
        return self._content_response(
            context,
            "\n\n".join(lines),
            source_type="document",
            citations=citations,
            evidence_found=True,
            retrieval_method=retrieval.method,
            notes=notes,
            error=error,
        )

    def _handle_translation(self, context: _PreparedTurn) -> ChatResponse:
        payload = (context.translation_payload or "").strip()
        if not payload:
            return self._content_response(
                context,
                message("translation_text_missing", context.target),
                source_type="translation",
            )

        source = detect_language(payload).code or "en"
        target = context.target or "en"
        notes: List[str] = []

        if source == target:
            return self._content_response(
                context,
                payload,
                source_type="translation",
                notes=[f"The text is already in {language_name(target)}."],
                translated=False,
            )

        if self._translator_available():
            translation = self._translate(
                payload,
                source,
                target,
                model=getattr(self.llm, "resolved_model", None),
            )
            if (
                translation is not None
                and translation.status in ("success", "unchanged")
                and translation.text
            ):
                notes.append(self._translation_note(target, translation.backend))
                notes.extend(translation.notes)
                return self._content_response(
                    context,
                    translation.text,
                    source_type="translation",
                    notes=notes,
                    backend=translation.backend,
                    translated=True,
                )
            if translation is not None and translation.error:
                notes.append(f"Neural translation failed: {translation.error}")

        if not self._llm_available():
            return self._content_response(
                context,
                message("translation_text_missing", target),
                source_type="error",
                error="No translator or language model is available.",
                notes=notes,
            )

        system = translation_system(language_name(source), language_name(target))
        result = self.llm.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": payload},
            ],
            temperature=0.1,
            num_predict=600,
        )
        if result.get("status") != "success" or not result.get("answer"):
            return self._content_response(
                context,
                message("model_unavailable", target),
                source_type="error",
                error=result.get("error"),
                notes=notes,
            )
        notes.append(
            message("translation_disclaimer", target)
            + " Backend: local language model (may be less reliable than a dedicated translator)."
        )
        return self._content_response(
            context,
            result["answer"],
            source_type="translation",
            model=result.get("model"),
            notes=notes,
            backend="llm",
            translated=True,
        )

    def _handle_summary(self, context: _PreparedTurn) -> ChatResponse:
        source = (context.summary_source or "").strip()
        if len(source) < 30:
            return self._content_response(
                context,
                message("summarize_missing_source", context.target),
                source_type="summary",
            )
        if not self._llm_available():
            return self._content_response(
                context,
                message("model_unavailable", context.target),
                source_type="error",
                error=getattr(self.llm, "last_error", None) or "Ollama is unavailable.",
            )

        if context.assist:
            system = summary_system(context.target, english=True)
            user = f"Text to summarize:\n{source}\n\nProvide the summary in English."
        else:
            system = summary_system(context.target)
            user = f"Text to summarize:\n{source}\n\nProvide the summary in {language_name(context.target)}."
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        result = self.llm.chat(messages)
        if result.get("status") != "success" or not result.get("answer"):
            return self._content_response(
                context,
                message("model_unavailable", context.target),
                source_type="error",
                error=result.get("error"),
            )

        answer = result["answer"]
        notes = ["Summary generated by the model; verify important facts against the source."]
        backend = None
        translated = False
        if context.assist:
            translation = self._translate(
                answer, "en", context.target, model=getattr(self.llm, "resolved_model", None)
            )
            if translation is not None and translation.ok and translation.status == "success":
                answer = translation.text
                notes.append(self._translation_note(context.target, translation.backend))
                backend = translation.backend
                translated = True
            else:
                notes.append("Machine translation failed; the summary is shown in English.")

        citations: List[Citation] = []
        if context.summary_from_documents:
            for index, info in enumerate(self.documents.documents.values(), start=1):
                citations.append(
                    Citation(
                        index=index,
                        doc_name=info.name,
                        page=None,
                        section="whole document",
                        excerpt=f"{info.chunks} indexed chunks",
                        chunk_id=info.doc_id,
                        cited=True,
                    )
                )
            notes.append("Summary covers the uploaded document(s) listed in the sources.")
        return self._content_response(
            context,
            answer,
            source_type="summary",
            model=result.get("model"),
            notes=notes,
            citations=citations,
            backend=backend,
            translated=translated,
        )

    # ------------------------------------------------------------------
    # Response helpers
    # ------------------------------------------------------------------
    def _content_response(
        self,
        context: _PreparedTurn,
        answer: str,
        *,
        route: Optional[str] = None,
        source_type: Optional[str] = None,
        model: Optional[str] = None,
        citations: Optional[List[Citation]] = None,
        notes: Optional[List[str]] = None,
        error: Optional[str] = None,
        evidence_found: Optional[bool] = None,
        retrieval_method: Optional[str] = None,
        backend: Optional[str] = None,
        translated: bool = False,
        assist: bool = False,
        latency_ms: Optional[float] = None,
    ) -> ChatResponse:
        return ChatResponse(
            answer=answer,
            route=route or context.decision.route,
            target_lang=context.target,
            detected_lang=context.detection.code,
            detection_confidence=context.detection.confidence,
            model=model,
            latency_ms=latency_ms
            if latency_ms is not None
            else round((time.perf_counter() - context.started) * 1000.0, 2),
            citations=citations or [],
            notes=list(dict.fromkeys(notes or [])),
            error=error,
            source_type=source_type or (route or context.decision.route),
            evidence_found=evidence_found,
            retrieval_method=retrieval_method,
            backend=backend,
            translated=translated,
            used_translation_assist=assist,
        )

    def _general_response(
        self,
        context: _PreparedTurn,
        answer: str,
        *,
        model: Optional[str],
        notes: Optional[List[str]] = None,
        assist: bool = False,
        backend: Optional[str] = None,
        translated: bool = False,
        error: Optional[str] = None,
    ) -> ChatResponse:
        return self._content_response(
            context,
            answer,
            route=ROUTE_GENERAL,
            source_type="general",
            model=model,
            notes=notes,
            assist=assist,
            backend=backend,
            translated=translated,
            error=error,
        )

    def _general_notes(self, context: _PreparedTurn) -> List[str]:
        notes = list(context.notes)
        if context.decision.time_sensitive:
            if message("time_sensitive_note", context.target) not in notes:
                notes.append(message("time_sensitive_note", context.target))
        return notes

    def _translation_note(self, target: str, backend: str) -> str:
        return message("translation_disclaimer", target) + f" Backend: {backend}."

    def _build_messages(self, system: str, user_text: str) -> List[Dict[str, str]]:
        return [{"role": "system", "content": system}] + self._llm_history() + [
            {"role": "user", "content": user_text}
        ]

    def _llm_history(self) -> List[Dict[str, str]]:
        if not self.history:
            return []
        recent = self.history[-HISTORY_MAX_MESSAGES:]
        total = sum(len(item.get("content", "")) for item in recent)
        while total > HISTORY_MAX_CHARS and len(recent) > 2:
            removed = recent.pop(0)
            total -= len(removed.get("content", ""))
        return [
            {"role": item["role"], "content": item["content"]}
            for item in recent
            if item.get("content")
        ]

    def _commit(self, context: _PreparedTurn, response: ChatResponse) -> None:
        self.history.append(
            {
                "role": "user",
                "content": context.message,
                "route": response.route,
                "lang": context.detection.code,
            }
        )
        self.history.append(
            {
                "role": "assistant",
                "content": response.answer,
                "route": response.route,
                "lang": response.target_lang,
            }
        )
        self._trim_history()
        self.last_route = response.route

    def _trim_history(self) -> None:
        if len(self.history) > HISTORY_MAX_MESSAGES:
            self.history = self.history[-HISTORY_MAX_MESSAGES:]
        total = sum(len(item.get("content", "")) for item in self.history)
        while total > HISTORY_MAX_CHARS and len(self.history) > 2:
            removed = self.history.pop(0)
            total -= len(removed.get("content", ""))

    # ------------------------------------------------------------------
    # Integrations
    # ------------------------------------------------------------------
    def _ensure_llm(self) -> None:
        if hasattr(self.llm, "refresh") and getattr(self.llm, "_status", None) is None:
            try:
                self.llm.refresh()
            except Exception:
                pass

    def _llm_available(self) -> bool:
        if hasattr(self.llm, "available"):
            try:
                return bool(self.llm.available())
            except Exception:
                return False
        return True

    def _translator_available(self) -> bool:
        if self.translator is None:
            return False
        try:
            return bool(self.translator.is_available())
        except Exception:
            return False

    def _should_translate_assist(self, target: str) -> bool:
        return target in self.translation_assist_langs and self._translator_available()

    def _translate(
        self,
        text: str,
        source: str,
        target: str,
        model: Optional[str] = None,
    ) -> Optional[TranslationResult]:
        if not self._translator_available():
            return None
        try:
            return self.translator.translate(
                text,
                source,
                target,
                unload_after=True,
                ollama_model_to_unload=model,
            )
        except TypeError:
            try:
                return self.translator.translate(text, source, target)
            except Exception:
                return None
        except Exception:
            return None

    def _translate_query_to_english(self, query: str) -> Optional[str]:
        result = self.llm.chat(
            [
                {"role": "system", "content": query_translation_system()},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            num_predict=120,
        )
        if result.get("status") == "success" and result.get("answer"):
            return result["answer"].strip()
        return None

    def _repair_script(
        self, answer: str, target: str, model: Optional[str]
    ) -> Tuple[str, Optional[str]]:
        if target == "en" or not answer:
            return answer, None
        purity = script_purity(answer, target)
        if purity >= SCRIPT_PURITY_MIN:
            return answer, None
        if not self._translator_available():
            return answer, (
                "The model output is only "
                f"{purity:.0%} {language_name(target)} script and may be mixed or imperfect."
            )
        detected = detect_language(answer)
        if detected.code != "en" and purity >= 0.35:
            return answer, (
                "The model output contains mixed-language text; verify the answer carefully."
            )
        translation = self._translate(answer, "en", target, model=model)
        if translation is not None and translation.status == "success" and translation.text:
            improved = script_purity(translation.text, target)
            if improved > purity:
                return translation.text, (
                    "The model produced mixed-script output; the answer was repaired by machine translation."
                )
        return answer, (
            "The model output may contain mixed-script or imperfect text; verify it carefully."
        )

    def _parse_citations(self, answer: str) -> List[int]:
        return sorted({int(match) for match in _CITATION_RE.findall(answer or "")})

    def _looks_like_missing_evidence(self, answer: str) -> bool:
        lowered = (answer or "").casefold()
        return any(phrase.casefold() in lowered for phrase in _MISSING_EVIDENCE_PHRASES)

    @staticmethod
    def _faq_context(entry: KnowledgeEntry) -> str:
        lines = [
            f"Topic: {entry.topic}",
            f"Domain: {entry.domain}",
        ]
        english = entry.answer.get("en")
        if english:
            lines.append(f"Reference answer (English): {english}")
        if entry.facts:
            lines.append("Key facts: " + "; ".join(entry.facts))
        source = entry.source or {}
        if source.get("reference"):
            lines.append(f"Source reference: {source.get('reference')}")
        if entry.fictional:
            lines.append("Note: this is fictional test fixture data, not real government policy.")
        return "\n".join(lines)
