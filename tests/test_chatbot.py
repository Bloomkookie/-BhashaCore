"""Offline orchestrator tests: routing outcomes, provenance, memory, failures.

These tests stub the language model, translator, and embeddings so they run
without Ollama or the NLLB weights.  Live multilingual quality is measured by
``evaluate_multilingual.py`` instead.
"""

import json
import unittest
from pathlib import Path

import numpy as np

from chatbot import Chatbot
from config import HISTORY_MAX_MESSAGES
from documents import DocumentStore
from knowledge_base import KnowledgeBase
from translation_service import TranslationResult

BASE_DIR = Path(__file__).resolve().parent.parent
TRANSLATIONS = {
    "en": "This is an English answer.",
    "hi": "यह हिंदी उत्तर है।",
    "bn": "এটি বাংলা উত্তর।",
    "pa": "ਇਹ ਪੰਜਾਬੀ ਜਵਾਬ ਹੈ।",
    "as": "এইটো অসমীয়া উত্তৰ।",
    "ne": "यो नेपाली उत्तर हो।",
}


class StubLLM:
    preferred_model = "stub"
    resolved_model = "stub"

    def __init__(self, answers=None, available=True):
        self.answers = list(answers or ["Stub answer."])
        self.available_flag = available
        self.calls = []

    def _next(self):
        if len(self.answers) > 1:
            return self.answers.pop(0)
        return self.answers[0]

    def available(self):
        return self.available_flag

    def refresh(self):
        if not self.available_flag:
            return {"available": False, "error": "stub offline"}
        return {"available": True, "model": "stub", "installed": ["stub"]}

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        if not self.available_flag:
            return {"status": "error", "error": "stub offline"}
        return {"status": "success", "answer": self._next(), "model": "stub"}

    def stream_chat(self, messages, meta=None):
        self.calls.append(messages)
        meta = meta if meta is not None else {}
        meta.update({"status": "success", "model": "stub"})
        for piece in ("Stub ", "streamed ", "answer."):
            yield piece


class StubTranslator:
    backend_name = "stub-nllb"

    def __init__(self, available=True, fail=False):
        self.available_flag = available
        self.fail = fail
        self.calls = []

    def is_available(self):
        return self.available_flag

    def translate(self, text, src, tgt, **kwargs):
        self.calls.append((text, src, tgt))
        if self.fail:
            return TranslationResult("", "error", "stub-nllb", src, tgt, error="stub failure")
        return TranslationResult(TRANSLATIONS.get(tgt, text), "success", "stub-nllb", src, tgt)


class StubEmbedder:
    """Returns identical vectors, so every chunk is semantically reachable."""

    def available(self):
        return True

    def embed(self, texts, keep_alive="0s"):
        return np.ones((len(texts), 4), dtype=np.float32)


def make_bot(answers=None, llm_available=True, translator=None, docs=None):
    store = DocumentStore(embedder=StubEmbedder()) if docs is None else docs
    if docs is None:
        store.add_document(
            "policy.txt",
            b"Library membership fee is INR 200 per year. Late return fine is INR 5 per day.",
        )
    kb = KnowledgeBase()
    kb.load_faq_file()
    return Chatbot(
        kb=kb,
        doc_store=store,
        llm=StubLLM(answers, available=llm_available),
        translator=translator or StubTranslator(),
    )


class TestGeneralConversation(unittest.TestCase):
    def test_english_general_answer_is_model_generated(self):
        bot = make_bot(["A Python function is a reusable block of code."])
        response = bot.respond("What is a Python function?")
        self.assertEqual(response.route, "general")
        self.assertEqual(response.source_type, "general")
        self.assertIn("reusable", response.answer)
        self.assertEqual(response.citations, [])
        self.assertEqual(bot.llm.calls[0][0]["role"], "system")

    def test_hindi_answer_is_direct(self):
        bot = make_bot(["फ़ंक्शन कोड का एक पुनः प्रयोग योग्य हिस्सा है।"])
        response = bot.respond("पायथन में फ़ंक्शन क्या है?")
        self.assertEqual(response.target_lang, "hi")
        self.assertFalse(response.translated)
        self.assertFalse(response.used_translation_assist)

    def test_punjabi_uses_translation_assist(self):
        bot = make_bot(["A function is a reusable block of code."])
        response = bot.respond("ਪਾਇਥਨ ਵਿੱਚ ਫੰਕਸ਼ਨ ਕੀ ਹੈ?")
        self.assertEqual(response.target_lang, "pa")
        self.assertTrue(response.translated)
        self.assertTrue(response.used_translation_assist)
        self.assertEqual(response.backend, "stub-nllb")
        self.assertEqual(response.answer, TRANSLATIONS["pa"])
        self.assertTrue(any("machine-translated" in note or "Backend" in note for note in response.notes))

    def test_translation_failure_falls_back_to_direct_generation(self):
        bot = make_bot(
            ["English draft.", "ਸਿੱਧਾ ਪੰਜਾਬੀ ਜਵਾਬ।"],
            translator=StubTranslator(fail=True),
        )
        response = bot.respond("ਪਾਇਥਨ ਵਿੱਚ ਫੰਕਸ਼ਨ ਕੀ ਹੈ?")
        self.assertEqual(response.answer, "ਸਿੱਧਾ ਪੰਜਾਬੀ ਜਵਾਬ।")
        self.assertTrue(any("translation was unavailable" in note for note in response.notes))

    def test_mixed_script_output_is_repaired_by_translation(self):
        bot = make_bot(["This answer is mostly English."])
        response = bot.respond("हिंदी में गुरुत्वाकर्षण समझाइए।")
        self.assertEqual(response.answer, TRANSLATIONS["hi"])
        self.assertTrue(any("repaired by machine translation" in note for note in response.notes))

    def test_unavailable_model_returns_localized_error(self):
        bot = make_bot(llm_available=False)
        response = bot.respond("आसमान नीला क्यों होता है?")
        self.assertEqual(response.source_type, "error")
        self.assertTrue(response.answer)
        self.assertTrue(response.error)

    def test_clarification_does_not_call_model(self):
        bot = make_bot()
        response = bot.respond("What?", selected_language="Auto")
        self.assertEqual(response.route, "clarify")
        self.assertEqual(bot.llm.calls, [])


class TestCuratedFaq(unittest.TestCase):
    def test_faq_answers_in_all_six_languages(self):
        data = json.loads((BASE_DIR / "evaluation_questions.json").read_text(encoding="utf-8"))
        bot = make_bot()
        for row in data:
            for lang in ("en", "hi", "bn", "pa", "as", "ne"):
                with self.subTest(row=row["id"], lang=lang):
                    response = bot.respond(row[f"question_{lang}"])
                    self.assertEqual(response.route, "faq")
                    self.assertEqual(response.source_type, "faq")
                    self.assertTrue(response.answer.strip())
                    self.assertTrue(response.evidence_found)
                    self.assertEqual(bot.llm.calls, [])


class TestTranslationAndSummary(unittest.TestCase):
    def test_translation_route_uses_translator(self):
        bot = make_bot()
        response = bot.respond("Translate to Hindi: The district office is closed on Sunday.")
        self.assertEqual(response.route, "translation")
        self.assertEqual(response.source_type, "translation")
        self.assertTrue(response.translated)
        self.assertEqual(response.answer, TRANSLATIONS["hi"])
        self.assertEqual(bot.translator.calls[0][1:], ("en", "hi"))

    def test_translation_of_same_language_is_unchanged(self):
        bot = make_bot()
        response = bot.respond("Translate to English: Hello world.")
        self.assertFalse(response.translated)
        self.assertIn("already in English", " ".join(response.notes))

    def test_summary_uses_previous_answer(self):
        previous = "Gravity attracts objects with mass and keeps planets in orbit around stars."
        bot = make_bot([previous, "गुरुत्वाकर्षण द्रव्यमान वाली वस्तुओं को खींचता है।"])
        bot.respond("Explain gravity in one sentence.")
        response = bot.respond("Summarize that in Hindi.")
        self.assertEqual(response.route, "summarize")
        self.assertEqual(response.source_type, "summary")
        summary_messages = bot.llm.calls[-1]
        self.assertIn(previous, summary_messages[-1]["content"])

    def test_document_summary_lists_the_document(self):
        bot = make_bot(["The policy sets a yearly fee and a late-return fine."])
        response = bot.respond("Summarize the uploaded document.")
        self.assertEqual(response.source_type, "summary")
        self.assertTrue(response.citations)
        self.assertEqual(response.citations[0].doc_name, "policy.txt")
        self.assertTrue(response.citations[0].cited)


class TestDocumentAnswers(unittest.TestCase):
    def test_grounded_answer_cites_retrieved_passage(self):
        bot = make_bot(["The membership fee is INR 200 per year [1]."])
        response = bot.respond("According to the uploaded file, what is the membership fee?")
        self.assertEqual(response.source_type, "document")
        self.assertTrue(response.evidence_found)
        self.assertEqual(len(response.citations), 1)
        self.assertTrue(response.citations[0].cited)
        self.assertIn("[1]", response.answer)

    def test_refusal_with_citation_is_not_presented_as_support(self):
        bot = make_bot(["The opening hours are not provided [1]."])
        response = bot.respond("According to the uploaded file, what are the opening hours?")
        self.assertFalse(response.evidence_found)
        self.assertEqual(response.citations, [])
        self.assertNotIn("[1]", response.answer)
        self.assertTrue(any("no source is cited" in note for note in response.notes))

    def test_uncited_answer_lists_retrieved_passages_honestly(self):
        bot = make_bot(["The membership fee is 200 rupees."])
        response = bot.respond("According to the uploaded file, what is the membership fee?")
        self.assertFalse(response.evidence_found)
        self.assertTrue(response.citations)
        self.assertTrue(all(not citation.cited for citation in response.citations))
        self.assertTrue(any("did not attach citation markers" in note for note in response.notes))
        self.assertNotIn("Sources:", response.answer)

    def test_invalid_citation_number_is_reported(self):
        bot = make_bot(["The fee is 200 rupees [4]."])
        response = bot.respond("According to the uploaded file, what is the membership fee?")
        self.assertFalse(response.evidence_found)
        self.assertTrue(any("not retrieved" in note for note in response.notes))

    def test_missing_retrieval_evidence_skips_model(self):
        store = DocumentStore(use_embeddings=False)
        store.add_document("policy.txt", b"Library membership fee is INR 200 per year.")
        bot = make_bot(docs=store)
        response = bot.respond("According to the uploaded file, who is the library director?")
        self.assertFalse(response.evidence_found)
        self.assertEqual(response.citations, [])
        self.assertEqual(bot.llm.calls, [])

    def test_no_documents_loaded(self):
        bot = make_bot()
        bot.clear_documents()
        response = bot.respond("According to the uploaded file, what is the fee?")
        self.assertEqual(response.route, "document")
        self.assertFalse(response.evidence_found)
        self.assertTrue(response.answer.strip())

    def test_extractive_fallback_when_model_is_down(self):
        bot = make_bot(llm_available=False)
        response = bot.respond("According to the uploaded file, what is the membership fee?")
        self.assertEqual(response.source_type, "document")
        self.assertTrue(response.citations)
        self.assertIn("policy.txt", response.answer)
        self.assertTrue(any("shown verbatim" in note for note in response.notes))

    def test_punjabi_document_answer_is_translated(self):
        bot = make_bot(["The membership fee is INR 200 per year [1]."])
        response = bot.respond("ਅੱਪਲੋਡ ਕੀਤੀ ਫਾਈਲ ਅਨੁਸਾਰ ਮੈਂਬਰਸ਼ਿਪ ਫੀਸ ਕਿੰਨੀ ਹੈ?")
        self.assertTrue(response.translated)
        self.assertEqual(response.answer, TRANSLATIONS["pa"])
        self.assertTrue(response.citations[0].cited)


class TestSessionMemory(unittest.TestCase):
    def test_document_followup_retrieves_again(self):
        bot = make_bot(
            [
                "The membership fee is INR 200 per year [1].",
                "The late return fine is INR 5 per day [1].",
            ]
        )
        bot.respond("According to the uploaded file, what is the membership fee?")
        response = bot.respond("What about the late return fine?")
        self.assertEqual(response.route, "document")
        self.assertEqual(len(bot.llm.calls), 2)
        second_prompt = bot.llm.calls[-1][-1]["content"]
        self.assertIn("late return fine", second_prompt.lower())
        self.assertTrue(response.evidence_found)

    def test_history_is_trimmed(self):
        bot = make_bot(["Short answer."])
        for index in range(6):
            bot.respond(f"Explain concept number {index}.")
        self.assertLessEqual(len(bot.history), HISTORY_MAX_MESSAGES)

    def test_clear_history_resets_state(self):
        bot = make_bot(["Short answer."])
        bot.respond("Explain gravity.")
        bot.clear_history()
        self.assertEqual(bot.history, [])
        self.assertIsNone(bot.last_route)
        self.assertIsNone(bot.last_assistant_message())

    def test_prior_answer_is_context_not_evidence(self):
        # A follow-up about a document must go back through retrieval; the
        # previous assistant answer is never reused as a verified source.
        bot = make_bot(
            [
                "The membership fee is INR 200 per year [1].",
                "The fine is INR 5 per day [1].",
            ]
        )
        first = bot.respond("According to the uploaded file, what is the membership fee?")
        second = bot.respond("And what is the fine?")
        self.assertEqual(first.route, "document")
        self.assertEqual(second.route, "document")
        self.assertEqual(second.retrieval_method, "hybrid(lexical+bge-m3)")


class TestStreaming(unittest.TestCase):
    def test_stream_yields_pieces_and_stores_response(self):
        bot = make_bot()
        holder = {}
        pieces = list(bot.respond_stream("Explain gravity.", holder=holder))
        self.assertEqual("".join(pieces), "Stub streamed answer.")
        self.assertIn("response", holder)
        self.assertEqual(holder["response"].answer, "Stub streamed answer.")
        self.assertEqual(holder["response"].route, "general")
        self.assertEqual(len(bot.history), 2)

    def test_empty_stream_is_an_error(self):
        bot = make_bot()
        bot.llm.stream_chat = lambda messages, meta=None: iter(())
        holder = {}
        pieces = list(bot.respond_stream("Explain gravity.", holder=holder))
        self.assertEqual(holder["response"].source_type, "error")
        self.assertTrue(pieces)
        self.assertIsNotNone(holder["response"].error)


if __name__ == "__main__":
    unittest.main()
