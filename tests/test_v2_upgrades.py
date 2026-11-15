"""Comprehensive automated test suite for BhashaCore v2.0 upgrade modules.

Covers:
- config.py default settings and environment overrides
- ollama_client.py status checks, error handling, parameter validation
- knowledge_base.py multi-domain storage, RAG sample loading, and search provenance
- translation_service.py fallback backend and Flores code mapping
"""

import os
import unittest
from pathlib import Path

from config import (
    DEFAULT_FAQ_PATH,
    OLLAMA_MODEL,
    SUPPORTED_LANGUAGES,
    get_config_summary,
)
from knowledge_base import KnowledgeBase, KnowledgeEntry
from ollama_client import check_model_available, check_ollama_status, generate_response
from translation_service import (
    INDICTRANS2_LANG_CODES,
    FallbackTranslationBackend,
    get_translation_backend,
    protect_currency_tokens,
    restore_currency_tokens,
)


class TestConfigModule(unittest.TestCase):
    """Verify configuration settings and env-var helper functions."""

    def test_default_config_summary(self):
        summary = get_config_summary()
        self.assertIn("ollama_url", summary)
        self.assertIn("configured_model", summary)
        self.assertEqual(summary["configured_model"], OLLAMA_MODEL)
        self.assertEqual(summary["supported_languages_count"], 6)

    def test_supported_languages_mapping(self):
        expected_codes = {"en", "hi", "bn", "pa", "as", "ne"}
        actual_codes = set(SUPPORTED_LANGUAGES.values())
        self.assertEqual(actual_codes, expected_codes)


class TestOllamaClient(unittest.TestCase):
    """Verify Ollama client behavior and graceful error handling when unreachable."""

    def test_check_ollama_status_offline_handling(self):
        # Point to an invalid dummy port to verify offline handling
        status = check_ollama_status("http://127.0.0.1:54321")
        self.assertFalse(status["available"])
        self.assertIsNotNone(status["error"])

    def test_check_model_available_offline(self):
        # Should cleanly return False when daemon is unreachable
        res = check_model_available("nonexistent_model", "http://127.0.0.1:54321")
        self.assertFalse(res)

    def test_generate_response_offline_fallback(self):
        # Should gracefully return error dictionary without crashing
        res = generate_response(
            question="What is the test deadline?",
            context="Test context",
            language="English",
            base_url="http://127.0.0.1:54321",
            timeout=2,
        )
        self.assertEqual(res["status"], "error")
        self.assertIn("refused", res["error"].lower())


class TestKnowledgeBase(unittest.TestCase):
    """Verify multi-domain knowledge base loading, indexing, and provenance."""

    def setUp(self):
        self.kb = KnowledgeBase()
        self.kb.load_faq_file(DEFAULT_FAQ_PATH)

    def test_load_faq_file_domains(self):
        domains = self.kb.get_domains()
        self.assertIn("education", domains)
        entries = self.kb.get_entries_by_domain("education")
        self.assertGreaterEqual(len(entries), 6)

    def test_add_custom_domain_entry(self):
        entry = KnowledgeEntry(
            entry_id="gov_test_01",
            topic="pension_scheme",
            domain="public_services",
            question_examples={
                "en": ["How to apply for public pension?"],
                "hi": ["सार्वजनिक पेंशन के लिए आवेदन कैसे करें?"],
            },
            answer={
                "en": "Visit the district social welfare office.",
                "hi": "जिला समाज कल्याण कार्यालय में संपर्क करें।",
            },
            fictional=False,
            translation_status="draft_unverified",
        )
        self.kb.add_entry(entry)
        domains = self.kb.get_domains()
        self.assertIn("public_services", domains)
        self.assertEqual(len(self.kb.get_entries_by_domain("public_services")), 1)

    def test_search_with_provenance(self):
        entry, score, prov = self.kb.search("What documents are required for the scholarship?")
        self.assertIsNotNone(entry)
        self.assertGreaterEqual(score, 0.40)
        self.assertEqual(prov["status"], "matched")
        self.assertEqual(prov["domain"], "education")
        self.assertTrue(prov["fictional"])

    def test_search_unsupported_rejection(self):
        entry, score, prov = self.kb.search("What is the capital of Australia?")
        self.assertIsNone(entry)
        self.assertIn("rejected", prov.get("status", ""))

    def test_compile_grounding_context(self):
        ctx = self.kb.compile_grounding_context(max_entries=3)
        self.assertIn("Verified Public Service", ctx)
        self.assertIn("Topic:", ctx)


class TestTranslationService(unittest.TestCase):
    """Verify translation interface, fallback behavior, and language code mappings."""

    def test_flores_codes_defined(self):
        for code in ["en", "hi", "bn", "pa", "as", "ne"]:
            self.assertIn(code, INDICTRANS2_LANG_CODES)

    def test_fallback_backend_returns_text(self):
        backend = FallbackTranslationBackend()
        self.assertTrue(backend.is_available())
        res = backend.translate("Hello world", "en", "hi")
        self.assertEqual(res, "Hello world")

    def test_get_translation_backend_default(self):
        backend = get_translation_backend(enabled=False)
        self.assertIsInstance(backend, FallbackTranslationBackend)

    def test_currency_codes_round_trip_unchanged(self):
        text = "Fee is INR 200 and Rs. 5, or USD 3."
        protected, mapping = protect_currency_tokens(text)
        self.assertNotIn("INR", protected)
        self.assertNotIn("USD", protected)
        self.assertEqual(restore_currency_tokens(protected, mapping), text)

    def test_unknown_currency_token_is_removed(self):
        self.assertEqual(restore_currency_tokens("fee ZQXCURZQX 200", {}), "fee  200")


if __name__ == "__main__":
    unittest.main()
