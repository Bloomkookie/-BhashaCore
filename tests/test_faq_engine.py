"""Comprehensive automated tests for BhashaCore FAQ Engine.

Covers:
- Unicode Indic normalization and combining-mark retention
- Multilingual tokenization and stopword separation
- Stem / inflection matching across Indic scripts
- Semantic similarity scoring
- Knowledge base loading and schema validation
- FAQ retrieval accuracy across all 6 supported languages
- Hallucination prevention on unsupported and out-of-domain queries
- Ambiguity rejection
- Missing translation handling
"""

import json
import tempfile
import unittest
from pathlib import Path

from faq_engine import (
    DEFAULT_FAQ_PATH,
    LANGUAGE_LABELS,
    are_stem_match,
    content_tokens,
    find_faq,
    format_fallback_message,
    get_answer,
    load_faqs,
    normalize,
    similarity,
    tokens,
)


class TestIndicNormalization(unittest.TestCase):
    """Verify Unicode normalization preserves Indic vowels, matras and halants."""

    def test_latin_normalization(self):
        text = "What Documents Are Required, For The Scholarship???"
        self.assertEqual(
            normalize(text),
            "what documents are required for the scholarship",
        )

    def test_hindi_matras_preserved(self):
        # छात्रवृत्ति has virama/halant (्) and matras (ा, ृ, ि)
        raw = "छात्रवृत्ति के लिए कौन-कौन से दस्तावेज़ आवश्यक हैं?"
        norm = normalize(raw)
        self.assertIn("छात्रवृत्ति", norm)
        self.assertIn("दस्तावेज़", norm)
        self.assertIn("आवश्यक", norm)
        # Verify words are not shredded into single characters
        toks = norm.split()
        self.assertIn("छात्रवृत्ति", toks)
        self.assertIn("दस्तावेज़", toks)

    def test_bengali_matras_preserved(self):
        raw = "বৃত্তির জন্য কোন কোন নথি প্রয়োজন?"
        norm = normalize(raw)
        self.assertIn("বৃত্তির", norm)
        self.assertIn("নথি", norm)
        self.assertIn("প্রয়োজন", norm)
        toks = norm.split()
        self.assertIn("বৃত্তির", toks)
        self.assertIn("নথি", toks)

    def test_punjabi_gurmukhi_preserved(self):
        raw = "ਵਜ਼ੀਫ਼ੇ ਲਈ ਕਿਹੜੇ ਦਸਤਾਵੇਜ਼ ਲੋੜੀਂਦੇ ਹਨ?"
        norm = normalize(raw)
        self.assertIn("ਵਜ਼ੀਫ਼ੇ", norm)
        self.assertIn("ਦਸਤਾਵੇਜ਼", norm)
        toks = norm.split()
        self.assertIn("ਵਜ਼ੀਫ਼ੇ", toks)
        self.assertIn("ਦਸਤਾਵੇਜ਼", toks)

    def test_assamese_characters_preserved(self):
        raw = "বৃত্তিৰ বাবে কি কি নথি-পত্ৰৰ প্ৰয়োজন?"
        norm = normalize(raw)
        self.assertIn("বৃত্তিৰ", norm)
        self.assertIn("নথি", norm)
        self.assertIn("প্ৰয়োজন", norm)

    def test_nepali_devanagari_preserved(self):
        raw = "छात्रवृत्तिका लागि कुन-कुन कागजात चाहिन्छ?"
        norm = normalize(raw)
        self.assertIn("छात्रवृत्तिका", norm)
        self.assertIn("कागजात", norm)
        self.assertIn("चाहिन्छ", norm)

    def test_empty_and_whitespace(self):
        self.assertEqual(normalize(""), "")
        self.assertEqual(normalize("   \n\t  "), "")
        self.assertEqual(normalize("???!!!"), "")
        self.assertEqual(normalize("123 456"), "123 456")


class TestTokenizationAndStemming(unittest.TestCase):
    """Test token extraction and morphological stem matching."""

    def test_tokens_extraction(self):
        toks = tokens("What documents are required?")
        self.assertEqual(toks, {"what", "documents", "are", "required"})

    def test_content_tokens_filter_stopwords(self):
        ct = content_tokens("What is the application deadline?")
        self.assertEqual(ct, {"application", "deadline"})

    def test_domain_stopwords_filtered(self):
        # "scholarship" is a common domain term present across all topics
        ct = content_tokens("What documents are required for the scholarship?")
        self.assertNotIn("scholarship", ct)
        self.assertIn("documents", ct)
        self.assertIn("required", ct)

    def test_are_stem_match_indic(self):
        self.assertTrue(are_stem_match("दस्तावेज़", "दस्तावेज़ों"))
        self.assertTrue(are_stem_match("कागजात", "कागजातहरू"))
        self.assertTrue(are_stem_match("নথি", "নথিগুলি"))
        self.assertTrue(are_stem_match("বৃত্তি", "বৃত্তির"))
        self.assertTrue(are_stem_match("ਅਰਜ਼ੀ", "ਅਰਜ਼ੀਆਂ"))

    def test_are_stem_match_latin(self):
        self.assertTrue(are_stem_match("document", "documents"))
        self.assertTrue(are_stem_match("deadline", "deadlines"))
        self.assertFalse(are_stem_match("apply", "documents"))


class TestSimilarityCalculation(unittest.TestCase):
    """Test semantic similarity computation on valid and invalid pairs."""

    def test_exact_match(self):
        self.assertEqual(
            similarity("What documents are required?", "What documents are required?"),
            1.0,
        )

    def test_indic_exact_match(self):
        self.assertEqual(
            similarity("আবেদনের শেষ তারিখ কী?", "আবেদনের শেষ তারিখ কী?"),
            1.0,
        )

    def test_unrelated_questions_score_low(self):
        # Stopword overlap (What is the...) must not fool the similarity scorer
        score = similarity(
            "What is the capital of India?",
            "What is the application deadline?",
        )
        self.assertLess(score, 0.35)

    def test_laptop_query_rejected(self):
        score = similarity(
            "Can I buy a laptop with this scholarship money?",
            "What is the scholarship amount?",
        )
        self.assertLess(score, 0.40)


class TestKnowledgeBaseLoading(unittest.TestCase):
    """Verify loading and schema enforcement for faq_data.json."""

    def test_load_default_faqs(self):
        data = load_faqs(DEFAULT_FAQ_PATH)
        self.assertIn("faqs", data)
        self.assertEqual(len(data["faqs"]), 6)

    def test_load_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_faqs(Path("/nonexistent/path/to/faq.json"))

    def test_invalid_json_structure_raises(self):
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmp:
            tmp.write('{"invalid": true}')
            tmp_path = Path(tmp.name)
        try:
            with self.assertRaises(ValueError):
                load_faqs(tmp_path)
        finally:
            tmp_path.unlink()

    def test_all_six_languages_defined(self):
        data = load_faqs(DEFAULT_FAQ_PATH)
        langs = data.get("languages", {})
        for code in ["en", "hi", "bn", "pa", "as", "ne"]:
            self.assertIn(code, langs)

    def test_fictional_and_draft_metadata_enforced(self):
        data = load_faqs(DEFAULT_FAQ_PATH)
        for faq in data["faqs"]:
            self.assertTrue(faq.get("fictional"), f"FAQ {faq['id']} must be marked fictional")
            self.assertEqual(
                faq.get("translation_status"),
                "draft_unverified",
                f"FAQ {faq['id']} translation must be marked draft_unverified",
            )
            # Ensure all 6 languages have answers
            answers = faq.get("answer", {})
            for code in ["en", "hi", "bn", "pa", "as", "ne"]:
                self.assertIn(code, answers)
                self.assertTrue(len(answers[code].strip()) > 0)


class TestFAQRetrieval(unittest.TestCase):
    """Verify matching across all 6 languages, unknowns, ambiguity and missing translations."""

    @classmethod
    def setUpClass(cls):
        cls.payload = load_faqs(DEFAULT_FAQ_PATH)

    def test_all_expanded_benchmark_scenarios(self):
        base_dir = Path(__file__).resolve().parent.parent
        eval_path = base_dir / "evaluation_questions_expanded.json"
        with eval_path.open("r", encoding="utf-8") as f:
            scenarios = json.load(f)

        langs = {
            "English": "en",
            "Hindi": "hi",
            "Bengali": "bn",
            "Punjabi": "pa",
            "Assamese": "as",
            "Nepali": "ne",
        }

        for item in scenarios:
            expected_topic = item["topic"]
            test_id = item["id"]
            for lang_name, code in langs.items():
                question = item.get(f"question_{code}")
                with self.subTest(test_id=test_id, language=lang_name):
                    faq, score = find_faq(question, self.payload)
                    self.assertIsNotNone(
                        faq,
                        f"Failed to match valid question for {test_id} [{lang_name}]: {question} (score={score})",
                    )
                    self.assertEqual(
                        faq["topic"],
                        expected_topic,
                        f"Incorrect topic for {test_id} [{lang_name}]: got {faq['topic']}, expected {expected_topic}",
                    )

    def test_baseline_evaluation_questions(self):
        base_dir = Path(__file__).resolve().parent.parent
        eval_path = base_dir / "evaluation_questions.json"
        with eval_path.open("r", encoding="utf-8") as f:
            scenarios = json.load(f)

        for item in scenarios:
            expected_topic = item["topic"]
            for code in ["en", "hi", "bn", "pa", "as", "ne"]:
                question = item.get(f"question_{code}")
                faq, score = find_faq(question, self.payload)
                self.assertIsNotNone(faq)
                self.assertEqual(faq["topic"], expected_topic)

    def test_unsupported_questions_return_no_match(self):
        unsupported = [
            "What is the weather in Delhi today?",
            "Who is the President of the United States?",
            "भारत का प्रधानमंत्री कौन है?",
            "আজকের আবহাওয়া কেমন?",
            "ਮੈਨੂੰ ਖਾਣਾ ਬਣਾਉਣਾ ਸਿਖਾਓ",
            "ব্ৰহ্মপুত্ৰ নদী ক’ৰ পৰা ওলাইছে?",
            "नेपालको राजधानी कहाँ छ?",
            "Is hostel accommodation provided with the scholarship?",
            "Can I buy a laptop with this scholarship money?",
            "Is there a quota for sports students?",
        ]
        for q in unsupported:
            with self.subTest(question=q):
                faq, score = find_faq(q, self.payload)
                self.assertIsNone(
                    faq,
                    f"Unsupported question was incorrectly matched to '{faq.get('topic') if faq else None}' with score {score}: {q}",
                )

    def test_ambiguous_queries_return_no_match(self):
        ambiguous = ["scholarship", "छात्रवृत्ति", "details", "help"]
        for q in ambiguous:
            with self.subTest(query=q):
                faq, score = find_faq(q, self.payload)
                self.assertIsNone(faq, f"Ambiguous query '{q}' should not match a specific topic")

    def test_missing_translation_fallback(self):
        mock_faq = {
            "id": "mock_id",
            "topic": "mock_topic",
            "answer": {"en": "English answer only."},
        }
        ans_hi = get_answer(mock_faq, "hi")
        self.assertIsNone(ans_hi)
        msg = format_fallback_message("missing_translation")
        self.assertIn("not available in the selected language", msg)

    def test_get_answer_retrieval(self):
        faq = self.payload["faqs"][0]  # required_documents
        ans_en = get_answer(faq, "en")
        self.assertIn("identity document", ans_en)
        ans_hi = get_answer(faq, "hi")
        self.assertIn("पहचान पत्र", ans_hi)


if __name__ == "__main__":
    unittest.main()
