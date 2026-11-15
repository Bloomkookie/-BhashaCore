"""Language detection and intent-routing tests across all six languages."""

import unittest

from language_utils import (
    detect_language,
    parse_target_language,
    script_purity,
    target_script,
)
from router import (
    ROUTE_CLARIFY,
    ROUTE_DOCUMENT,
    ROUTE_GENERAL,
    ROUTE_SUMMARIZE,
    ROUTE_TRANSLATION,
    route,
)


class TestLanguageDetection(unittest.TestCase):
    def test_native_sentences(self):
        cases = {
            "Why is the sky blue?": "en",
            "आसमान नीला क्यों होता है?": "hi",
            "আকাশ নীল কেন হয়?": "bn",
            "ਅਸਮਾਨ ਨੀਲਾ ਕਿਉਂ ਹੁੰਦਾ ਹੈ?": "pa",
            "আকাশখন নীলা কিয় হয়?": "as",
            "आकाश नीलो किन हुन्छ?": "ne",
        }
        for text, expected in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(detect_language(text).code, expected)

    def test_shared_script_ambiguity_is_reported(self):
        # Assamese and Bengali share a script; short sentences must not claim
        # high confidence for a guess.
        detection = detect_language("কি হৈছে?")
        self.assertIn(detection.code, {"as", "bn"})
        self.assertLessEqual(detection.confidence, 0.75)
        self.assertIn("markers", detection.evidence)

    def test_romanised_indic_is_low_confidence(self):
        detection = detect_language("scholarship ke liye kya documents chahiye")
        self.assertEqual(detection.code, "en")
        self.assertLessEqual(detection.confidence, 0.6)
        self.assertIn("manually", detection.evidence)

    def test_mixed_script_is_flagged(self):
        detection = detect_language("Python में function क्या है?")
        self.assertTrue(detection.mixed)

    def test_empty_input(self):
        self.assertEqual(detect_language("").confidence, 0.0)

    def test_script_purity(self):
        self.assertEqual(script_purity("यह एक हिंदी वाक्य है।", "hi"), 1.0)
        self.assertLess(script_purity("This is English", "hi"), 0.1)
        self.assertEqual(target_script("pa"), "gurmukhi")
        self.assertEqual(target_script("as"), "bengali")

    def test_parse_target_language(self):
        self.assertEqual(parse_target_language("Translate to Assamese: hello"), "as")
        self.assertEqual(parse_target_language("हिंदी में बताइए"), "hi")
        self.assertIsNone(parse_target_language("What is a function?"))


class TestRouting(unittest.TestCase):
    def test_general_questions_in_six_languages(self):
        cases = [
            "Why is the sky blue?",
            "आसमान नीला क्यों होता है?",
            "আকাশ নীল কেন হয়?",
            "ਅਸਮਾਨ ਨੀਲਾ ਕਿਉਂ ਹੁੰਦਾ ਹੈ?",
            "আকাশখন নীলা কিয় হয়?",
            "आकाश नीलो किन हुन्छ?",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(route(text).route, ROUTE_GENERAL)

    def test_translation_requests_in_six_languages(self):
        cases = {
            "Translate to Hindi: The district office is closed.": "hi",
            "इसे हिंदी में अनुवाद करें: The district office is closed.": "hi",
            "এই বাক্যটি বাংলায় অনুবাদ করুন: The office is closed.": "bn",
            "ਇਸ ਵਾਕ ਦਾ ਪੰਜਾਬੀ ਵਿੱਚ ਤਰਜਮਾ ਕਰੋ: The office is closed.": "pa",
            "এই বাক্যটো অসমীয়াত অনুবাদ কৰক: The office is closed.": "as",
            "यो वाक्य नेपालीमा अनुवाद गर्नुहोस्: The office is closed.": "ne",
        }
        for text, expected in cases.items():
            with self.subTest(expected=expected):
                decision = route(text)
                self.assertEqual(decision.route, ROUTE_TRANSLATION)
                self.assertEqual(decision.target_lang, expected)

    def test_summary_requests_in_six_languages(self):
        cases = [
            "Summarize this document.",
            "इस दस्तावेज़ का सारांश दें",
            "এই নথির সারসংক্ষেপ দিন",
            "ਇਸ ਦਸਤਾਵੇਜ਼ ਦਾ ਸੰਖੇਪ ਦਿਓ",
            "এই দস্তাবেজৰ সাৰাংশ দিয়ক",
            "यो कागजातको सारांश दिनुहोस्",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(route(text, has_documents=True).route, ROUTE_SUMMARIZE)

    def test_explicit_document_reference_routes_to_documents(self):
        cases = [
            "According to the uploaded file, what is the fee?",
            "अपलोड किए गए दस्तावेज़ के अनुसार फीस कितनी है?",
            "আপলোড করা ফাইলে কী লেখা আছে?",
            "ਅੱਪਲੋਡ ਕੀਤੀ ਫਾਈਲ ਅਨੁਸਾਰ ਫੀਸ ਕਿੰਨੀ ਹੈ?",
            "আপলোড কৰা ফাইলত কি আছে?",
            "अपलोड गरिएको फाइलमा के छ?",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(route(text, has_documents=True).route, ROUTE_DOCUMENT)

    def test_domain_question_with_bare_document_noun_is_not_document_qa(self):
        # Regression: Indic words for "documents" appear in FAQ questions like
        # "which documents are needed for the scholarship?" and must not be
        # mistaken for a reference to an uploaded file.
        cases = [
            "What documents are required for the scholarship?",
            "छात्रवृत्ति के लिए कौन से दस्तावेज़ चाहिए?",
            "ਵਜ਼ੀਫ਼ੇ ਲਈ ਕਿਹੜੇ ਦਸਤਾਵੇਜ਼ ਚਾਹੀਦੇ ਹਨ?",
            "বৃত্তির জন্য কোন নথি প্রয়োজন?",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(route(text, has_documents=True).route, ROUTE_GENERAL)

    def test_summary_cue_not_matched_inside_word(self):
        # Punjabi "ਸਾਰ" (summary) must not match inside "ਅਨੁਸਾਰ" (according to).
        text = "ਅੱਪਲੋਡ ਕੀਤੀ ਫਾਈਲ ਅਨੁਸਾਰ ਲਾਇਬ੍ਰੇਰੀ ਦੀ ਫੀਸ ਕਿੰਨੀ ਹੈ?"
        self.assertEqual(route(text, has_documents=True).route, ROUTE_DOCUMENT)

    def test_bare_question_words_request_clarification(self):
        for text in ["What?", "क्या?", "কী?", "ਕੀ?", "কি?", "के?"]:
            with self.subTest(text=text):
                self.assertEqual(route(text).route, ROUTE_CLARIFY)

    def test_short_greeting_is_general(self):
        self.assertEqual(route("Hello").route, ROUTE_GENERAL)

    def test_empty_message(self):
        self.assertEqual(route("   ").route, ROUTE_CLARIFY)


if __name__ == "__main__":
    unittest.main()
