"""Integration tests for app.py module exports and compatibility."""

import unittest
from pathlib import Path

import app


class TestAppIntegration(unittest.TestCase):
    """Ensure app.py maintains full backward compatibility with original exports."""

    def test_module_attributes(self):
        self.assertTrue(hasattr(app, "APP_DIR"))
        self.assertTrue(hasattr(app, "FAQ_PATH"))
        self.assertTrue(hasattr(app, "LANGUAGE_LABELS"))
        self.assertTrue(hasattr(app, "MATCH_THRESHOLD"))

    def test_module_functions(self):
        for fn_name in [
            "load_faqs",
            "normalize",
            "tokens",
            "similarity",
            "find_faq",
            "get_answer",
            "format_fallback_message",
            "main",
        ]:
            self.assertTrue(hasattr(app, fn_name), f"Missing exported function: {fn_name}")
            self.assertTrue(callable(getattr(app, fn_name)))

    def test_faq_path_points_to_valid_file(self):
        self.assertTrue(app.FAQ_PATH.exists())
        self.assertEqual(app.FAQ_PATH.name, "faq_data.json")

    def test_app_load_faqs_callable(self):
        faqs = app.load_faqs()
        self.assertIn("faqs", faqs)
        self.assertEqual(len(faqs["faqs"]), 6)


if __name__ == "__main__":
    unittest.main()
