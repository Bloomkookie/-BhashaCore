"""Document ingestion, metadata, and retrieval tests (no network required)."""

import unittest

from documents import DocumentStore


class TestIngestion(unittest.TestCase):
    def setUp(self):
        self.store = DocumentStore(use_embeddings=False)

    def test_txt_ingestion(self):
        result = self.store.add_document("policy.txt", b"Membership fee is INR 200 per year.")
        self.assertIsNone(result.error)
        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(result.chunks[0].lang, "en")
        self.assertEqual(result.chunks[0].doc_name, "policy.txt")
        self.assertIn("fee", self.store.summary_context())

    def test_markdown_ingestion(self):
        text = "# Rules\n\nMembership fee is INR 200 per year.\n\n## Borrowing\n\nFour books for 14 days."
        result = self.store.add_document("rules.md", text.encode())
        self.assertIsNone(result.error)
        self.assertTrue(result.chunks)

    def test_json_ingestion_keeps_sections(self):
        payload = b'{"scheme": {"documents": "Identity card and bank details", "fee": "INR 200"}}'
        result = self.store.add_document("scheme.json", payload)
        self.assertIsNone(result.error)
        sections = {chunk.section for chunk in result.chunks}
        self.assertIn("scheme", sections)

    def test_csv_ingestion(self):
        payload = b"topic,detail\nfee,INR 200\nloan,four books\n"
        result = self.store.add_document("data.csv", payload)
        self.assertIsNone(result.error)
        self.assertIn("INR 200", self.store.summary_context())

    def test_duplicate_file_is_reused(self):
        first = self.store.add_document("policy.txt", b"Membership fee is INR 200 per year.")
        second = self.store.add_document("policy.txt", b"Membership fee is INR 200 per year.")
        self.assertEqual(first.doc.doc_id, second.doc.doc_id)
        self.assertTrue(second.warnings)
        self.assertEqual(self.store.stats()["documents"], 1)

    def test_unsupported_and_empty_files_are_rejected(self):
        bad = self.store.add_document("image.png", b"not really a png")
        self.assertIsNotNone(bad.error)
        self.assertIn("Unsupported", bad.error)
        empty = self.store.add_document("empty.txt", b"")
        self.assertIsNotNone(empty.error)

    def test_page_and_section_metadata_reach_citations(self):
        result = self.store.add_document(
            "policy.pdf",
            b"synthetic",
            sources=[("Fees", 3, "Membership fee is INR 200 per year.")],
        )
        self.assertIsNone(result.error)
        retrieval = self.store.search("membership fee")
        self.assertTrue(retrieval.evidence_found)
        citations = self.store.build_citations(retrieval.chunks, {1})
        self.assertEqual(citations[0].page, 3)
        self.assertEqual(citations[0].section, "Fees")
        self.assertIn("p. 3", citations[0].label())
        self.assertTrue(citations[0].cited)

    def test_clear_documents(self):
        self.store.add_document("policy.txt", b"Membership fee is INR 200 per year.")
        self.store.clear()
        self.assertEqual(self.store.stats()["documents"], 0)
        self.assertFalse(self.store.search("membership fee").evidence_found)


class TestRetrieval(unittest.TestCase):
    def setUp(self):
        self.store = DocumentStore(use_embeddings=False)
        self.store.add_document(
            "library.txt",
            b"Library membership fee is INR 200 per year. Borrowing limit is four books "
            b"for 14 days. Late return fine is INR 5 per day.",
        )

    def test_relevant_query_finds_evidence(self):
        result = self.store.search("What is the library membership fee?")
        self.assertTrue(result.evidence_found, result.best_lexical)
        self.assertEqual(result.method, "lexical")
        self.assertTrue(result.chunks)

    def test_unrelated_query_is_not_grounded(self):
        # Regression: a single-chunk corpus used to score 1.0 for any shared
        # stopword. IDF-weighted coverage plus a stricter gate prevents that.
        result = self.store.search("Who is the library director?")
        self.assertFalse(result.evidence_found, (result.best_lexical, result.best_semantic))

    def test_empty_store(self):
        empty = DocumentStore(use_embeddings=False)
        result = empty.search("anything")
        self.assertFalse(result.evidence_found)
        self.assertEqual(result.chunks, [])


if __name__ == "__main__":
    unittest.main()
