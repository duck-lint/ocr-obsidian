from __future__ import annotations

import unittest

from ingest.text_clean import clean_ocr_lines, dehyphenate_linebreaks, reflow_paragraphs


class TextCleanTests(unittest.TestCase):
    def test_dehyphenation_merges_soft_line_break(self) -> None:
        lines = ["published pos-", "thumously."]
        merged = dehyphenate_linebreaks(lines)
        self.assertEqual(merged, ["published posthumously."])

    def test_dehyphenation_does_not_merge_into_year_block(self) -> None:
        lines = ["con-", "1770 Rousseau returns to Paris."]
        merged = dehyphenate_linebreaks(lines)
        self.assertEqual(merged, lines)

    def test_pipe_and_junk_cleanup(self) -> None:
        lines = ["|", "i|", "\\", "foo | bar", " | | "]
        cleaned = clean_ocr_lines(lines)
        self.assertIn("i", cleaned)
        self.assertIn("foo bar", cleaned)
        self.assertNotIn("|", cleaned)
        self.assertNotIn("\\", cleaned)

    def test_reflow_soft_wrapped_sentence(self) -> None:
        lines = ["This is a sentence", "broken across OCR", "lines for reading."]
        text = reflow_paragraphs(lines)
        self.assertEqual(text, "This is a sentence broken across OCR lines for reading.")

    def test_reflow_preserves_year_entries(self) -> None:
        lines = [
            "1759 Rousseau moves to Montmorency.",
            "1761 Publishes Julie, which becomes a best seller.",
        ]
        text = reflow_paragraphs(lines)
        self.assertEqual(
            text,
            "1759 Rousseau moves to Montmorency.\n\n1761 Publishes Julie, which becomes a best seller.",
        )


if __name__ == "__main__":
    unittest.main()
