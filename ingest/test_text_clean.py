from __future__ import annotations

import unittest

from ingest.text_clean import clean_ocr_text


class TextCleanTests(unittest.TestCase):
    def test_token_junk_removal(self) -> None:
        text = clean_ocr_text(["maréchal de Luxembourg. tH! )"])
        self.assertIn("maréchal de Luxembourg.", text)
        self.assertNotIn("tH!", text)
        self.assertNotIn(")", text)

    def test_pipe_junk_removal(self) -> None:
        text = clean_ocr_text(["| 1762 Publishes the Social Contract |", "| |"])
        self.assertIn("1762 Publishes the Social Contract", text)
        self.assertNotIn("|", text)

    def test_hyphenation_across_junk(self) -> None:
        text = clean_ocr_text(["Both are con- )", "i", "demned in Geneva."])
        self.assertIn("Both are condemned in Geneva.", text)
        self.assertNotIn("\ni\n", text)

    def test_prose_reflow(self) -> None:
        text = clean_ocr_text(["This is a sentence", "broken across lines", "because OCR."])
        self.assertEqual(text, "This is a sentence broken across lines because OCR.")

    def test_list_preservation(self) -> None:
        text = clean_ocr_text(["1759 Rousseau moves...", "near the home...", "1761 Publishes Julie..."])
        self.assertEqual(
            text,
            "1759 Rousseau moves... near the home...\n\n1761 Publishes Julie...",
        )


if __name__ == "__main__":
    unittest.main()
