from __future__ import annotations

import unittest

from ingest.render_text import render_lines


def _word(text: str, conf: float = 90.0) -> dict:
    return {"text": text, "confidence": conf, "bbox": [0, 0, 0, 0]}


def _line(words: list[dict], bbox: list[int] | None = None, text: str = "") -> dict:
    return {"bbox": bbox or [0, 0, 100, 20], "words": words, "text": text}


class RenderTextTests(unittest.TestCase):
    def test_removes_pipe_only_and_single_char_junk_lines(self) -> None:
        lines = [
            _line([_word("|", 90)]),
            _line([_word("a", 95)]),
            _line([_word("i", 40)]),
            _line([_word("hello", 95), _word("world", 95)]),
        ]
        rendered = render_lines(lines)
        self.assertEqual(rendered, "a hello world")

    def test_hyphenated_line_break_joins_without_space(self) -> None:
        lines = [
            _line([_word("con-", 90)]),
            _line([_word("demned", 90), _word("in", 90), _word("Geneva.", 90)]),
        ]
        rendered = render_lines(lines)
        self.assertEqual(rendered, "condemned in Geneva.")

    def test_joins_simple_wrapped_paragraph_continuation(self) -> None:
        lines = [
            _line([_word("This", 95), _word("is", 95), _word("a", 95), _word("line", 95)]),
            _line([_word("continued", 95), _word("with", 95), _word("lowercase", 95)]),
        ]
        rendered = render_lines(lines)
        self.assertEqual(rendered, "This is a line continued with lowercase")


if __name__ == "__main__":
    unittest.main()
