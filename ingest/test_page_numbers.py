from __future__ import annotations

import unittest

from ingest.page_numbers import (
    apply_printed_page_mode,
    detect_printed_page,
    infer_scan_side,
    is_plausible_roman,
    roman_to_int,
)
from ingest.types import OcrLine, OcrWord, PageRecord


def _word(text: str, bbox: list[int], conf: float = 95.0) -> OcrWord:
    return OcrWord(text=text, bbox=bbox, confidence=conf)


def _line(line_id: str, words: list[OcrWord]) -> OcrLine:
    x1 = min(w.bbox[0] for w in words)
    y1 = min(w.bbox[1] for w in words)
    x2 = max(w.bbox[2] for w in words)
    y2 = max(w.bbox[3] for w in words)
    return OcrLine(
        line_id=line_id,
        bbox=[x1, y1, x2, y2],
        words=words,
        text=" ".join(w.text for w in words),
    )


def _page(scan_relpath: str, words: list[OcrWord], lines: list[OcrLine]) -> PageRecord:
    return PageRecord(
        book_id="book1",
        page_num=1,
        scan_relpath=scan_relpath,
        ocr_engine="tesseract+pytesseract",
        config={},
        words=words,
        lines=lines,
    )


class PageNumberTests(unittest.TestCase):
    def test_roman_to_int_strict(self) -> None:
        self.assertEqual(roman_to_int("xiv"), 14)
        self.assertEqual(roman_to_int("XXXV"), 35)
        self.assertIsNone(roman_to_int("iix"))
        self.assertIsNone(roman_to_int("vx"))
        self.assertTrue(is_plausible_roman("xiv", min_len=2, max_value=80))
        self.assertFalse(is_plausible_roman("m", min_len=2, max_value=80))

    def test_detect_roman_line_tail(self) -> None:
        words = [
            _word("Introduction", [180, 20, 360, 56], 92.0),
            _word("xiv", [930, 20, 980, 56], 89.0),
        ]
        lines = [_line("p1_l1", words)]
        page = _page("scan_0001.png", words, lines)
        result, _debug = detect_printed_page(page, page_width=1000, page_height=1400, side="neutral")
        self.assertEqual(result["printed_page"], 14)
        self.assertEqual(result["printed_page_text"], "xiv")
        self.assertEqual(result["printed_page_kind"], "roman")

    def test_detect_embedded_roman_terminal(self) -> None:
        words = [
            _word("Chronology", [100, 24, 330, 60], 93.0),
            _word("...", [340, 24, 430, 60], 88.0),
            _word("XXXV", [910, 24, 980, 60], 91.0),
        ]
        lines = [_line("p1_l1", words)]
        page = _page("scan_0002.png", words, lines)
        result, _debug = detect_printed_page(page, page_width=1000, page_height=1400, side="neutral")
        self.assertEqual(result["printed_page"], 35)
        self.assertEqual(result["printed_page_text"], "XXXV")
        self.assertEqual(result["printed_page_kind"], "roman")

    def test_detect_arabic_top_right(self) -> None:
        words = [_word("122", [940, 20, 990, 52], 95.0)]
        lines = [_line("p1_l1", words)]
        page = _page("scan_0003_R.png", words, lines)
        side = infer_scan_side(page.scan_relpath)
        result, _debug = detect_printed_page(page, page_width=1000, page_height=1200, side=side)
        self.assertEqual(result["printed_page"], 122)
        self.assertEqual(result["printed_page_text"], "122")
        self.assertEqual(result["printed_page_kind"], "arabic")

    def test_reject_single_letter_roman(self) -> None:
        words = [_word("m", [960, 30, 990, 60], 99.0)]
        lines = [_line("p1_l1", words)]
        page = _page("scan_0004.png", words, lines)
        result, _debug = detect_printed_page(
            page,
            page_width=1000,
            page_height=1200,
            side="neutral",
            roman_min_len=2,
            roman_max_value=80,
        )
        self.assertIsNone(result["printed_page"])
        self.assertIsNone(result["printed_page_text"])
        self.assertIsNone(result["printed_page_kind"])

    def test_side_preference_determinism(self) -> None:
        words = [
            _word("14", [40, 20, 90, 52], 90.0),
            _word("14", [900, 20, 950, 52], 90.0),
        ]
        lines = [_line("p1_l1", words)]

        left_page = _page("scan_0005_L.png", words, lines)
        left_result, left_debug = detect_printed_page(
            left_page,
            page_width=1000,
            page_height=1200,
            side=infer_scan_side(left_page.scan_relpath),
            debug=True,
        )
        self.assertEqual(left_result["printed_page"], 14)
        self.assertEqual(left_result["printed_page_text"], "14")
        self.assertLess(float(left_debug["selected"]["x_center_norm"]), 0.35)

        right_page = _page("scan_0005_R.png", words, lines)
        right_result, right_debug = detect_printed_page(
            right_page,
            page_width=1000,
            page_height=1200,
            side=infer_scan_side(right_page.scan_relpath),
            debug=True,
        )
        self.assertEqual(right_result["printed_page"], 14)
        self.assertEqual(right_result["printed_page_text"], "14")
        self.assertGreater(float(right_debug["selected"]["x_center_norm"]), 0.65)

    def test_mode_switch_ignores_later_roman(self) -> None:
        mode = "auto"
        accepted_arabic, mode = apply_printed_page_mode(
            {"printed_page": 12, "printed_page_text": "12", "printed_page_kind": "arabic"},
            mode,
            arabic_switch_min=10,
        )
        self.assertEqual(accepted_arabic["printed_page"], 12)
        self.assertEqual(mode, "arabic")

        roman_after_switch, mode = apply_printed_page_mode(
            {"printed_page": 14, "printed_page_text": "xiv", "printed_page_kind": "roman"},
            mode,
            arabic_switch_min=10,
        )
        self.assertIsNone(roman_after_switch["printed_page"])
        self.assertIsNone(roman_after_switch["printed_page_text"])
        self.assertIsNone(roman_after_switch["printed_page_kind"])
        self.assertEqual(mode, "arabic")

    def test_debug_determinism(self) -> None:
        words = [
            _word("122", [940, 20, 990, 52], 95.0),
            _word("14", [40, 20, 90, 52], 90.0),
        ]
        lines = [_line("p1_l1", words)]
        page = _page("scan_0006_R.png", words, lines)
        kwargs = {
            "page_width": 1000,
            "page_height": 1200,
            "side": infer_scan_side(page.scan_relpath),
            "debug": True,
        }
        result_a, debug_a = detect_printed_page(page, **kwargs)
        result_b, debug_b = detect_printed_page(page, **kwargs)
        self.assertEqual(result_a, result_b)
        self.assertEqual(debug_a, debug_b)


if __name__ == "__main__":
    unittest.main()
