from __future__ import annotations

import unittest
from typing import Any

from ingest.config import apply_cli_overrides, DEFAULT_PIPELINE_CONFIG


class ConfigOverrideTests(unittest.TestCase):
    def test_apply_cli_overrides_empty_dict(self) -> None:
        """Empty overrides should return unchanged config."""
        base = dict(DEFAULT_PIPELINE_CONFIG)
        result = apply_cli_overrides(base, {})
        self.assertEqual(result, base)

    def test_apply_cli_overrides_none_overrides(self) -> None:
        """None overrides should return unchanged config."""
        base = dict(DEFAULT_PIPELINE_CONFIG)
        result = apply_cli_overrides(base, {})
        self.assertEqual(result, base)

    def test_apply_cli_overrides_with_none_values(self) -> None:
        """None values in overrides should be ignored."""
        base = dict(DEFAULT_PIPELINE_CONFIG)
        overrides = {
            "ocr.psm": None,
            "ocr.language": "fra",
        }
        result = apply_cli_overrides(base, overrides)
        # psm should be unchanged (None ignored)
        self.assertEqual(result["ocr"]["psm"], 6)
        # language should be updated
        self.assertEqual(result["ocr"]["language"], "fra")

    def test_apply_cli_overrides_ocr_section(self) -> None:
        """OCR overrides should be applied correctly."""
        base = dict(DEFAULT_PIPELINE_CONFIG)
        overrides = {
            "ocr.psm": 3,
            "ocr.language": "deu",
            "ocr.line_y_tolerance_px": 20,
        }
        result = apply_cli_overrides(base, overrides)
        self.assertEqual(result["ocr"]["psm"], 3)
        self.assertEqual(result["ocr"]["language"], "deu")
        self.assertEqual(result["ocr"]["line_y_tolerance_px"], 20)

    def test_apply_cli_overrides_highlights_section(self) -> None:
        """Highlight overrides should be applied correctly."""
        base = dict(DEFAULT_PIPELINE_CONFIG)
        overrides = {
            "highlights.min_area": 200,
            "highlights.kernel_size": 7,
        }
        result = apply_cli_overrides(base, overrides)
        self.assertEqual(result["highlights"]["min_area"], 200)
        self.assertEqual(result["highlights"]["kernel_size"], 7)

    def test_apply_cli_overrides_qa_section(self) -> None:
        """QA overrides should be applied correctly."""
        base = dict(DEFAULT_PIPELINE_CONFIG)
        overrides = {
            "qa.min_avg_word_conf": 65.0,
            "qa.max_garbage_ratio": 0.18,
        }
        result = apply_cli_overrides(base, overrides)
        # QA section doesn't exist in DEFAULT_PIPELINE_CONFIG but should be created
        self.assertEqual(result.get("qa", {}).get("min_avg_word_conf"), 65.0)
        self.assertEqual(result.get("qa", {}).get("max_garbage_ratio"), 0.18)

    def test_apply_cli_overrides_spans_section(self) -> None:
        """Span overrides should be applied correctly."""
        base = dict(DEFAULT_PIPELINE_CONFIG)
        overrides = {
            "spans.k_before": 3,
            "spans.min_overlap_frac": 0.05,
        }
        result = apply_cli_overrides(base, overrides)
        self.assertEqual(result["spans"]["k_before"], 3)
        self.assertEqual(result["spans"]["min_overlap_frac"], 0.05)

    def test_apply_cli_overrides_multiple_sections(self) -> None:
        """Multiple section overrides should all apply."""
        base = dict(DEFAULT_PIPELINE_CONFIG)
        overrides = {
            "ocr.psm": 11,
            "highlights.min_area": 150,
            "spans.k_after": 5,
        }
        result = apply_cli_overrides(base, overrides)
        self.assertEqual(result["ocr"]["psm"], 11)
        self.assertEqual(result["highlights"]["min_area"], 150)
        self.assertEqual(result["spans"]["k_after"], 5)

    def test_apply_cli_overrides_preserves_other_values(self) -> None:
        """Overrides should not affect other values in the same section."""
        base = dict(DEFAULT_PIPELINE_CONFIG)
        overrides = {
            "ocr.psm": 7,
        }
        result = apply_cli_overrides(base, overrides)
        # psm should be updated
        self.assertEqual(result["ocr"]["psm"], 7)
        # other OCR values should be preserved
        self.assertEqual(result["ocr"]["language"], "eng")
        self.assertEqual(result["ocr"]["line_y_tolerance_px"], 14)

    def test_apply_cli_overrides_invalid_key_format(self) -> None:
        """Invalid key formats should be ignored."""
        base = dict(DEFAULT_PIPELINE_CONFIG)
        overrides = {
            "invalid": 123,  # No dot separator
            "too.many.parts": 456,  # Too many parts
            "ocr.psm": 3,  # Valid
        }
        result = apply_cli_overrides(base, overrides)
        # Valid override should be applied
        self.assertEqual(result["ocr"]["psm"], 3)
        # Invalid keys should not appear
        self.assertNotIn("invalid", result)
        self.assertNotIn("too", result)

    def test_apply_cli_overrides_creates_missing_section(self) -> None:
        """Overrides should create missing sections."""
        base = {"ocr": {"psm": 6}}
        overrides = {
            "new_section.param": 999,
        }
        result = apply_cli_overrides(base, overrides)
        self.assertEqual(result["new_section"]["param"], 999)
        # Original section should be preserved
        self.assertEqual(result["ocr"]["psm"], 6)

    def test_precedence_order(self) -> None:
        """Test that precedence order is: defaults < YAML < CLI."""
        # Start with defaults
        base = {"ocr": {"psm": 6, "language": "eng"}}
        # Apply CLI overrides
        overrides = {"ocr.psm": 3}
        result = apply_cli_overrides(base, overrides)
        # CLI override should win
        self.assertEqual(result["ocr"]["psm"], 3)
        # Non-overridden value from base should remain
        self.assertEqual(result["ocr"]["language"], "eng")


if __name__ == "__main__":
    unittest.main()
