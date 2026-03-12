from __future__ import annotations

import unittest

from ingest.scenarios import SCENARIOS, get_scenario, list_scenarios


class ScenarioTests(unittest.TestCase):
    def test_list_scenarios_returns_expected_names(self) -> None:
        """list_scenarios should return all expected scenario names."""
        scenarios = list_scenarios()
        expected = ["baseline", "conservative_text", "highlight_sensitive", "messy_scan_rescue"]
        self.assertEqual(sorted(scenarios), sorted(expected))

    def test_baseline_scenario_empty(self) -> None:
        """baseline scenario should have no overrides (reference point)."""
        scenario = get_scenario("baseline")
        self.assertEqual(scenario, {})

    def test_conservative_text_scenario_structure(self) -> None:
        """conservative_text scenario should have expected overrides."""
        scenario = get_scenario("conservative_text")
        # Should have QA, highlights, and spans overrides
        self.assertIn("qa.min_avg_word_conf", scenario)
        self.assertIn("qa.max_garbage_ratio", scenario)
        self.assertIn("highlights.min_area", scenario)
        self.assertIn("spans.min_overlap_frac", scenario)

    def test_conservative_text_scenario_values(self) -> None:
        """conservative_text scenario should have tighter thresholds."""
        scenario = get_scenario("conservative_text")
        # Higher confidence threshold
        self.assertEqual(scenario["qa.min_avg_word_conf"], 65.0)
        # Lower garbage tolerance
        self.assertEqual(scenario["qa.max_garbage_ratio"], 0.18)
        # Higher alpha ratio requirement
        self.assertEqual(scenario["qa.min_alpha_ratio"], 0.50)
        # Larger highlight minimum area
        self.assertEqual(scenario["highlights.min_area"], 150)
        # More overlap required
        self.assertEqual(scenario["spans.min_overlap_frac"], 0.03)

    def test_messy_scan_rescue_scenario_structure(self) -> None:
        """messy_scan_rescue scenario should have expected overrides."""
        scenario = get_scenario("messy_scan_rescue")
        # Should have QA, OCR, and highlights overrides
        self.assertIn("qa.min_avg_word_conf", scenario)
        self.assertIn("ocr.line_y_tolerance_px", scenario)
        self.assertIn("highlights.kernel_size", scenario)

    def test_messy_scan_rescue_scenario_values(self) -> None:
        """messy_scan_rescue scenario should have more forgiving thresholds."""
        scenario = get_scenario("messy_scan_rescue")
        # Lower confidence threshold (more forgiving)
        self.assertEqual(scenario["qa.min_avg_word_conf"], 48.0)
        # Higher garbage tolerance
        self.assertEqual(scenario["qa.max_garbage_ratio"], 0.28)
        # Lower alpha ratio requirement
        self.assertEqual(scenario["qa.min_alpha_ratio"], 0.38)
        # More Y tolerance for line grouping
        self.assertEqual(scenario["ocr.line_y_tolerance_px"], 18)
        # Larger morphological kernel
        self.assertEqual(scenario["highlights.kernel_size"], 7)

    def test_highlight_sensitive_scenario_structure(self) -> None:
        """highlight_sensitive scenario should have expected overrides."""
        scenario = get_scenario("highlight_sensitive")
        # Should have highlights and spans overrides
        self.assertIn("highlights.min_area", scenario)
        self.assertIn("highlights.edge_margin_px", scenario)
        self.assertIn("highlights.max_height_frac", scenario)
        self.assertIn("spans.min_x_overlap_px", scenario)

    def test_highlight_sensitive_scenario_values(self) -> None:
        """highlight_sensitive scenario should be more permissive for highlights."""
        scenario = get_scenario("highlight_sensitive")
        # Smaller minimum area (catches faint highlights)
        self.assertEqual(scenario["highlights.min_area"], 80)
        # Less edge filtering
        self.assertEqual(scenario["highlights.edge_margin_px"], 15)
        # Allow taller highlights
        self.assertEqual(scenario["highlights.max_height_frac"], 0.20)
        # Smaller kernel for finer detection
        self.assertEqual(scenario["highlights.kernel_size"], 3)
        # Less X overlap required
        self.assertEqual(scenario["spans.min_x_overlap_px"], 30)

    def test_get_scenario_unknown_raises_error(self) -> None:
        """get_scenario should raise ValueError for unknown scenario name."""
        with self.assertRaises(ValueError) as ctx:
            get_scenario("nonexistent_scenario")
        self.assertIn("Unknown scenario", str(ctx.exception))
        self.assertIn("nonexistent_scenario", str(ctx.exception))

    def test_get_scenario_returns_copy(self) -> None:
        """get_scenario should return a copy, not the original dict."""
        scenario1 = get_scenario("conservative_text")
        scenario2 = get_scenario("conservative_text")
        # Modifying one should not affect the other
        scenario1["test_key"] = "test_value"
        self.assertNotIn("test_key", scenario2)

    def test_all_scenario_keys_valid_format(self) -> None:
        """All scenario keys should use dotted notation (section.param)."""
        for name, overrides in SCENARIOS.items():
            if not overrides:  # baseline is empty
                continue
            for key in overrides.keys():
                self.assertEqual(key.count("."), 1, f"Invalid key format in {name}: {key}")
                section, param = key.split(".")
                self.assertTrue(section, f"Empty section in {name}: {key}")
                self.assertTrue(param, f"Empty param in {name}: {key}")

    def test_scenario_values_are_typed_correctly(self) -> None:
        """All scenario values should have appropriate types."""
        for name, overrides in SCENARIOS.items():
            if not overrides:
                continue
            for key, value in overrides.items():
                # Values should be int, float, str, or list (for hsv ranges)
                self.assertIn(
                    type(value),
                    [int, float, str, list],
                    f"Invalid value type in {name} for {key}: {type(value)}"
                )

    def test_scenarios_are_meaningfully_different(self) -> None:
        """Scenarios should have different configurations."""
        conservative = get_scenario("conservative_text")
        messy = get_scenario("messy_scan_rescue")
        highlight = get_scenario("highlight_sensitive")

        # Conservative and messy should have different QA thresholds
        if "qa.min_avg_word_conf" in conservative and "qa.min_avg_word_conf" in messy:
            self.assertNotEqual(
                conservative["qa.min_avg_word_conf"],
                messy["qa.min_avg_word_conf"]
            )

        # Highlight sensitive should have different highlight settings
        if "highlights.min_area" in highlight:
            # Should be different from default (120)
            self.assertNotEqual(highlight["highlights.min_area"], 120)


if __name__ == "__main__":
    unittest.main()
