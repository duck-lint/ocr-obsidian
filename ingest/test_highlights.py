from __future__ import annotations

import unittest

from ingest.highlights import _passes_candidate_shape_filters


class HighlightFilterTests(unittest.TestCase):
    def test_rejects_tall_skinny_edge_component(self) -> None:
        keep = _passes_candidate_shape_filters(
            [2, 10, 12, 450],
            page_width=1000,
            page_height=1400,
            edge_margin_px=25,
            max_hw_ratio=3.0,
            max_height_frac=0.15,
        )
        self.assertFalse(keep)

    def test_keeps_reasonable_horizontal_component(self) -> None:
        keep = _passes_candidate_shape_filters(
            [200, 300, 500, 350],
            page_width=1000,
            page_height=1400,
            edge_margin_px=25,
            max_hw_ratio=3.0,
            max_height_frac=0.15,
        )
        self.assertTrue(keep)


if __name__ == "__main__":
    unittest.main()
