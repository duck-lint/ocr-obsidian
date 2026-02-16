from __future__ import annotations

import unittest

from ingest.spans import _select_line_indexes


class SpanSelectionTests(unittest.TestCase):
    def test_vertical_stripe_trigger_does_not_select_all_lines(self) -> None:
        lines = [
            {"bbox": [100, 10, 500, 30], "line_id": "l1"},
            {"bbox": [100, 40, 500, 60], "line_id": "l2"},
            {"bbox": [100, 70, 500, 90], "line_id": "l3"},
            {"bbox": [100, 100, 500, 120], "line_id": "l4"},
            {"bbox": [100, 130, 500, 150], "line_id": "l5"},
        ]
        vertical_edge_trigger = [0, 0, 12, 200]
        indexes = _select_line_indexes(
            lines,
            vertical_edge_trigger,
            min_overlap_frac=0.02,
            min_x_overlap_px=40,
            max_overlap_lines=8,
        )
        self.assertLess(len(indexes), len(lines))
        self.assertEqual(len(indexes), 1)


if __name__ == "__main__":
    unittest.main()
