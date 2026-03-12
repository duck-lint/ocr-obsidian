"""
Tuning scenario definitions for OCR pipeline.

Each scenario provides a named set of configuration overrides
for testing different OCR/highlight/QA/spans settings.
"""

from __future__ import annotations

from typing import Any


# Scenario definitions: name -> dict[dotted_path, value]
SCENARIOS: dict[str, dict[str, Any]] = {
    "baseline": {
        # Current defaults - no overrides
        # Useful as a reference point
    },
    "conservative_text": {
        # Prefer cleaner text and fewer false positives
        # Tighter QA thresholds and less permissive grouping
        "qa.min_avg_word_conf": 65.0,  # Higher confidence threshold
        "qa.max_garbage_ratio": 0.18,  # Less garbage tolerated
        "qa.min_alpha_ratio": 0.50,  # More alpha characters required
        "highlights.min_area": 150,  # Larger highlights only
        "spans.min_overlap_frac": 0.03,  # More overlap required for matching
    },
    "messy_scan_rescue": {
        # More forgiving settings for noisy, low-confidence, or uneven scans
        # Lower thresholds to capture more marginal content
        "qa.min_avg_word_conf": 48.0,  # Lower confidence threshold
        "qa.max_garbage_ratio": 0.28,  # More garbage tolerated
        "qa.min_alpha_ratio": 0.38,  # Fewer alpha characters required
        "ocr.line_y_tolerance_px": 18,  # More tolerance for line grouping
        "highlights.kernel_size": 7,  # Larger morphological kernel for noise
    },
    "highlight_sensitive": {
        # More permissive highlight detection for faint or partial highlighting
        # Useful for scans with weak highlight markers
        "highlights.min_area": 80,  # Smaller highlights accepted
        "highlights.edge_margin_px": 15,  # Less aggressive edge filtering
        "highlights.max_height_frac": 0.20,  # Allow taller highlights
        "highlights.kernel_size": 3,  # Smaller kernel for finer detection
        "spans.min_x_overlap_px": 30,  # Less X overlap required
    },
}


def get_scenario(name: str) -> dict[str, Any]:
    """
    Get configuration overrides for a named scenario.

    Args:
        name: Scenario name (e.g., "baseline", "conservative_text")

    Returns:
        Dictionary mapping dotted config paths to override values

    Raises:
        ValueError: If scenario name is not found
    """
    if name not in SCENARIOS:
        available = ", ".join(sorted(SCENARIOS.keys()))
        raise ValueError(f"Unknown scenario '{name}'. Available scenarios: {available}")
    return dict(SCENARIOS[name])


def list_scenarios() -> list[str]:
    """Return list of available scenario names."""
    return sorted(SCENARIOS.keys())
