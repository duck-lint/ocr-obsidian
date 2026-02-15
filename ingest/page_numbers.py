from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

from .types import OcrLine, PageRecord


ROMAN_CHARS = set("ivxlcdm")
ROMAN_STRICT_RE = re.compile(r"^m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$")
ARABIC_TOKEN_RE = re.compile(r"^\d{1,4}$")
ALNUM_TOKEN_RE = re.compile(r"([A-Za-z0-9]+)[^A-Za-z0-9]*$")
ROMAN_SUBTRACTIVES = {"iv", "ix", "xl", "xc", "cd", "cm"}
ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}


def normalize_roman(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch in ROMAN_CHARS)


def roman_to_int(s: str) -> int | None:
    normalized = normalize_roman(s)
    if not normalized:
        return None
    if not ROMAN_STRICT_RE.fullmatch(normalized):
        return None

    total = 0
    index = 0
    while index < len(normalized):
        curr = normalized[index]
        if index + 1 < len(normalized):
            nxt = normalized[index + 1]
            if ROMAN_VALUES[curr] < ROMAN_VALUES[nxt]:
                pair = curr + nxt
                if pair not in ROMAN_SUBTRACTIVES:
                    return None
                total += ROMAN_VALUES[nxt] - ROMAN_VALUES[curr]
                index += 2
                continue
        total += ROMAN_VALUES[curr]
        index += 1
    return total


def is_plausible_roman(s: str, *, min_len: int, max_value: int) -> bool:
    normalized = normalize_roman(s)
    if len(normalized) < max(1, int(min_len)):
        return False
    value = roman_to_int(normalized)
    if value is None:
        return False
    return value <= int(max_value)


def infer_scan_side(scan_relpath: str) -> Literal["left", "right", "neutral"]:
    stem = Path(scan_relpath).stem.lower()
    if stem.endswith("_l"):
        return "left"
    if stem.endswith("_r"):
        return "right"
    return "neutral"


def _center_norm(bbox: list[int], *, page_width: int, page_height: int) -> tuple[float, float]:
    width = max(1, int(page_width))
    height = max(1, int(page_height))
    x_center = (bbox[0] + bbox[2]) / 2.0
    y_center = (bbox[1] + bbox[3]) / 2.0
    return (x_center / width, y_center / height)


def _is_preferred_region(x_center_norm: float, side: Literal["left", "right", "neutral"]) -> bool:
    if side == "left":
        return x_center_norm < 0.35
    if side == "right":
        return x_center_norm > 0.65
    return True


def _edge_score(x_center_norm: float, side: Literal["left", "right", "neutral"]) -> float:
    if side == "left":
        return 1.0 - x_center_norm
    if side == "right":
        return x_center_norm
    return max(x_center_norm, 1.0 - x_center_norm)


def _extract_terminal_line_token(line: OcrLine) -> str | None:
    if line.words:
        return line.words[-1].text
    match = ALNUM_TOKEN_RE.search(line.text.strip())
    if not match:
        return None
    return match.group(1)


def _candidate_record(
    *,
    text: str,
    confidence: float,
    bbox: list[int],
    source: str,
    line_id: str | None,
    page_width: int,
    page_height: int,
    side: Literal["left", "right", "neutral"],
) -> dict[str, Any]:
    x_center_norm, y_center_norm = _center_norm(bbox, page_width=page_width, page_height=page_height)
    preferred = _is_preferred_region(x_center_norm, side)
    return {
        "text": text,
        "conf": float(confidence),
        "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
        "x_center_norm": x_center_norm,
        "y_center_norm": y_center_norm,
        "source": source,
        "line_id": line_id,
        "_preferred": preferred,
        "_edge_score": _edge_score(x_center_norm, side),
    }


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        0 if candidate["_preferred"] else 1,
        -float(candidate["_edge_score"]),
        -float(candidate["conf"]),
        float(candidate["y_center_norm"]),
        str(candidate["text"]),
        "" if candidate.get("line_id") is None else str(candidate["line_id"]),
        str(candidate["source"]),
        tuple(int(v) for v in candidate["bbox"]),
    )


def _strip_internal_fields(candidate: dict[str, Any]) -> dict[str, Any]:
    payload = dict(candidate)
    payload.pop("_preferred", None)
    payload.pop("_edge_score", None)
    return payload


def detect_printed_page(
    page: PageRecord,
    *,
    page_width: int,
    page_height: int,
    top_band_frac: float = 0.12,
    min_conf: float = 40.0,
    roman_min_len: int = 2,
    roman_max_value: int = 80,
    side: Literal["left", "right", "neutral"] = "neutral",
    max_top_lines: int = 5,
    debug: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    top_band_limit = max(0.0, float(top_band_frac)) * max(1, int(page_height))
    min_confidence = float(min_conf)
    candidates: list[dict[str, Any]] = []

    for word in page.words:
        x_norm, y_norm = _center_norm(word.bbox, page_width=page_width, page_height=page_height)
        if (y_norm * page_height) > top_band_limit:
            continue
        candidates.append(
            _candidate_record(
                text=word.text,
                confidence=word.confidence,
                bbox=word.bbox,
                source="word",
                line_id=None,
                page_width=page_width,
                page_height=page_height,
                side=side,
            )
        )

    sorted_lines = sorted(
        page.lines,
        key=lambda line: (
            _center_norm(line.bbox, page_width=page_width, page_height=page_height)[1],
            line.bbox[0],
            line.line_id,
        ),
    )
    for line in sorted_lines[: max(0, int(max_top_lines))]:
        _x_norm, y_norm = _center_norm(line.bbox, page_width=page_width, page_height=page_height)
        if (y_norm * page_height) > top_band_limit:
            continue
        token = _extract_terminal_line_token(line)
        if not token:
            continue
        confidence = float(line.words[-1].confidence) if line.words else 0.0
        candidates.append(
            _candidate_record(
                text=token,
                confidence=confidence,
                bbox=line.bbox,
                source="line",
                line_id=line.line_id,
                page_width=page_width,
                page_height=page_height,
                side=side,
            )
        )

    arabic_candidates = [
        candidate
        for candidate in candidates
        if float(candidate["conf"]) >= min_confidence and ARABIC_TOKEN_RE.fullmatch(str(candidate["text"]))
    ]
    arabic_ranked = sorted(arabic_candidates, key=_candidate_sort_key)
    selected_arabic = arabic_ranked[0] if arabic_ranked else None
    if selected_arabic is not None:
        result = {
            "printed_page": int(str(selected_arabic["text"])),
            "printed_page_text": str(selected_arabic["text"]),
            "printed_page_kind": "arabic",
        }
        debug_payload = {}
        if debug:
            debug_payload = {
                "selected": _strip_internal_fields(selected_arabic),
                "selected_kind": "arabic",
                "arabic_top_candidates": [_strip_internal_fields(c) for c in arabic_ranked[:10]],
                "roman_top_candidates": [],
            }
        return result, debug_payload

    roman_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        if float(candidate["conf"]) < min_confidence:
            continue
        normalized = normalize_roman(str(candidate["text"]))
        if len(normalized) < int(roman_min_len):
            continue
        value = roman_to_int(normalized)
        if value is None or value > int(roman_max_value):
            continue
        candidate_with_value = dict(candidate)
        candidate_with_value["_roman_value"] = value
        roman_candidates.append(candidate_with_value)

    roman_ranked = sorted(roman_candidates, key=_candidate_sort_key)
    selected_roman = roman_ranked[0] if roman_ranked else None
    if selected_roman is not None:
        result = {
            "printed_page": int(selected_roman["_roman_value"]),
            "printed_page_text": str(selected_roman["text"]),
            "printed_page_kind": "roman",
        }
        debug_payload = {}
        if debug:
            debug_payload = {
                "selected": _strip_internal_fields(selected_roman),
                "selected_kind": "roman",
                "arabic_top_candidates": [],
                "roman_top_candidates": [_strip_internal_fields(c) for c in roman_ranked[:10]],
            }
        return result, debug_payload

    debug_payload = {}
    if debug:
        debug_payload = {
            "selected": None,
            "selected_kind": None,
            "arabic_top_candidates": [],
            "roman_top_candidates": [],
        }
    return {"printed_page": None, "printed_page_text": None, "printed_page_kind": None}, debug_payload


def apply_printed_page_mode(
    result: dict[str, Any],
    mode: Literal["auto", "arabic"],
    *,
    arabic_switch_min: int,
) -> tuple[dict[str, Any], Literal["auto", "arabic"]]:
    current_mode: Literal["auto", "arabic"] = "arabic" if mode == "arabic" else "auto"
    next_mode: Literal["auto", "arabic"] = current_mode
    applied = dict(result)

    if current_mode == "arabic" and applied.get("printed_page_kind") == "roman":
        applied["printed_page"] = None
        applied["printed_page_text"] = None
        applied["printed_page_kind"] = None
        return applied, next_mode

    if applied.get("printed_page_kind") == "arabic":
        page_value = applied.get("printed_page")
        if isinstance(page_value, int) and page_value >= int(arabic_switch_min):
            next_mode = "arabic"
    return applied, next_mode
