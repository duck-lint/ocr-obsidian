from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .artifacts import write_json_file
from .config import load_book_config
from .textmap import load_pages_jsonl
from .types import Span, to_jsonable
from .utils_paths import (
    OverwriteMode,
    check_write_allowed,
    discover_images,
    ensure_parent,
    find_latest_run_id,
)


def _bbox_overlap(a: list[int], b: list[int]) -> int:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def _bbox_union(boxes: list[list[int]]) -> list[int]:
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def _bbox_intersection(line_bbox: list[int], trigger_bbox: list[int]) -> tuple[int, int, int]:
    x1 = max(line_bbox[0], trigger_bbox[0])
    y1 = max(line_bbox[1], trigger_bbox[1])
    x2 = min(line_bbox[2], trigger_bbox[2])
    y2 = min(line_bbox[3], trigger_bbox[3])
    if x2 <= x1 or y2 <= y1:
        return 0, 0, 0
    overlap_width = x2 - x1
    overlap_height = y2 - y1
    return overlap_width * overlap_height, overlap_width, overlap_height


def _line_matches_trigger(
    line_bbox: list[int],
    trigger_bbox: list[int],
    *,
    min_overlap_frac: float,
    min_x_overlap_px: int,
) -> bool:
    intersection_area, overlap_width, overlap_height = _bbox_intersection(line_bbox, trigger_bbox)
    if intersection_area <= 0:
        return False
    line_area = max(1, (line_bbox[2] - line_bbox[0]) * (line_bbox[3] - line_bbox[1]))
    if (intersection_area / line_area) >= min_overlap_frac:
        return True
    return overlap_width >= min_x_overlap_px and overlap_height > 0


def _select_line_indexes(
    lines: list[dict[str, Any]],
    trigger_bbox: list[int],
    *,
    min_overlap_frac: float,
    min_x_overlap_px: int,
    max_overlap_lines: int,
) -> list[int]:
    overlaps: list[int] = []
    for idx, line in enumerate(lines):
        if _line_matches_trigger(
            line["bbox"],
            trigger_bbox,
            min_overlap_frac=min_overlap_frac,
            min_x_overlap_px=min_x_overlap_px,
        ):
            overlaps.append(idx)
    if overlaps:
        if len(overlaps) > max_overlap_lines:
            trigger_center = (trigger_bbox[1] + trigger_bbox[3]) / 2.0
            nearest_idx = min(
                overlaps,
                key=lambda idx: abs(((lines[idx]["bbox"][1] + lines[idx]["bbox"][3]) / 2.0) - trigger_center),
            )
            return [nearest_idx]
        return overlaps

    trigger_center = (trigger_bbox[1] + trigger_bbox[3]) / 2.0
    nearest_idx = min(
        range(len(lines)),
        key=lambda idx: abs(((lines[idx]["bbox"][1] + lines[idx]["bbox"][3]) / 2.0) - trigger_center),
    )
    return [nearest_idx]


def _merge_raw_spans(raw_spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[tuple[str, ...], dict[str, Any]] = {}
    for span in raw_spans:
        key = tuple(span["line_ids"])
        if key not in deduped:
            deduped[key] = dict(span)
            continue
        existing = deduped[key]
        existing["trigger_bboxes"] = existing["trigger_bboxes"] + span["trigger_bboxes"]
        existing["span_bbox"] = _bbox_union([existing["span_bbox"], span["span_bbox"]])
    return list(deduped.values())


def _save_overlay(
    path: Path,
    image: Image.Image,
    *,
    dry_run: bool,
    overwrite: OverwriteMode,
    run_root: Path,
) -> None:
    check_write_allowed(path, overwrite=overwrite, dry_run=dry_run, run_root=run_root)
    ensure_parent(path, dry_run=dry_run)
    if dry_run:
        print(f"[dry-run] write image: {path}")
        return
    image.save(path)


def run_make_spans(args) -> int:
    book, pipeline_config, _config_hash = load_book_config(args.book, args.pipeline)
    runs_root = Path(args.runs)
    run_id = args.run_id or find_latest_run_id(
        runs_root, book.book_id, required_filename="highlight_candidates.json"
    )
    if not run_id:
        raise RuntimeError(
            "No run id provided and no recent run with highlight candidates was found. "
            "Pass --run-id explicitly."
        )
    run_root = (runs_root / run_id).resolve()

    corpus_root = Path(getattr(args, "corpus", "corpus"))
    pages_path = corpus_root / "books" / book.book_id / "pages.jsonl"
    pages = load_pages_jsonl(pages_path)
    page_map = {int(page["page_num"]): page for page in pages}
    image_paths = discover_images(book.scans_path)
    image_by_page = {idx: path for idx, path in enumerate(image_paths, start=1)}

    spans_cfg = pipeline_config.get("spans", {})
    arg_k_before = getattr(args, "k_before", None)
    arg_k_after = getattr(args, "k_after", None)
    k_before = int(arg_k_before if arg_k_before is not None else spans_cfg.get("k_before", 2))
    k_after = int(arg_k_after if arg_k_after is not None else spans_cfg.get("k_after", 2))
    min_overlap_frac = float(spans_cfg.get("min_overlap_frac", 0.02))
    min_x_overlap_px = int(spans_cfg.get("min_x_overlap_px", 40))
    max_overlap_lines = int(spans_cfg.get("max_overlap_lines", 8))

    page_candidate_files = sorted(run_root.glob(f"book_{book.book_id}/page_*/highlight_candidates.json"))
    if args.max_pages is not None and args.max_pages > 0:
        page_candidate_files = page_candidate_files[: args.max_pages]
    if not page_candidate_files:
        raise RuntimeError(f"No highlight candidate files found under run {run_id}.")

    processed = 0
    for candidate_file in page_candidate_files:
        payload = json.loads(candidate_file.read_text(encoding="utf-8-sig"))
        page_num = int(payload["page_num"])
        page = page_map.get(page_num)
        if page is None:
            continue
        lines: list[dict[str, Any]] = page.get("lines") or []
        if not lines:
            continue

        raw_spans: list[dict[str, Any]] = []
        for candidate in payload.get("candidates", []):
            trigger_bbox = [int(v) for v in candidate["bbox"]]
            trigger_line_indexes = _select_line_indexes(
                lines,
                trigger_bbox,
                min_overlap_frac=min_overlap_frac,
                min_x_overlap_px=min_x_overlap_px,
                max_overlap_lines=max_overlap_lines,
            )
            anchor_idx = sorted(trigger_line_indexes)[len(trigger_line_indexes) // 2]
            start_idx = max(0, anchor_idx - k_before)
            end_idx = min(len(lines) - 1, anchor_idx + k_after)
            selected = lines[start_idx : end_idx + 1]
            span_bbox = _bbox_union([line["bbox"] for line in selected])
            raw_spans.append(
                {
                    "page_num": page_num,
                    "line_ids": [line["line_id"] for line in selected],
                    "trigger_bboxes": [trigger_bbox],
                    "span_bbox": span_bbox,
                }
            )

        merged = _merge_raw_spans(raw_spans)
        merged_spans: list[Span] = []
        for span_idx, merged_span in enumerate(merged, start=1):
            merged_spans.append(
                Span(
                    span_id=f"p{page_num}_s{span_idx}",
                    page_num=page_num,
                    line_ids=merged_span["line_ids"],
                    trigger_bboxes=merged_span["trigger_bboxes"],
                    span_bbox=merged_span["span_bbox"],
                )
            )

        page_dir = candidate_file.parent
        write_json_file(
            page_dir / "spans.json",
            {
                "book_id": book.book_id,
                "page_num": page_num,
                "k_before": k_before,
                "k_after": k_after,
                "spans": to_jsonable(merged_spans),
            },
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            run_root=run_root,
        )

        image_path = image_by_page.get(page_num)
        if image_path is not None:
            with Image.open(image_path) as image:
                overlay = image.convert("RGB")
                draw = ImageDraw.Draw(overlay)
                for span in merged_spans:
                    draw.rectangle(span.span_bbox, outline=(0, 255, 0), width=3)
                    for trigger_bbox in span.trigger_bboxes:
                        draw.rectangle(trigger_bbox, outline=(255, 0, 0), width=2)
                    draw.text((span.span_bbox[0], max(0, span.span_bbox[1] - 14)), span.span_id, fill=(0, 255, 0))
                _save_overlay(
                    page_dir / "spans_overlay.png",
                    overlay,
                    dry_run=args.dry_run,
                    overwrite=args.overwrite,
                    run_root=run_root,
                )
        processed += 1

    print(f"Span generation completed for {processed} pages in run {run_id}.")
    return 0
