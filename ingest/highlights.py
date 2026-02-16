from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import write_json_file
from .config import load_book_config
from .types import HighlightCandidate, to_jsonable
from .utils_paths import (
    OverwriteMode,
    check_write_allowed,
    discover_images,
    ensure_parent,
    utc_run_id,
)


class HighlightDependencyError(RuntimeError):
    pass


def _load_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise HighlightDependencyError(
            "opencv-python-headless is required for highlight detection. "
            "Install with: pip install opencv-python-headless"
        ) from exc
    return cv2


def _save_image(
    cv2: Any,
    path: Path,
    image: np.ndarray,
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
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image: {path}")


def _component_bbox(stats_row: np.ndarray) -> list[int]:
    x = int(stats_row[0])
    y = int(stats_row[1])
    w = int(stats_row[2])
    h = int(stats_row[3])
    return [x, y, x + w, y + h]


def _passes_candidate_shape_filters(
    bbox: list[int],
    *,
    page_width: int,
    page_height: int,
    edge_margin_px: int,
    max_hw_ratio: float,
    max_height_frac: float,
) -> bool:
    x1, y1, x2, y2 = bbox
    width = max(1, int(x2) - int(x1))
    height = max(1, int(y2) - int(y1))
    hw_ratio = height / width
    height_frac = height / max(1, page_height)
    near_vertical_edge = x1 <= edge_margin_px or x2 >= (page_width - edge_margin_px)

    if hw_ratio > max_hw_ratio:
        return False
    if height_frac > max_height_frac:
        return False
    if near_vertical_edge and height_frac > (max_height_frac * 0.6):
        return False
    return True


def run_detect_highlights(args) -> int:
    cv2 = _load_cv2()
    book, pipeline_config, _config_hash = load_book_config(args.book, args.pipeline)

    run_id = args.run_id or utc_run_id()
    run_root = (Path(args.runs) / run_id).resolve()

    image_paths = discover_images(book.scans_path)
    if args.max_pages is not None and args.max_pages > 0:
        image_paths = image_paths[: args.max_pages]

    cfg = pipeline_config.get("highlights", {})
    hsv_low = np.array(cfg.get("hsv_low", [15, 20, 80]), dtype=np.uint8)
    hsv_high = np.array(cfg.get("hsv_high", [95, 255, 255]), dtype=np.uint8)
    min_area = int(cfg.get("min_area", 120))
    kernel_size = int(cfg.get("kernel_size", 5))
    edge_margin_px = int(cfg.get("edge_margin_px", 25))
    max_hw_ratio = float(cfg.get("max_hw_ratio", 3.0))
    max_height_frac = float(cfg.get("max_height_frac", 0.15))
    frame_crop_frac = float(cfg.get("frame_crop_frac", 0.02))
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    processed_pages = 0
    for page_num, image_path in enumerate(image_paths, start=1):
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        page_height, page_width = bgr.shape[:2]

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_low, hsv_high)
        if frame_crop_frac > 0:
            crop = int(round(page_width * frame_crop_frac))
            if crop > 0:
                mask[:, :crop] = 0
                mask[:, page_width - crop :] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        candidates: list[HighlightCandidate] = []
        for component_idx in range(1, num_labels):
            area = int(stats[component_idx, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            bbox = _component_bbox(stats[component_idx])
            if not _passes_candidate_shape_filters(
                bbox,
                page_width=page_width,
                page_height=page_height,
                edge_margin_px=edge_margin_px,
                max_hw_ratio=max_hw_ratio,
                max_height_frac=max_height_frac,
            ):
                continue
            component_mask = labels == component_idx
            hue_values = hsv[:, :, 0][component_mask]
            sat_values = hsv[:, :, 1][component_mask]
            val_values = hsv[:, :, 2][component_mask]
            candidates.append(
                HighlightCandidate(
                    bbox=bbox,
                    area=area,
                    color_stats={
                        "h_mean": float(hue_values.mean()) if hue_values.size else 0.0,
                        "s_mean": float(sat_values.mean()) if sat_values.size else 0.0,
                        "v_mean": float(val_values.mean()) if val_values.size else 0.0,
                    },
                )
            )

        overlay = bgr.copy()
        for idx, candidate in enumerate(candidates, start=1):
            x1, y1, x2, y2 = candidate.bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                overlay,
                f"h{idx}",
                (x1, max(16, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        page_dir = run_root / f"book_{book.book_id}" / f"page_{page_num:04d}"
        _save_image(
            cv2,
            page_dir / "highlight_mask.png",
            mask,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            run_root=run_root,
        )
        _save_image(
            cv2,
            page_dir / "highlights_overlay.png",
            overlay,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            run_root=run_root,
        )
        write_json_file(
            page_dir / "highlight_candidates.json",
            {
                "book_id": book.book_id,
                "page_num": page_num,
                "scan_filename": image_path.name,
                "candidates": to_jsonable(candidates),
                "config": {
                    "hsv_low": hsv_low.tolist(),
                    "hsv_high": hsv_high.tolist(),
                    "min_area": min_area,
                    "kernel_size": kernel_size,
                    "edge_margin_px": edge_margin_px,
                    "max_hw_ratio": max_hw_ratio,
                    "max_height_frac": max_height_frac,
                    "frame_crop_frac": frame_crop_frac,
                },
            },
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            run_root=run_root,
        )
        processed_pages += 1

    print(f"Highlight detection completed for {processed_pages} pages in run {run_id}.")
    return 0
