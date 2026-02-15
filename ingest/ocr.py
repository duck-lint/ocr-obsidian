from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .artifacts import write_json_file, write_jsonl_records
from .config import load_book_config
from .page_numbers import apply_printed_page_mode, detect_printed_page, infer_scan_side
from .types import OcrLine, OcrWord, PageRecord, to_jsonable
from .utils_paths import (
    OverwriteMode,
    check_write_allowed,
    discover_images,
    ensure_parent,
    safe_relpath,
    utc_run_id,
)


class OcrDependencyError(RuntimeError):
    pass


def _load_pytesseract() -> Any:
    try:
        import pytesseract
    except ImportError as exc:
        raise OcrDependencyError(
            "pytesseract is required for OCR. Install with: pip install pytesseract"
        ) from exc

    if shutil.which("tesseract") is None:
        raise OcrDependencyError(
            "The tesseract binary was not found in PATH. Install Tesseract OCR and retry."
        )
    return pytesseract


def _bbox_from_ltrb(left: int, top: int, width: int, height: int) -> list[int]:
    return [left, top, left + width, top + height]


def _extract_words(image: Image.Image, pytesseract: Any, language: str, psm: int) -> list[OcrWord]:
    raw = pytesseract.image_to_data(
        image,
        lang=language,
        config=f"--psm {psm}",
        output_type=pytesseract.Output.DICT,
    )
    words: list[OcrWord] = []
    total = len(raw.get("text", []))
    for index in range(total):
        text = str(raw["text"][index]).strip()
        if not text:
            continue
        try:
            confidence = float(raw["conf"][index])
        except (TypeError, ValueError):
            continue
        if confidence < 0:
            continue

        left = int(raw["left"][index])
        top = int(raw["top"][index])
        width = int(raw["width"][index])
        height = int(raw["height"][index])
        if width <= 0 or height <= 0:
            continue
        words.append(
            OcrWord(
                text=text,
                bbox=_bbox_from_ltrb(left=left, top=top, width=width, height=height),
                confidence=confidence,
            )
        )
    return words


def _y_center(bbox: list[int]) -> float:
    return (bbox[1] + bbox[3]) / 2.0


def _merge_bbox(boxes: list[list[int]]) -> list[int]:
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def _group_lines(words: list[OcrWord], page_num: int, y_tolerance_px: int) -> list[OcrLine]:
    if not words:
        return []

    sorted_words = sorted(words, key=lambda w: (_y_center(w.bbox), w.bbox[0]))
    clusters: list[dict[str, Any]] = []
    for word in sorted_words:
        center_y = _y_center(word.bbox)
        matched = False
        for cluster in clusters:
            if abs(center_y - cluster["center_y"]) <= y_tolerance_px:
                cluster["words"].append(word)
                cluster["center_y"] = sum(_y_center(w.bbox) for w in cluster["words"]) / len(cluster["words"])
                matched = True
                break
        if not matched:
            clusters.append({"center_y": center_y, "words": [word]})

    clusters.sort(key=lambda c: c["center_y"])
    lines: list[OcrLine] = []
    for index, cluster in enumerate(clusters, start=1):
        cluster_words: list[OcrWord] = sorted(cluster["words"], key=lambda w: w.bbox[0])
        line_text = " ".join(word.text for word in cluster_words)
        line_bbox = _merge_bbox([word.bbox for word in cluster_words])
        lines.append(
            OcrLine(
                line_id=f"p{page_num}_l{index}",
                bbox=line_bbox,
                words=cluster_words,
                text=line_text,
            )
        )
    return lines


def _draw_overlay(base_image: Image.Image, words: list[OcrWord], lines: list[OcrLine]) -> Image.Image:
    overlay = base_image.convert("RGB")
    draw = ImageDraw.Draw(overlay)

    for word in words:
        draw.rectangle(word.bbox, outline=(60, 160, 255), width=1)
    for line in lines:
        draw.rectangle(line.bbox, outline=(255, 80, 80), width=2)
        draw.text((line.bbox[0], max(0, line.bbox[1] - 12)), line.line_id, fill=(255, 80, 80))

    return overlay


def _save_overlay_image(
    path: Path,
    image: Image.Image,
    *,
    dry_run: bool,
    overwrite: OverwriteMode,
    run_root: Path | None,
) -> None:
    check_write_allowed(path, overwrite=overwrite, dry_run=dry_run, run_root=run_root)
    ensure_parent(path, dry_run=dry_run)
    if dry_run:
        print(f"[dry-run] write image: {path}")
        return
    image.save(path)


def run_ocr(args) -> int:
    pytesseract = _load_pytesseract()
    book, pipeline_config, config_hash = load_book_config(args.book, args.pipeline)

    run_id = args.run_id or utc_run_id()
    run_root = (Path(args.runs) / run_id).resolve()
    out_root = Path(args.out)

    image_paths = discover_images(book.scans_path)
    if args.max_pages is not None and args.max_pages > 0:
        image_paths = image_paths[: args.max_pages]
    if not image_paths:
        raise RuntimeError("No pages selected for OCR.")

    ocr_cfg = pipeline_config.get("ocr", {})
    language = str(ocr_cfg.get("language", "eng"))
    psm = int(ocr_cfg.get("psm", 6))
    y_tolerance_px = int(ocr_cfg.get("line_y_tolerance_px", 14))
    printed_page_detect = bool(getattr(args, "printed_page_detect", True))
    printed_page_top_band_frac = float(getattr(args, "printed_page_top_band_frac", 0.12))
    printed_page_min_conf = float(getattr(args, "printed_page_min_conf", 40.0))
    printed_page_roman_max = int(getattr(args, "printed_page_roman_max", 80))
    printed_page_roman_min_len = int(getattr(args, "printed_page_roman_min_len", 2))
    printed_page_arabic_switch_min = int(getattr(args, "printed_page_arabic_switch_min", 10))
    printed_page_debug = bool(getattr(args, "printed_page_debug", False))
    printed_page_mode: str = "auto"

    page_records: list[dict[str, Any]] = []
    for page_num, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            page_width, page_height = image.size
            words = _extract_words(image=image, pytesseract=pytesseract, language=language, psm=psm)
            lines = _group_lines(words=words, page_num=page_num, y_tolerance_px=y_tolerance_px)
            page_text = "\n".join(line.text for line in lines if line.text.strip())
            scan_relpath = safe_relpath(image_path, book.scans_path)

            printed_page_result = {
                "printed_page": None,
                "printed_page_text": None,
                "printed_page_kind": None,
            }
            printed_page_debug_payload: dict[str, Any] = {}
            if printed_page_detect:
                side = infer_scan_side(scan_relpath)
                raw_result, raw_debug = detect_printed_page(
                    PageRecord(
                        book_id=book.book_id,
                        page_num=page_num,
                        scan_relpath=scan_relpath,
                        ocr_engine="tesseract+pytesseract",
                        config={},
                        words=words,
                        lines=lines,
                    ),
                    page_width=page_width,
                    page_height=page_height,
                    top_band_frac=printed_page_top_band_frac,
                    min_conf=printed_page_min_conf,
                    roman_min_len=printed_page_roman_min_len,
                    roman_max_value=printed_page_roman_max,
                    side=side,
                    debug=printed_page_debug,
                )
                printed_page_result, printed_page_mode = apply_printed_page_mode(
                    raw_result,
                    "arabic" if printed_page_mode == "arabic" else "auto",
                    arabic_switch_min=printed_page_arabic_switch_min,
                )
                printed_page_debug_payload = raw_debug

            page_record = PageRecord(
                book_id=book.book_id,
                page_num=page_num,
                scan_relpath=scan_relpath,
                ocr_engine="tesseract+pytesseract",
                config={
                    "config_hash": config_hash,
                    "line_y_tolerance_px": y_tolerance_px,
                    "language": language,
                    "psm": psm,
                    "printed_page_detect": printed_page_detect,
                    "printed_page_top_band_frac": printed_page_top_band_frac,
                    "printed_page_min_conf": printed_page_min_conf,
                    "printed_page_roman_max": printed_page_roman_max,
                    "printed_page_roman_min_len": printed_page_roman_min_len,
                    "printed_page_arabic_switch_min": printed_page_arabic_switch_min,
                    "printed_page_debug": printed_page_debug,
                },
                words=words,
                lines=lines,
                printed_page=printed_page_result.get("printed_page"),
                printed_page_text=printed_page_result.get("printed_page_text"),
                printed_page_kind=printed_page_result.get("printed_page_kind"),
            )
            page_records.append(to_jsonable(page_record))

            page_dir = run_root / f"book_{book.book_id}" / f"page_{page_num:04d}"
            page_text_payload: dict[str, Any] = {
                "book_id": book.book_id,
                "page_num": page_num,
                "scan_relpath": page_record.scan_relpath,
                "text": page_text,
                "words": to_jsonable(words),
                "lines": to_jsonable(lines),
                "printed_page": page_record.printed_page,
                "printed_page_text": page_record.printed_page_text,
                "printed_page_kind": page_record.printed_page_kind,
                "config_hash": config_hash,
            }
            if printed_page_debug and printed_page_detect:
                page_text_payload["printed_page_debug"] = {
                    "mode": printed_page_mode,
                    "result": printed_page_result,
                    "candidates": printed_page_debug_payload,
                }

            write_json_file(
                page_dir / "page_text.json",
                page_text_payload,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
                run_root=run_root,
            )
            overlay = _draw_overlay(image, words, lines)
            _save_overlay_image(
                page_dir / "page_overlay.png",
                overlay,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
                run_root=run_root,
            )

    corpus_file = out_root / "books" / book.book_id / "pages.jsonl"
    corpus_overwrite: OverwriteMode = "always" if args.overwrite == "always" else "never"
    write_jsonl_records(
        corpus_file,
        page_records,
        dry_run=args.dry_run,
        overwrite=corpus_overwrite,
        run_root=None,
    )
    print(
        f"OCR completed for {len(page_records)} pages. "
        f"Corpus: {corpus_file} | Run artifacts: {run_root / f'book_{book.book_id}'}"
    )
    return 0
