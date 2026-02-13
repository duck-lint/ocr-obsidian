from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from .emit_obsidian import run_emit_obsidian
from .highlights import run_detect_highlights
from .ocr import run_ocr
from .spans import run_make_spans


def _add_common_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--book", required=True, type=Path, help="Path to the book config YAML.")
    parser.add_argument(
        "--pipeline",
        type=Path,
        default=Path("configs/pipeline.yaml"),
        help="Path to pipeline config YAML.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without writing files.")
    parser.add_argument(
        "--overwrite",
        choices=["never", "if_same_run", "always"],
        default="never",
        help="Overwrite policy. Default is fail-closed.",
    )
    parser.add_argument("--run-id", default=None, help="Optional run id. Defaults to UTC timestamp.")
    parser.add_argument("--max-pages", type=int, default=None, help="Optional max number of pages to process.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ingest", description="Single-pass OCR ingestion toolchain.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ocr_parser = subparsers.add_parser("ocr", help="Run OCR spine and build canonical pages JSONL.")
    _add_common_flags(ocr_parser)
    ocr_parser.add_argument("--out", type=Path, default=Path("corpus"), help="Corpus output root.")
    ocr_parser.add_argument("--runs", type=Path, default=Path("runs"), help="Run artifacts root.")
    ocr_parser.set_defaults(handler=run_ocr)

    highlights_parser = subparsers.add_parser("detect-highlights", help="Detect highlight candidates.")
    _add_common_flags(highlights_parser)
    highlights_parser.add_argument("--runs", type=Path, default=Path("runs"), help="Run artifacts root.")
    highlights_parser.set_defaults(handler=run_detect_highlights)

    spans_parser = subparsers.add_parser("make-spans", help="Create text spans from highlights + OCR lines.")
    _add_common_flags(spans_parser)
    spans_parser.add_argument("--runs", type=Path, default=Path("runs"), help="Run artifacts root.")
    spans_parser.add_argument("--k-before", type=int, default=2, help="Context lines to include before triggers.")
    spans_parser.add_argument("--k-after", type=int, default=2, help="Context lines to include after triggers.")
    spans_parser.set_defaults(handler=run_make_spans)

    emit_parser = subparsers.add_parser("emit-obsidian", help="Emit Obsidian markdown notes from spans.")
    _add_common_flags(emit_parser)
    emit_parser.add_argument("--runs", type=Path, default=Path("runs"), help="Run artifacts root.")
    emit_parser.add_argument("--vault", type=Path, default=None, help="Override vault output path.")
    emit_parser.add_argument(
        "--sidecar-json",
        dest="sidecar_json",
        action="store_true",
        default=True,
        help="Write same-basename sidecar JSON provenance files (default).",
    )
    emit_parser.add_argument(
        "--no-sidecar-json",
        dest="sidecar_json",
        action="store_false",
        help="Disable sidecar JSON provenance files.",
    )
    emit_parser.set_defaults(handler=run_emit_obsidian)

    export_parser = subparsers.add_parser("export-book-text", help="Export quick concatenated book text.")
    _add_common_flags(export_parser)
    export_parser.add_argument("--out", type=Path, default=Path("corpus"), help="Corpus root.")
    export_parser.set_defaults(handler=run_export_book_text)

    return parser


def run_export_book_text(_args) -> int:
    raise NotImplementedError("Book text export is not implemented yet.")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return int(args.handler(args))
    except NotImplementedError as exc:
        print(f"ERROR: {exc}")
        return 2
    except Exception as exc:  # pragma: no cover - defensive CLI path
        print(f"ERROR: {exc}")
        traceback.print_exc()
        return 1
