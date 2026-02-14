from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from .artifacts import write_text_file
from .config import ConfigError
from .emit_obsidian import run_emit_obsidian
from .highlights import HighlightDependencyError, run_detect_highlights
from .ocr import OcrDependencyError, run_ocr
from .spans import run_make_spans
from .sweep import run_sweep
from .textmap import load_pages_jsonl
from .utils_paths import MissingPathError, OverwriteError


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
    spans_parser.add_argument("--corpus", type=Path, default=Path("corpus"), help="Corpus root.")
    spans_parser.add_argument("--k-before", type=int, default=2, help="Context lines to include before triggers.")
    spans_parser.add_argument("--k-after", type=int, default=2, help="Context lines to include after triggers.")
    spans_parser.set_defaults(handler=run_make_spans)

    emit_parser = subparsers.add_parser("emit-obsidian", help="Emit Obsidian markdown notes from spans.")
    _add_common_flags(emit_parser)
    emit_parser.add_argument("--runs", type=Path, default=Path("runs"), help="Run artifacts root.")
    emit_parser.add_argument("--corpus", type=Path, default=Path("corpus"), help="Corpus root.")
    emit_parser.add_argument("--vault", type=Path, default=None, help="Override vault output path.")
    emit_parser.add_argument(
        "--template",
        type=Path,
        default=Path("templates/obsidian_note.md"),
        help="Markdown note template path.",
    )
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
    export_parser.add_argument(
        "--format",
        choices=["txt", "md"],
        default="txt",
        help="Output format for exported book text.",
    )
    export_parser.set_defaults(handler=run_export_book_text)

    sweep_parser = subparsers.add_parser("sweep", help="Run deterministic QA sweep over emitted span sidecars.")
    sweep_parser.add_argument("--corpus_dir", required=True, type=Path, help="Canonical corpus root.")
    sweep_parser.add_argument("--sidecars_dir", required=True, type=Path, help="Directory containing *.span.json.")
    sweep_parser.add_argument("--notes_dir", type=Path, default=None, help="Optional directory containing notes.")
    sweep_parser.add_argument("--glob", default="*.span.json", help="Glob pattern under --sidecars_dir.")
    sweep_parser.add_argument("--out_dir", required=True, type=Path, help="Directory to write QA reports.")
    sweep_parser.add_argument("--max_items", type=int, default=None, help="Optional max number of sidecars.")
    sweep_parser.add_argument("--dry_run", action="store_true", help="Print actions without writing files.")
    sweep_parser.add_argument(
        "--overwrite",
        choices=["never", "if_same_run", "always"],
        default="never",
        help="Overwrite policy for report files. Default is fail-closed.",
    )
    sweep_parser.add_argument(
        "--thresholds",
        type=Path,
        default=None,
        help="Optional YAML/JSON threshold overrides.",
    )
    sweep_parser.add_argument("--fail_alpha_min", type=float, default=None)
    sweep_parser.add_argument("--fail_conf_min", type=float, default=None)
    sweep_parser.add_argument("--fail_garbage_max", type=float, default=None)
    sweep_parser.add_argument("--warn_line_max", type=int, default=None)
    sweep_parser.add_argument("--warn_char_max", type=int, default=None)
    sweep_parser.add_argument("--warn_pipe_max", type=float, default=None)
    sweep_parser.set_defaults(handler=run_sweep)

    return parser


def run_export_book_text(args) -> int:
    from .config import load_book_config

    book, _pipeline, _config_hash = load_book_config(args.book, args.pipeline)
    corpus_root = Path(args.out)
    pages_path = corpus_root / "books" / book.book_id / "pages.jsonl"
    pages = load_pages_jsonl(pages_path)
    if not pages:
        raise RuntimeError(f"No pages found in canonical corpus: {pages_path}")

    if args.format == "md":
        text_parts: list[str] = [f"# {book.title or book.book_id}".strip(), ""]
    else:
        text_parts = []

    for page in sorted(pages, key=lambda p: int(p["page_num"])):
        page_num = int(page["page_num"])
        printed_page = page.get("printed_page")
        display_page = page_num if printed_page in (None, "") else printed_page
        scan_relpath = str(page.get("scan_relpath", ""))
        lines = page.get("lines") or []
        page_text = "\n".join(str(line.get("text", "")) for line in lines if str(line.get("text", "")).strip())
        if args.format == "md":
            text_parts.append(f"## Page {display_page} (scan: {scan_relpath})".rstrip())
            text_parts.append(page_text.strip())
            text_parts.append("---")
            text_parts.append("")
        else:
            text_parts.append(f"# Page {page_num}\n{page_text}".strip())

    output_name = "book.md" if args.format == "md" else "book.txt"
    output_path = corpus_root / "books" / book.book_id / output_name
    corpus_overwrite = "always" if args.overwrite == "always" else "never"
    write_text_file(
        output_path,
        "\n\n".join(text_parts).strip() + "\n",
        dry_run=args.dry_run,
        overwrite=corpus_overwrite,
        run_root=None,
    )
    print(f"Exported book text to {output_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return int(args.handler(args))
    except (ConfigError, MissingPathError) as exc:
        print(f"ERROR: {exc}")
        return 3
    except OverwriteError as exc:
        print(f"ERROR: {exc}")
        return 4
    except OcrDependencyError as exc:
        print(f"ERROR: {exc}")
        return 5
    except HighlightDependencyError as exc:
        print(f"ERROR: {exc}")
        return 6
    except NotImplementedError as exc:
        print(f"ERROR: {exc}")
        return 2
    except Exception as exc:  # pragma: no cover - defensive CLI path
        print(f"ERROR: {exc}")
        traceback.print_exc()
        return 1
