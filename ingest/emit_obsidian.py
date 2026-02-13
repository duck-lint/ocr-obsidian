from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

from .artifacts import write_json_file, write_text_file
from .config import load_book_config
from .textmap import load_pages_jsonl
from .utils_paths import find_latest_run_id


ALLOWED_YAML_KEYS = {
    "uuid",
    "note_version",
    "YAML_schema_version",
    "note_type",
    "note_status",
    "tags",
    "format",
    "title",
    "creator",
    "year",
    "publisher_studio",
    "register",
}


def _sanitize_filename(value: str) -> str:
    collapsed = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return collapsed or "note"


def _render_template(template: str, replacements: dict[str, str]) -> str:
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def _build_tags_block(tags: list[str]) -> str:
    unique: list[str] = []
    for tag in tags:
        if tag not in unique:
            unique.append(tag)
    if not unique:
        return "  - ingest/highlight_excerpt"
    return "\n".join(f"  - {tag}" for tag in unique)


def _collect_quote(lines: list[dict[str, Any]], line_ids: list[str]) -> str:
    line_by_id = {line["line_id"]: line for line in lines}
    ordered = [line_by_id[line_id]["text"] for line_id in line_ids if line_id in line_by_id]
    return "\n".join(text for text in ordered if str(text).strip())


def _source_block(
    *,
    book_id: str,
    page_num: int,
    scan_relpath: str,
    span: dict[str, Any],
    run_id: str,
    config_hash: str,
) -> str:
    return "\n".join(
        [
            f"- book_id: {book_id}",
            f"- page_num: {page_num}",
            f"- scan_relpath: {scan_relpath}",
            f"- span_id: {span['span_id']}",
            f"- line_ids: {', '.join(span.get('line_ids', []))}",
            f"- run_id: {run_id}",
            f"- config_hash: {config_hash}",
        ]
    )


def run_emit_obsidian(args) -> int:
    book, _pipeline_config, config_hash = load_book_config(args.book, args.pipeline)
    runs_root = Path(args.runs)
    run_id = args.run_id or find_latest_run_id(runs_root, book.book_id, required_filename="spans.json")
    if not run_id:
        raise RuntimeError("No run id provided and no recent run with span artifacts was found. Pass --run-id.")
    run_root = (runs_root / run_id).resolve()

    vault_root = Path(args.vault) if args.vault is not None else (book.vault_out_path or (run_root / "obsidian_staging"))
    vault_out = vault_root / book.book_id

    template_path = Path(getattr(args, "template", "templates/obsidian_note.md"))
    if not template_path.exists():
        raise RuntimeError(f"Template file not found: {template_path}")
    template = template_path.read_text(encoding="utf-8")

    pages = load_pages_jsonl(Path(getattr(args, "corpus", "corpus")) / "books" / book.book_id / "pages.jsonl")
    page_map = {int(page["page_num"]): page for page in pages}

    span_files = sorted(run_root.glob(f"book_{book.book_id}/page_*/spans.json"))
    if args.max_pages is not None and args.max_pages > 0:
        span_files = span_files[: args.max_pages]
    if not span_files:
        raise RuntimeError(f"No spans.json files found for run {run_id}.")

    emitted = 0
    for span_file in span_files:
        payload = json.loads(span_file.read_text(encoding="utf-8-sig"))
        page_num = int(payload["page_num"])
        page = page_map.get(page_num)
        if page is None:
            continue
        lines = page.get("lines") or []

        for span in payload.get("spans", []):
            quote_text = _collect_quote(lines=lines, line_ids=span.get("line_ids", []))
            if not quote_text.strip():
                continue

            note_uuid = str(uuid.uuid4())
            title = f"{book.title or book.book_id} p{page_num} {span['span_id']}"
            note_name = _sanitize_filename(f"{book.book_id}_{span['span_id']}")
            note_path = vault_out / f"{note_name}.md"
            sidecar_path = vault_out / f"{note_name}.span.json"

            tags = ["book/" + book.book_id, "ingest/highlight_excerpt"] + list(book.tags)
            tags_block = _build_tags_block(tags)
            source_block = _source_block(
                book_id=book.book_id,
                page_num=page_num,
                scan_relpath=str(page.get("scan_relpath", "")),
                span=span,
                run_id=run_id,
                config_hash=config_hash,
            )
            replacements = {
                "uuid": note_uuid,
                "note_version": book.note_version,
                "YAML_schema_version": book.yaml_schema_version,
                "note_type": book.note_type,
                "note_status": book.note_status,
                "tags_block": tags_block,
                "format": book.format,
                "title": title,
                "creator": book.creator,
                "year": book.year,
                "publisher_studio": book.publisher_studio,
                "register": book.register,
                "quote_text": quote_text,
                "source_block": source_block,
            }
            unexpected_keys = set(replacements).difference(
                {
                    "uuid",
                    "note_version",
                    "YAML_schema_version",
                    "note_type",
                    "note_status",
                    "tags_block",
                    "format",
                    "title",
                    "creator",
                    "year",
                    "publisher_studio",
                    "register",
                    "quote_text",
                    "source_block",
                }
            )
            if unexpected_keys:
                raise RuntimeError(f"Internal template key mismatch: {unexpected_keys}")

            # Enforce schema safety by using a fixed replacement set tied to known YAML keys.
            if not ALLOWED_YAML_KEYS.issuperset(
                {
                    "uuid",
                    "note_version",
                    "YAML_schema_version",
                    "note_type",
                    "note_status",
                    "tags",
                    "format",
                    "title",
                    "creator",
                    "year",
                    "publisher_studio",
                    "register",
                }
            ):
                raise RuntimeError("Allowed YAML key set is misconfigured.")

            note_content = _render_template(template=template, replacements=replacements)
            write_text_file(
                note_path,
                note_content,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
                run_root=run_root,
            )

            if args.sidecar_json:
                write_json_file(
                    sidecar_path,
                    {
                        "book_id": book.book_id,
                        "page_num": page_num,
                        "span_id": span["span_id"],
                        "line_ids": span.get("line_ids", []),
                        "trigger_bboxes": span.get("trigger_bboxes", []),
                        "span_bbox": span.get("span_bbox", []),
                        "run_id": run_id,
                        "config_hash": config_hash,
                        "scan_relpath": page.get("scan_relpath", ""),
                    },
                    dry_run=args.dry_run,
                    overwrite=args.overwrite,
                    run_root=run_root,
                )

            emitted += 1

    print(f"Emitted {emitted} notes to {vault_out}.")
    return 0
