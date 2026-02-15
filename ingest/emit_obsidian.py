from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

import yaml

from .artifacts import write_json_file, write_text_file
from .config import load_book_config
from .text_clean import clean_ocr_text
from .textmap import load_pages_jsonl
from .utils_paths import find_latest_run_id


OBSIDIAN_SCHEMA_KEYS = {
    "address",
    "aliases",
    "birthday",
    "book_read_today",
    "bridge_applicability_scope",
    "bridge_applied",
    "bridge_broken",
    "bridge_conditions",
    "bridge_isomorphism",
    "bridge_justification",
    "bridge_methods",
    "bridge_preservation",
    "bridge_required",
    "bridge_uuids",
    "canonical_name",
    "cash_out",
    "creator",
    "dislikes",
    "dream_location",
    "dream_lucidity",
    "dream_motif",
    "dream_motif_valence",
    "email",
    "entity_type",
    "first_met",
    "format",
    "from_mode",
    "from_register",
    "hypnagogic_resonance",
    "interface",
    "iso_broken",
    "iso_justification",
    "iso_structure",
    "layer",
    "likes",
    "note_status",
    "note_type",
    "note_version",
    "occupation",
    "origin",
    "phone",
    "pillar",
    "publisher_studio",
    "quarantine_reasons",
    "racing_thoughts_while_awake",
    "ran_script_when_racing",
    "ran_script_yesterday",
    "reactivity",
    "recall_ability",
    "register",
    "register_mode",
    "relationship",
    "revision_triggers",
    "rhetoric_allowed",
    "rhetorical_device",
    "root",
    "speculation_quarantine",
    "stop_rule",
    "tags",
    "temporal_pace",
    "tension_type",
    "title",
    "to_mode",
    "to_register",
    "transition_attempted",
    "unity_level",
    "uuid",
    "vector_direction",
    "YAML_schema_version",
    "year",
}

_FRONTMATTER_BLOCK_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---(?:\r?\n|$)", re.DOTALL)
_TOP_LEVEL_KEY_RE = re.compile(r"^([A-Za-z0-9_]+)\s*:")


def yaml_quote(value: str) -> str:
    """Return a YAML-safe double-quoted scalar."""
    return json.dumps("" if value is None else str(value), ensure_ascii=False)


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
        return f"  - {yaml_quote('ingest/highlight_excerpt')}"
    return "\n".join(f"  - {yaml_quote(tag)}" for tag in unique)


def _collect_quote_lines(lines: list[dict[str, Any]], line_ids: list[str]) -> list[str]:
    line_by_id = {line["line_id"]: line for line in lines}
    ordered = [line_by_id[line_id]["text"] for line_id in line_ids if line_id in line_by_id]
    return [str(text) for text in ordered if str(text).strip()]


def _render_quote_text(lines: list[str], *, clean_text: bool) -> str:
    if not lines:
        return ""
    if not clean_text:
        return "\n".join(line for line in lines if line.strip())
    return clean_ocr_text(lines)


def _source_block(
    *,
    book_id: str,
    page_num: int,
    scan_relpath: str,
    printed_page_text: str | None,
    printed_page_kind: str | None,
    span: dict[str, Any],
    run_id: str,
    config_hash: str,
) -> str:
    rows = [
        f"- book_id: {book_id}",
        f"- page_num: {page_num}",
        f"- scan_relpath: {scan_relpath}",
        f"- span_id: {span['span_id']}",
        f"- line_ids: {', '.join(span.get('line_ids', []))}",
        f"- run_id: {run_id}",
        f"- config_hash: {config_hash}",
    ]
    if printed_page_text:
        if printed_page_kind:
            rows.append(f"- printed_page: {printed_page_text} ({printed_page_kind})")
        else:
            rows.append(f"- printed_page: {printed_page_text}")
    return "\n".join(rows)


def _extract_frontmatter_block(note_content: str) -> str:
    match = _FRONTMATTER_BLOCK_RE.search(note_content)
    if not match:
        raise RuntimeError("Rendered note is missing a valid frontmatter block delimited by ---.")
    return match.group(1)


def _extract_top_level_frontmatter_keys(frontmatter_block: str) -> set[str]:
    keys: set[str] = set()
    for line in frontmatter_block.splitlines():
        if not line:
            continue
        if line.startswith((" ", "\t", "-")):
            continue
        match = _TOP_LEVEL_KEY_RE.match(line)
        if match:
            keys.add(match.group(1))
    return keys


def _validate_frontmatter_schema_keys(note_content: str) -> None:
    frontmatter = _extract_frontmatter_block(note_content)
    keys = _extract_top_level_frontmatter_keys(frontmatter)
    invalid = sorted(key for key in keys if key not in OBSIDIAN_SCHEMA_KEYS)
    if invalid:
        raise RuntimeError(
            "Rendered note contains frontmatter keys outside OBSIDIAN_SCHEMA_KEYS: "
            + ", ".join(invalid)
        )


def _assert_frontmatter_yaml(note_content: str) -> None:
    frontmatter = _extract_frontmatter_block(note_content)
    try:
        parsed = yaml.safe_load(frontmatter)
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Rendered note frontmatter is not valid YAML: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("Rendered note frontmatter did not parse into a mapping.")


def smoke_check_note_render(template_path: Path = Path("templates/obsidian_note.md")) -> None:
    template = template_path.read_text(encoding="utf-8")
    replacements = {
        "uuid": yaml_quote("00000000-0000-0000-0000-000000000000"),
        "note_version": yaml_quote("v0.1.3"),
        "YAML_schema_version": yaml_quote("v0.1.2"),
        "note_type": yaml_quote("literature_review"),
        "note_status": yaml_quote("inbox"),
        "tags_block": _build_tags_block(['book/sample_book', 'quote "escaped" check']),
        "format": yaml_quote("book"),
        "title": yaml_quote('Example "Title"'),
        "creator": yaml_quote('Creator "Name"'),
        "year": yaml_quote("1900"),
        "publisher_studio": yaml_quote('Publisher "Studio"'),
        "register": yaml_quote("public"),
        "quote_text": "Example quote line.",
        "source_block": "- page_num: 1\n- span_id: p1_s1",
    }
    rendered = _render_template(template=template, replacements=replacements)
    _validate_frontmatter_schema_keys(rendered)
    _assert_frontmatter_yaml(rendered)


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
            quote_lines = _collect_quote_lines(lines=lines, line_ids=span.get("line_ids", []))
            quote_text = _render_quote_text(quote_lines, clean_text=bool(getattr(args, "clean_text", True)))
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
                printed_page_text=(
                    None if page.get("printed_page_text") in (None, "") else str(page.get("printed_page_text"))
                ),
                printed_page_kind=None if page.get("printed_page_kind") in (None, "") else str(page.get("printed_page_kind")),
                span=span,
                run_id=run_id,
                config_hash=config_hash,
            )
            replacements = {
                "uuid": yaml_quote(note_uuid),
                "note_version": yaml_quote(book.note_version),
                "YAML_schema_version": yaml_quote(book.yaml_schema_version),
                "note_type": yaml_quote(book.note_type),
                "note_status": yaml_quote(book.note_status),
                "tags_block": tags_block,
                "format": yaml_quote(book.format),
                "title": yaml_quote(title),
                "creator": yaml_quote(book.creator),
                "year": yaml_quote(book.year),
                "publisher_studio": yaml_quote(book.publisher_studio),
                "register": yaml_quote(book.register),
                "quote_text": quote_text,
                "source_block": source_block,
            }
            note_content = _render_template(template=template, replacements=replacements)
            _validate_frontmatter_schema_keys(note_content)
            _assert_frontmatter_yaml(note_content)

            write_text_file(
                note_path,
                note_content,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
                run_root=run_root,
            )

            if args.sidecar_json:
                sidecar_payload: dict[str, Any] = {
                    "book_id": book.book_id,
                    "page_num": page_num,
                    "span_id": span["span_id"],
                    "line_ids": span.get("line_ids", []),
                    "trigger_bboxes": span.get("trigger_bboxes", []),
                    "span_bbox": span.get("span_bbox", []),
                    "run_id": run_id,
                    "config_hash": config_hash,
                    "scan_relpath": page.get("scan_relpath", ""),
                }
                if page.get("printed_page_text") not in (None, ""):
                    sidecar_payload["printed_page"] = str(page.get("printed_page_text"))
                elif page.get("printed_page") is not None:
                    sidecar_payload["printed_page"] = str(page.get("printed_page"))

                write_json_file(
                    sidecar_path,
                    sidecar_payload,
                    dry_run=args.dry_run,
                    overwrite=args.overwrite,
                    run_root=run_root,
                )

            emitted += 1

    print(f"Emitted {emitted} notes to {vault_out}.")
    return 0
