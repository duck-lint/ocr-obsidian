from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .utils_paths import OverwriteMode, check_write_allowed, ensure_parent


def write_text_file(
    path: Path,
    content: str,
    *,
    dry_run: bool,
    overwrite: OverwriteMode,
    run_root: Path | None = None,
) -> None:
    check_write_allowed(path, overwrite=overwrite, dry_run=dry_run, run_root=run_root)
    ensure_parent(path, dry_run=dry_run)
    if dry_run:
        print(f"[dry-run] write text: {path}")
        return
    path.write_text(content, encoding="utf-8")


def write_json_file(
    path: Path,
    payload: Mapping | Sequence | list | dict,
    *,
    dry_run: bool,
    overwrite: OverwriteMode,
    run_root: Path | None = None,
    indent: int = 2,
) -> None:
    check_write_allowed(path, overwrite=overwrite, dry_run=dry_run, run_root=run_root)
    ensure_parent(path, dry_run=dry_run)
    if dry_run:
        print(f"[dry-run] write json: {path}")
        return
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=indent), encoding="utf-8")


def write_jsonl_records(
    path: Path,
    records: Iterable[dict],
    *,
    dry_run: bool,
    overwrite: OverwriteMode,
    run_root: Path | None = None,
) -> None:
    check_write_allowed(path, overwrite=overwrite, dry_run=dry_run, run_root=run_root)
    ensure_parent(path, dry_run=dry_run)
    if dry_run:
        print(f"[dry-run] write jsonl: {path}")
        return
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
