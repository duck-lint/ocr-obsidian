from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_pages_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"pages.jsonl not found: {path}")
    pages: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            pages.append(json.loads(line))
    return pages


def map_lines_by_page(pages: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    mapping: dict[int, list[dict[str, Any]]] = {}
    for page in pages:
        page_num = int(page["page_num"])
        lines = page.get("lines") or []
        mapping[page_num] = list(lines)
    return mapping
