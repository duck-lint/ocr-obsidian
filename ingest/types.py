from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


BBox = list[int]


@dataclass(slots=True)
class BookConfig:
    book_id: str
    scans_path: Path
    vault_out_path: Path | None = None
    title: str = ""
    creator: str = ""
    year: str = ""
    format: str = "book"
    publisher_studio: str = ""
    note_type: str = "literature_review"
    note_status: str = "inbox"
    note_version: str = "v0.1.3"
    yaml_schema_version: str = "v0.1.2"
    register: str = "public"
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OcrWord:
    text: str
    bbox: BBox
    confidence: float


@dataclass(slots=True)
class OcrLine:
    line_id: str
    bbox: BBox
    words: list[OcrWord]
    text: str


@dataclass(slots=True)
class PageRecord:
    book_id: str
    page_num: int
    scan_relpath: str
    ocr_engine: str
    config: dict[str, Any]
    words: list[OcrWord]
    lines: list[OcrLine]
    printed_page: int | None = None
    printed_page_text: str | None = None
    printed_page_kind: str | None = None


@dataclass(slots=True)
class HighlightCandidate:
    bbox: BBox
    area: int
    color_stats: dict[str, float]


@dataclass(slots=True)
class Span:
    span_id: str
    page_num: int
    line_ids: list[str]
    trigger_bboxes: list[BBox]
    span_bbox: BBox


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    return value
