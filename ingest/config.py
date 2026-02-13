from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .types import BookConfig


DEFAULT_PIPELINE_CONFIG: dict[str, Any] = {
    "ocr": {
        "line_y_tolerance_px": 14,
        "language": "eng",
        "psm": 6,
    },
    "highlights": {
        "hsv_low": [15, 20, 80],
        "hsv_high": [95, 255, 255],
        "min_area": 120,
        "kernel_size": 5,
    },
    "spans": {
        "k_before": 2,
        "k_after": 2,
    },
    "emit_obsidian": {
        "sidecar_json_default": True,
    },
}


class ConfigError(RuntimeError):
    pass


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"Expected mapping in YAML file: {path}")
    return loaded


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _resolve_path(config_path: Path, value: str | None) -> Path | None:
    if value is None or value == "":
        return None
    resolved = Path(value)
    if resolved.is_absolute():
        return resolved
    return (config_path.parent / resolved).resolve()


def load_pipeline_config(pipeline_path: Path | None) -> dict[str, Any]:
    if pipeline_path is None:
        return dict(DEFAULT_PIPELINE_CONFIG)
    loaded = _load_yaml(pipeline_path)
    return _deep_merge(DEFAULT_PIPELINE_CONFIG, loaded)


def load_book_config(book_path: Path, pipeline_path: Path | None) -> tuple[BookConfig, dict[str, Any], str]:
    pipeline_config = load_pipeline_config(pipeline_path)
    raw = _load_yaml(book_path)

    book_id = str(raw.get("book_id", "")).strip()
    if not book_id:
        raise ConfigError(f"book_id is required in {book_path}")

    scans_path = _resolve_path(book_path, raw.get("scans_path"))
    if scans_path is None:
        raise ConfigError(f"scans_path is required in {book_path}")

    vault_out_path = _resolve_path(book_path, raw.get("vault_out_path"))
    tags = raw.get("tags") or []
    if not isinstance(tags, list):
        raise ConfigError(f"tags must be a list in {book_path}")
    tag_values = [str(tag) for tag in tags]

    known_keys = {
        "book_id",
        "title",
        "creator",
        "year",
        "format",
        "scans_path",
        "vault_out_path",
        "publisher_studio",
        "note_type",
        "note_status",
        "note_version",
        "YAML_schema_version",
        "register",
        "tags",
    }
    metadata = {k: v for k, v in raw.items() if k not in known_keys}

    book = BookConfig(
        book_id=book_id,
        scans_path=scans_path,
        vault_out_path=vault_out_path,
        title=str(raw.get("title", "")),
        creator=str(raw.get("creator", "")),
        year=str(raw.get("year", "")),
        format=str(raw.get("format", "book")),
        publisher_studio=str(raw.get("publisher_studio", "")),
        note_type=str(raw.get("note_type", "literature_review")),
        note_status=str(raw.get("note_status", "inbox")),
        note_version=str(raw.get("note_version", "v0.1.3")),
        yaml_schema_version=str(raw.get("YAML_schema_version", "v0.1.2")),
        register=str(raw.get("register", "public")),
        tags=tag_values,
        metadata=metadata,
    )

    config_material = {"book": raw, "pipeline": pipeline_config}
    packed = json.dumps(config_material, sort_keys=True, ensure_ascii=True).encode("utf-8")
    config_hash = hashlib.sha256(packed).hexdigest()[:16]

    return book, pipeline_config, config_hash
