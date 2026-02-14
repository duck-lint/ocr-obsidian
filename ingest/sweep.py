from __future__ import annotations

import glob as globlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import write_json_file, write_text_file


GARBAGE_CHARS = set("|{}[]<>_~^")


@dataclass(frozen=True, slots=True)
class SweepThresholds:
    fail_alpha_min: float = 0.65
    fail_conf_min: float = 65.0
    fail_garbage_max: float = 0.08
    warn_line_max: int = 25
    warn_char_max: int = 2000
    warn_pipe_max: float = 0.02


@dataclass(frozen=True, slots=True)
class SpanSidecar:
    path: Path
    book_id: str
    page_num: int
    span_id: str
    line_ids: list[str]
    scan_relpath: str
    printed_page: str | None = None


def load_pages_index(pages_jsonl_path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    if not pages_jsonl_path.exists():
        raise FileNotFoundError(f"pages.jsonl not found: {pages_jsonl_path}")
    mapping: dict[tuple[str, int], dict[str, Any]] = {}
    with pages_jsonl_path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            page = json.loads(raw)
            book_id = str(page.get("book_id", "")).strip()
            page_num = int(page["page_num"])
            mapping[(book_id, page_num)] = page
    return mapping


def load_sidecars(glob_pattern: str) -> list[SpanSidecar]:
    matches = [Path(p) for p in globlib.glob(glob_pattern, recursive=True)]
    sidecar_paths = [p for p in matches if p.is_file() and p.name.lower().endswith(".span.json")]
    sidecar_paths.sort(key=lambda p: p.as_posix().lower())

    sidecars: list[SpanSidecar] = []
    for sidecar_path in sidecar_paths:
        try:
            payload = json.loads(sidecar_path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON in sidecar {sidecar_path}: {exc}") from exc

        line_ids_raw = payload.get("line_ids") or []
        if not isinstance(line_ids_raw, list):
            raise RuntimeError(f"Expected 'line_ids' to be a list in {sidecar_path}")
        line_ids = [str(line_id) for line_id in line_ids_raw]

        sidecars.append(
            SpanSidecar(
                path=sidecar_path,
                book_id=str(payload.get("book_id", "")).strip(),
                page_num=int(payload["page_num"]),
                span_id=str(payload["span_id"]),
                line_ids=line_ids,
                scan_relpath=str(payload.get("scan_relpath", "")),
                printed_page=(None if payload.get("printed_page") is None else str(payload.get("printed_page"))),
            )
        )
    return sidecars


def extract_span_text(page_record: dict[str, Any], line_ids: list[str]) -> tuple[str, int]:
    lines = page_record.get("lines") or []
    by_id = {str(line.get("line_id")): line for line in lines}
    selected_text: list[str] = []
    confidences_used_count = 0

    for line_id in line_ids:
        line = by_id.get(line_id)
        if line is None:
            continue
        text = str(line.get("text", ""))
        if text:
            selected_text.append(text)
        for word in line.get("words") or []:
            confidence = word.get("confidence")
            if isinstance(confidence, (int, float)):
                confidences_used_count += 1

    return "\n".join(selected_text), confidences_used_count


def compute_metrics(text: str, avg_word_conf: float | None, line_count: int) -> dict[str, Any]:
    non_space_chars = [ch for ch in text if not ch.isspace()]
    non_space_count = len(non_space_chars)
    alpha_count = sum(1 for ch in non_space_chars if ch.isalpha())
    garbage_count = sum(1 for ch in non_space_chars if ch in GARBAGE_CHARS)
    pipe_count = sum(1 for ch in non_space_chars if ch == "|")

    if non_space_count == 0:
        alpha_ratio = 0.0
        garbage_ratio = 0.0
        pipe_ratio = 0.0
    else:
        alpha_ratio = alpha_count / non_space_count
        garbage_ratio = garbage_count / non_space_count
        pipe_ratio = pipe_count / non_space_count

    return {
        "char_count": len(text),
        "line_count": int(line_count),
        "avg_word_conf": (None if avg_word_conf is None else round(float(avg_word_conf), 3)),
        "alpha_ratio": round(alpha_ratio, 6),
        "garbage_ratio": round(garbage_ratio, 6),
        "pipe_ratio": round(pipe_ratio, 6),
    }


def decide_verdict(metrics: dict[str, Any], thresholds: SweepThresholds) -> tuple[str, list[str]]:
    fail_reasons: list[str] = []
    warn_reasons: list[str] = []
    avg_word_conf = metrics.get("avg_word_conf")

    if float(metrics["alpha_ratio"]) < thresholds.fail_alpha_min:
        fail_reasons.append(
            f"alpha_ratio {float(metrics['alpha_ratio']):.6f} < fail_alpha_min {thresholds.fail_alpha_min:.6f}"
        )
    if avg_word_conf is not None and float(avg_word_conf) < thresholds.fail_conf_min:
        fail_reasons.append(
            f"avg_word_conf {float(avg_word_conf):.3f} < fail_conf_min {thresholds.fail_conf_min:.3f}"
        )
    if float(metrics["garbage_ratio"]) > thresholds.fail_garbage_max:
        fail_reasons.append(
            f"garbage_ratio {float(metrics['garbage_ratio']):.6f} > fail_garbage_max {thresholds.fail_garbage_max:.6f}"
        )

    if int(metrics["line_count"]) > thresholds.warn_line_max:
        warn_reasons.append(f"line_count {int(metrics['line_count'])} > warn_line_max {thresholds.warn_line_max}")
    if int(metrics["char_count"]) > thresholds.warn_char_max:
        warn_reasons.append(f"char_count {int(metrics['char_count'])} > warn_char_max {thresholds.warn_char_max}")
    if float(metrics["pipe_ratio"]) > thresholds.warn_pipe_max:
        warn_reasons.append(
            f"pipe_ratio {float(metrics['pipe_ratio']):.6f} > warn_pipe_max {thresholds.warn_pipe_max:.6f}"
        )

    if fail_reasons:
        return "FAIL", fail_reasons + warn_reasons
    if warn_reasons:
        return "WARN", warn_reasons
    return "PASS", []


def write_reports(
    out_dir: Path,
    records: list[dict[str, Any]],
    *,
    dry_run: bool = False,
    overwrite: str = "never",
) -> tuple[Path, Path]:
    out_path = Path(out_dir)
    json_path = out_path / "qa_report.json"
    md_path = out_path / "qa_report.md"

    write_json_file(
        json_path,
        records,
        dry_run=dry_run,
        overwrite=overwrite,
        run_root=out_path,
    )

    grouped: dict[str, list[dict[str, Any]]] = {"FAIL": [], "WARN": [], "PASS": []}
    for record in records:
        grouped.setdefault(str(record.get("verdict", "PASS")), []).append(record)

    lines: list[str] = ["# QA Sweep Report", "", f"Total spans: {len(records)}", ""]
    for verdict in ("FAIL", "WARN", "PASS"):
        entries = grouped.get(verdict, [])
        lines.append(f"## {verdict} ({len(entries)})")
        lines.append("")
        for entry in entries:
            preview = str(entry.get("preview", "")).replace("```", "'''")
            reasons = entry.get("reasons") or []
            reason_text = "; ".join(str(reason) for reason in reasons) if reasons else "(none)"
            printed_page = entry.get("printed_page")
            printed_part = f", printed_page={printed_page}" if printed_page is not None else ""
            metrics = entry.get("metrics") or {}
            metric_summary = (
                f"char_count={metrics.get('char_count')}, "
                f"line_count={metrics.get('line_count')}, "
                f"avg_word_conf={metrics.get('avg_word_conf')}, "
                f"alpha_ratio={metrics.get('alpha_ratio')}, "
                f"garbage_ratio={metrics.get('garbage_ratio')}, "
                f"pipe_ratio={metrics.get('pipe_ratio')}"
            )

            lines.append(f"### {entry.get('book_id')}:{entry.get('span_id')}")
            lines.append(f"- sidecar_path: {entry.get('sidecar_path')}")
            lines.append(f"- note_path: {entry.get('note_path')}")
            lines.append(f"- scan_relpath: {entry.get('scan_relpath')}")
            lines.append(f"- page_num={entry.get('page_num')}{printed_part}")
            lines.append(f"- line_ids: {', '.join(entry.get('line_ids') or [])}")
            lines.append(f"- verdict: {entry.get('verdict')}")
            lines.append(f"- reasons: {reason_text}")
            lines.append(f"- metrics: {metric_summary}")
            lines.append("- preview:")
            lines.append("```text")
            lines.append(preview)
            lines.append("```")
            lines.append("")

    write_text_file(
        md_path,
        "\n".join(lines).rstrip() + "\n",
        dry_run=dry_run,
        overwrite=overwrite,
        run_root=out_path,
    )

    return json_path, md_path


def _sidecar_note_basename(sidecar_path: Path) -> str:
    name = sidecar_path.name
    if name.lower().endswith(".span.json"):
        return name[: -len(".span.json")]
    return sidecar_path.stem


def _truncate_preview(text: str, max_chars: int = 300) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _average_word_conf(page_record: dict[str, Any], line_ids: list[str]) -> tuple[float | None, int]:
    lines = page_record.get("lines") or []
    by_id = {str(line.get("line_id")): line for line in lines}
    confidences: list[float] = []
    for line_id in line_ids:
        line = by_id.get(line_id)
        if line is None:
            continue
        for word in line.get("words") or []:
            confidence = word.get("confidence")
            if isinstance(confidence, (int, float)):
                confidences.append(float(confidence))
    if not confidences:
        return None, 0
    return (sum(confidences) / len(confidences), len(confidences))


def _line_count(page_record: dict[str, Any], line_ids: list[str]) -> int:
    lines = page_record.get("lines") or []
    available = {str(line.get("line_id")) for line in lines}
    return sum(1 for line_id in line_ids if line_id in available)


def _load_threshold_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Threshold config must be a mapping: {path}")
        return payload

    parsed: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        if not key:
            continue
        if value.lower() in {"null", "none", "~"}:
            parsed[key] = None
            continue
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            parsed[key] = value[1:-1]
            continue
        try:
            if "." in value or "e" in value.lower():
                parsed[key] = float(value)
            else:
                parsed[key] = int(value)
        except ValueError:
            parsed[key] = value
    return parsed


def _resolve_thresholds(args) -> SweepThresholds:
    default_thresholds = SweepThresholds()
    values: dict[str, Any] = {
        "fail_alpha_min": default_thresholds.fail_alpha_min,
        "fail_conf_min": default_thresholds.fail_conf_min,
        "fail_garbage_max": default_thresholds.fail_garbage_max,
        "warn_line_max": default_thresholds.warn_line_max,
        "warn_char_max": default_thresholds.warn_char_max,
        "warn_pipe_max": default_thresholds.warn_pipe_max,
    }

    thresholds_path = getattr(args, "thresholds", None)
    if thresholds_path is not None:
        file_values = _load_threshold_mapping(Path(thresholds_path))
        for key in values:
            if key in file_values and file_values[key] is not None:
                values[key] = file_values[key]

    for key in values:
        cli_value = getattr(args, key, None)
        if cli_value is not None:
            values[key] = cli_value

    return SweepThresholds(
        fail_alpha_min=float(values["fail_alpha_min"]),
        fail_conf_min=float(values["fail_conf_min"]),
        fail_garbage_max=float(values["fail_garbage_max"]),
        warn_line_max=int(values["warn_line_max"]),
        warn_char_max=int(values["warn_char_max"]),
        warn_pipe_max=float(values["warn_pipe_max"]),
    )


def run_sweep(args) -> int:
    corpus_dir = Path(args.corpus_dir)
    sidecars_dir = Path(args.sidecars_dir)
    notes_dir = None if getattr(args, "notes_dir", None) is None else Path(args.notes_dir)
    out_dir = Path(args.out_dir)

    glob_pattern = str(sidecars_dir / getattr(args, "glob", "*.span.json"))
    sidecars = load_sidecars(glob_pattern)
    if not sidecars:
        raise RuntimeError(f"No sidecars found for pattern: {glob_pattern}")
    max_items = getattr(args, "max_items", None)
    if max_items is not None and int(max_items) > 0:
        sidecars = sidecars[: int(max_items)]

    thresholds = _resolve_thresholds(args)

    notes_index: dict[str, list[Path]] = {}
    if notes_dir is not None:
        note_paths = sorted(notes_dir.rglob("*.md"), key=lambda p: p.as_posix().lower())
        for note_path in note_paths:
            notes_index.setdefault(note_path.stem.lower(), []).append(note_path)

    pages_cache: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    records: list[dict[str, Any]] = []

    for sidecar in sidecars:
        if not sidecar.book_id:
            raise RuntimeError(f"Missing book_id in sidecar: {sidecar.path}")

        if sidecar.book_id not in pages_cache:
            pages_path = corpus_dir / "books" / sidecar.book_id / "pages.jsonl"
            pages_cache[sidecar.book_id] = load_pages_index(pages_path)
        pages_index = pages_cache[sidecar.book_id]
        page_record = pages_index.get((sidecar.book_id, sidecar.page_num))

        note_path: str | None = None
        note_basename = _sidecar_note_basename(sidecar.path)
        sibling_note = sidecar.path.parent / f"{note_basename}.md"
        if sibling_note.exists():
            note_path = str(sibling_note)
        elif notes_dir is not None:
            candidates = notes_index.get(note_basename.lower()) or []
            if candidates:
                note_path = str(candidates[0])

        extra_reasons: list[str] = []
        if page_record is None:
            text = ""
            avg_word_conf = None
            line_count = 0
            extra_reasons.append(
                f"canonical page missing for {sidecar.book_id} page_num={sidecar.page_num}"
            )
            confidences_used_count = 0
        else:
            text, confidences_used_count = extract_span_text(page_record, sidecar.line_ids)
            avg_word_conf, _conf_count = _average_word_conf(page_record, sidecar.line_ids)
            line_count = _line_count(page_record, sidecar.line_ids)

        metrics = compute_metrics(text=text, avg_word_conf=avg_word_conf, line_count=line_count)
        verdict, reasons = decide_verdict(metrics=metrics, thresholds=thresholds)
        if extra_reasons:
            reasons = extra_reasons + reasons
            verdict = "FAIL"

        scan_relpath = sidecar.scan_relpath
        if (not scan_relpath) and page_record is not None:
            scan_relpath = str(page_record.get("scan_relpath", ""))

        record: dict[str, Any] = {
            "book_id": sidecar.book_id,
            "page_num": sidecar.page_num,
            "scan_relpath": scan_relpath,
            "span_id": sidecar.span_id,
            "line_ids": sidecar.line_ids,
            "note_path": note_path,
            "sidecar_path": str(sidecar.path),
            "metrics": metrics,
            "verdict": verdict,
            "reasons": reasons,
            "preview": _truncate_preview(text, max_chars=300),
            "confidences_used_count": confidences_used_count,
        }
        if sidecar.printed_page is not None:
            record["printed_page"] = sidecar.printed_page
        records.append(record)

    json_path, md_path = write_reports(
        out_dir=out_dir,
        records=records,
        dry_run=bool(getattr(args, "dry_run", False)),
        overwrite=str(getattr(args, "overwrite", "never")),
    )
    print(f"Sweep wrote reports: {json_path} and {md_path}")
    return 0
