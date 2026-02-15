from __future__ import annotations

import re


_PIPE_ONLY_RE = re.compile(r"^\|(?:\s*\|)*$")
_JUNK_ONLY_RE = re.compile(r"^[\s\|\\\"'`.:;,_\-–—()\[\]{}!]+$")
_YEAR_LINE_RE = re.compile(r"^\d{3,4}\s")
_LIST_MARKER_RE = re.compile(r"^[-*•]\s")
_MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s")
_HEADINGISH_RE = re.compile(r"^(CHAPTER|PART|APPENDIX)\b", re.IGNORECASE)
_LOWER_START_RE = re.compile(r"^[a-z]")
_UPPER_START_RE = re.compile(r"^[A-Z]")
_PUNCT_ARTIFACT_START_RE = re.compile(r"^[\(\[][\s\|\\\"'`.:;,_\-–—()\[\]{}!]*$")
_STRONG_END_RE = re.compile(r"""[.!?]["')\]]*$""")


def clean_ocr_lines(lines: list[str]) -> list[str]:
    cleaned: list[str] = []
    for raw in lines:
        line = re.sub(r"\s+", " ", str(raw)).strip()
        if not line:
            continue
        if _PIPE_ONLY_RE.fullmatch(line):
            continue

        line = line.replace(" | ", " ")
        line = re.sub(r"^\|+\s*", "", line)
        line = re.sub(r"\s*\|+$", "", line)
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        if _PIPE_ONLY_RE.fullmatch(line):
            continue
        if _JUNK_ONLY_RE.fullmatch(line):
            continue

        alnum_count = sum(1 for ch in line if ch.isalnum())
        alnum_ratio = alnum_count / len(line)
        if len(line) <= 6 and alnum_ratio < 0.2:
            continue

        cleaned.append(line)
    return cleaned


def _looks_like_block_start(line: str) -> bool:
    return bool(
        _YEAR_LINE_RE.match(line)
        or _LIST_MARKER_RE.match(line)
        or _MARKDOWN_HEADING_RE.match(line)
        or _PUNCT_ARTIFACT_START_RE.match(line)
    )


def dehyphenate_linebreaks(lines: list[str]) -> list[str]:
    merged: list[str] = []
    i = 0
    while i < len(lines):
        current = lines[i]
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            if (
                current.endswith("-")
                and _LOWER_START_RE.match(next_line) is not None
                and not _looks_like_block_start(next_line)
            ):
                merged.append(current[:-1] + next_line.lstrip())
                i += 2
                continue
        merged.append(current)
        i += 1
    return merged


def _should_break_after(current: str, next_line: str) -> bool:
    if _YEAR_LINE_RE.match(next_line):
        return True
    if _LIST_MARKER_RE.match(next_line):
        return True
    if _MARKDOWN_HEADING_RE.match(next_line):
        return True
    if _HEADINGISH_RE.match(next_line):
        return True
    if _STRONG_END_RE.search(current) and _UPPER_START_RE.match(next_line):
        return True
    return False


def reflow_paragraphs(lines: list[str]) -> str:
    if not lines:
        return ""

    blocks: list[str] = []
    current_parts: list[str] = []

    def flush_current() -> None:
        if current_parts:
            blocks.append(" ".join(current_parts).strip())
            current_parts.clear()

    for idx, raw in enumerate(lines):
        line = str(raw).strip()
        if not line:
            flush_current()
            continue

        if _YEAR_LINE_RE.match(line):
            flush_current()
            blocks.append(line)
            continue

        current_parts.append(line)
        if idx + 1 >= len(lines):
            flush_current()
            continue

        next_line = str(lines[idx + 1]).strip()
        if not next_line:
            flush_current()
            continue
        if _should_break_after(line, next_line):
            flush_current()

    flush_current()
    return "\n\n".join(block for block in blocks if block)
