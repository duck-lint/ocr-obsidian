from __future__ import annotations

import re


_PUNCT_ONLY_RE = re.compile(r"^[\|\\\"'`.:;,_\-–—()\[\]{}<>!]+$")
_YEAR_LINE_RE = re.compile(r"^\d{3,4}\s")
_LIST_MARKER_RE = re.compile("^(\\-|\\*|\u2022)\\s")
_HEADER_BLEED_RE = re.compile(r"^Chronology of .* [IVXLCDM]+$", re.IGNORECASE)
_LOWER_START_RE = re.compile(r"^[a-z]")
_UPPER_START_RE = re.compile(r"^[A-Z]")
_SENTENCE_END_RE = re.compile(r"[.?!]$")
_HYPHEN_TAIL_RE = re.compile(r"-\s*[)\]}\'\"`.:;,_!]*$")
_LEADING_PUNCT_RE = re.compile(r"^[\s\|\\\"'`.:;,_\-–—()\[\]{}<>!]+")

_SHORT_JUNK_CHARS = {"|", "\\", ")", "(", "{", "}", "[", "]", "!", ":", ";", "\"", "'"}
_KNOWN_JUNK_EXACT = {"fi", "fl", "Hl", "Hh", "tH", "tH!", "i|", "|i", "Il", "l|"}
_KNOWN_JUNK_LOWER = {token.lower() for token in _KNOWN_JUNK_EXACT}


def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def _alpha_ratio(tok: str) -> float:
    if not tok:
        return 0.0
    letters = sum(1 for ch in tok if ch.isalpha())
    return letters / len(tok)


def is_garbage_token(tok: str) -> bool:
    token = str(tok).strip()
    if not token:
        return True
    if _PUNCT_ONLY_RE.fullmatch(token):
        return True
    if token in {"I", "A"} and token.isalpha():
        return False
    if token.isdigit():
        return False
    if len(token) == 1 and not token.isalnum():
        return True
    if len(token) <= 2:
        if any(ch in token for ch in _SHORT_JUNK_CHARS):
            return True
        if _alpha_ratio(token) < 0.6:
            return True
    if _alpha_ratio(token) < 0.45 and len(token) < 6:
        return True
    if token in _KNOWN_JUNK_EXACT or token.lower() in _KNOWN_JUNK_LOWER:
        return True
    return False


def clean_line_tokens(line: str) -> str:
    raw_tokens = normalize_whitespace(line).split(" ")
    kept: list[str] = []
    for raw in raw_tokens:
        token = raw.strip("|\\()")
        if not token:
            continue
        if is_garbage_token(token):
            continue
        kept.append(token)

    filtered: list[str] = []
    for idx, tok in enumerate(kept):
        if tok == "i" and 0 < idx < len(kept) - 1:
            continue
        filtered.append(tok)
    return normalize_whitespace(" ".join(filtered))


def drop_garbage_lines(lines: list[str]) -> list[str]:
    output: list[str] = []
    for line in lines:
        cleaned = clean_line_tokens(line)
        if not cleaned:
            continue
        if len(cleaned) <= 2 and cleaned.lower() in {"i", "ii", "iii"}:
            continue
        if _HEADER_BLEED_RE.fullmatch(cleaned):
            continue
        output.append(cleaned)
    return output


def _strip_hyphen_tail(line: str) -> str:
    normalized = normalize_whitespace(line)
    if _HYPHEN_TAIL_RE.search(normalized):
        normalized = _HYPHEN_TAIL_RE.sub("", normalized)
    return normalize_whitespace(normalized)


def dehyphenate_across_lines(lines: list[str], lookahead: int = 2) -> list[str]:
    merged = list(lines)
    i = 0
    while i < len(merged):
        current = normalize_whitespace(merged[i])
        if not _HYPHEN_TAIL_RE.search(current):
            i += 1
            continue

        target_idx: int | None = None
        next_meaningful: str | None = None
        for j in range(i + 1, min(len(merged), i + lookahead + 1)):
            candidate = normalize_whitespace(merged[j])
            if not candidate:
                continue
            target_idx = j
            next_meaningful = candidate
            break

        if target_idx is None or next_meaningful is None:
            i += 1
            continue
        if _LOWER_START_RE.match(next_meaningful) is None:
            i += 1
            continue

        left = _strip_hyphen_tail(current)
        right = normalize_whitespace(_LEADING_PUNCT_RE.sub("", next_meaningful))
        merged[i] = normalize_whitespace(left + right)
        del merged[i + 1 : target_idx + 1]
        i += 1
    return merged


def looks_like_list_item(s: str) -> bool:
    line = normalize_whitespace(s)
    return bool(_YEAR_LINE_RE.match(line) or _LIST_MARKER_RE.match(line))


def reflow_prose(lines: list[str]) -> str:
    blocks: list[str] = []
    paragraph_parts: list[str] = []

    def flush_paragraph() -> None:
        if paragraph_parts:
            blocks.append(normalize_whitespace(" ".join(paragraph_parts)))
            paragraph_parts.clear()

    for idx, raw in enumerate(lines):
        line = normalize_whitespace(raw)
        if not line:
            flush_paragraph()
            continue

        if looks_like_list_item(line):
            flush_paragraph()
            blocks.append(line)
            continue

        if blocks and looks_like_list_item(blocks[-1]):
            blocks[-1] = normalize_whitespace(blocks[-1] + " " + line)
            continue

        paragraph_parts.append(line)
        if idx + 1 >= len(lines):
            flush_paragraph()
            continue

        next_line = normalize_whitespace(lines[idx + 1])
        if not next_line:
            flush_paragraph()
            continue
        paragraph_now = normalize_whitespace(" ".join(paragraph_parts))
        if (
            len(paragraph_now) > 60
            and _SENTENCE_END_RE.search(paragraph_now)
            and _UPPER_START_RE.match(next_line)
        ):
            flush_paragraph()

    flush_paragraph()
    return "\n\n".join(block for block in blocks if block)


def clean_ocr_text(lines: list[str]) -> str:
    cleaned_lines = [clean_line_tokens(line) for line in lines]
    cleaned_lines = drop_garbage_lines(cleaned_lines)
    cleaned_lines = dehyphenate_across_lines(cleaned_lines)
    return reflow_prose(cleaned_lines)
