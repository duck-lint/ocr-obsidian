from __future__ import annotations

import re
from typing import Any


_SPACE_RE = re.compile(r"\s+")
_PIPE_ONLY_RE = re.compile(r"^\|+$")
_STRONG_BREAK_RE = re.compile(r"""[.!?]["')\]]*$""")
_HYPHEN_TAIL_RE = re.compile(r"-\s*[)\]}\"'`.:;,_!]*$")
_LEADING_PUNCT_RE = re.compile(r"^[\s\|\\\"'`.:;,_\-\u2013\u2014()\[\]{}<>!]+")
_LIST_LINE_RE = re.compile(r"^(?:\d{3,4}\s|[-*\u2022]\s)")
_LOWER_START_RE = re.compile(r"^[a-z]")

_LOW_CONF_SPEW = {"fi", "fl", "hl", "hh", "th", "th!", "i|", "|i", "l|", "il"}


def _normalize_spaces(text: str) -> str:
    return _SPACE_RE.sub(" ", text).strip()


def _alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    alpha_count = sum(1 for ch in text if ch.isalpha())
    return alpha_count / len(text)


def _token_is_junk(token: str, confidence: float) -> bool:
    value = str(token).strip()
    conf = float(confidence)
    if not value:
        return True
    if _PIPE_ONLY_RE.fullmatch(value):
        return True
    if "|" in value and conf < 85.0:
        return True
    if "\\" in value and conf < 85.0:
        return True
    if len(value) == 1 and not value.isalnum():
        return True
    if len(value) <= 2 and value.lower() in _LOW_CONF_SPEW and conf < 85.0:
        return True
    if len(value) <= 2 and value.isalpha() and value not in {"I", "A", "a"} and conf < 55.0:
        return True
    if _alpha_ratio(value) < 0.4 and len(value) < 5 and conf < 70.0:
        return True
    return False


def _line_tokens(line: dict[str, Any]) -> list[tuple[str, float]]:
    words = line.get("words")
    if isinstance(words, list) and words:
        tokens: list[tuple[str, float]] = []
        for word in words:
            text = str(word.get("text", ""))
            conf = float(word.get("confidence", 100.0))
            tokens.append((text, conf))
        return tokens

    raw_text = str(line.get("text", ""))
    return [(tok, 100.0) for tok in raw_text.split()]


def _clean_line(line: dict[str, Any]) -> str:
    kept: list[str] = []
    for raw, conf in _line_tokens(line):
        token = _normalize_spaces(raw).strip("|\\()")
        if not token:
            continue
        if _token_is_junk(token, conf):
            continue
        kept.append(token)

    text = _normalize_spaces(" ".join(tok for tok in kept if tok != "|"))
    if not text:
        return ""
    if _PIPE_ONLY_RE.fullmatch(text):
        return ""
    if len(text) == 1 and not text.isalnum():
        return ""
    return text


def _merge_hyphen_breaks(lines: list[str]) -> list[str]:
    merged: list[str] = []
    i = 0
    while i < len(lines):
        current = lines[i]
        if i + 1 < len(lines) and _HYPHEN_TAIL_RE.search(current):
            next_line = lines[i + 1]
            if _LOWER_START_RE.match(next_line):
                left = _HYPHEN_TAIL_RE.sub("", current).rstrip()
                right = _LEADING_PUNCT_RE.sub("", next_line).lstrip()
                merged.append(_normalize_spaces(left + right))
                i += 2
                continue
        merged.append(current)
        i += 1
    return merged


def _looks_continuation(current: str, next_line: str) -> bool:
    if _LIST_LINE_RE.match(current) or _LIST_LINE_RE.match(next_line):
        return False
    if _STRONG_BREAK_RE.search(current):
        return False
    return _LOWER_START_RE.match(next_line) is not None


def render_lines(lines: list[dict[str, Any]]) -> str:
    cleaned_lines = [_clean_line(line) for line in lines]
    cleaned_lines = [line for line in cleaned_lines if line]
    if not cleaned_lines:
        return ""

    cleaned_lines = _merge_hyphen_breaks(cleaned_lines)
    blocks: list[str] = []
    paragraph = cleaned_lines[0]
    for idx in range(1, len(cleaned_lines)):
        next_line = cleaned_lines[idx]
        if _looks_continuation(paragraph, next_line):
            paragraph = _normalize_spaces(paragraph + " " + next_line)
        else:
            blocks.append(paragraph)
            paragraph = next_line
    blocks.append(paragraph)
    return "\n\n".join(_normalize_spaces(block) for block in blocks if _normalize_spaces(block))
