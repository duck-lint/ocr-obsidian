from __future__ import annotations

from typing import Any


DEFAULT_QA_THRESHOLDS: dict[str, float] = {
    "min_avg_word_conf": 58.0,
    "max_garbage_ratio": 0.22,
    "max_pipe_ratio": 0.04,
    "min_alpha_ratio": 0.45,
}


def resolve_qa_thresholds(pipeline_config: dict[str, Any] | None) -> dict[str, float]:
    merged = dict(DEFAULT_QA_THRESHOLDS)
    if not isinstance(pipeline_config, dict):
        return merged
    qa_cfg = pipeline_config.get("qa", {})
    if not isinstance(qa_cfg, dict):
        return merged
    for key in DEFAULT_QA_THRESHOLDS:
        if key in qa_cfg:
            try:
                merged[key] = float(qa_cfg[key])
            except (TypeError, ValueError):
                continue
    return merged


def _line_text(line: dict[str, Any]) -> str:
    words = line.get("words")
    if isinstance(words, list) and words:
        joined = " ".join(str(word.get("text", "")) for word in words if str(word.get("text", "")).strip())
        return joined.strip()
    return str(line.get("text", "")).strip()


def compute_text_metrics(lines: list[dict[str, Any]]) -> dict[str, Any]:
    texts: list[str] = []
    confidences: list[float] = []
    for line in lines:
        text = _line_text(line)
        if not text:
            continue
        texts.append(text)
        words = line.get("words")
        if isinstance(words, list):
            for word in words:
                raw_conf = word.get("confidence")
                try:
                    conf = float(raw_conf)
                except (TypeError, ValueError):
                    continue
                if conf >= 0:
                    confidences.append(conf)

    content = "\n".join(texts)
    char_count = len(content)
    line_count = len(texts)
    alpha_count = sum(1 for ch in content if ch.isalpha())
    alnum_count = sum(1 for ch in content if ch.isalnum())
    nonspace_count = sum(1 for ch in content if not ch.isspace())
    garbage_count = sum(1 for ch in content if (not ch.isalnum()) and (not ch.isspace()))
    pipe_count = content.count("|")

    alpha_ratio = (alpha_count / alnum_count) if alnum_count else 0.0
    garbage_ratio = (garbage_count / nonspace_count) if nonspace_count else 0.0
    pipe_ratio = (pipe_count / nonspace_count) if nonspace_count else 0.0
    avg_word_conf = (sum(confidences) / len(confidences)) if confidences else None

    return {
        "char_count": char_count,
        "line_count": line_count,
        "avg_word_conf": avg_word_conf,
        "alpha_ratio": alpha_ratio,
        "garbage_ratio": garbage_ratio,
        "pipe_ratio": pipe_ratio,
    }


def is_obviously_empty_or_garbage(
    metrics: dict[str, Any],
    thresholds: dict[str, float] | None = None,
) -> bool:
    cfg = dict(DEFAULT_QA_THRESHOLDS)
    if thresholds:
        cfg.update(thresholds)

    char_count = int(metrics.get("char_count", 0))
    line_count = int(metrics.get("line_count", 0))
    alpha_ratio = float(metrics.get("alpha_ratio", 0.0))
    garbage_ratio = float(metrics.get("garbage_ratio", 0.0))
    pipe_ratio = float(metrics.get("pipe_ratio", 0.0))
    avg_word_conf = metrics.get("avg_word_conf")

    if line_count == 0 or char_count == 0:
        return True
    if char_count < 12 and alpha_ratio < 0.35:
        return True
    if pipe_ratio > cfg["max_pipe_ratio"] and alpha_ratio < cfg["min_alpha_ratio"]:
        return True
    if garbage_ratio > cfg["max_garbage_ratio"] and alpha_ratio < cfg["min_alpha_ratio"]:
        return True
    if avg_word_conf is not None and float(avg_word_conf) < cfg["min_avg_word_conf"] and garbage_ratio > (
        cfg["max_garbage_ratio"] * 0.75
    ):
        return True
    return False
