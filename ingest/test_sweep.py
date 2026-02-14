from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

from ingest.sweep import SweepThresholds, compute_metrics, decide_verdict, extract_span_text, run_sweep


def _fake_page_record() -> dict:
    return {
        "book_id": "book1",
        "page_num": 1,
        "scan_relpath": "page_0001.png",
        "lines": [
            {
                "line_id": "p1_l1",
                "text": "Alpha beta",
                "words": [
                    {"text": "Alpha", "confidence": 90.0},
                    {"text": "beta", "confidence": 80.0},
                ],
            },
            {
                "line_id": "p1_l2",
                "text": "Gamma delta",
                "words": [
                    {"text": "Gamma", "confidence": 88.0},
                    {"text": "delta", "confidence": 86.0},
                ],
            },
        ],
    }


class SweepTests(unittest.TestCase):
    def test_extract_span_text_and_confidence_count(self) -> None:
        page = _fake_page_record()
        text, confidence_count = extract_span_text(page, ["p1_l1", "p1_l2"])
        self.assertEqual(text, "Alpha beta\nGamma delta")
        self.assertEqual(confidence_count, 4)

    def test_verdict_classification_pass_warn_fail(self) -> None:
        thresholds = SweepThresholds()

        pass_metrics = compute_metrics(
            text="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            avg_word_conf=92.0,
            line_count=2,
        )
        pass_verdict, pass_reasons = decide_verdict(pass_metrics, thresholds)
        self.assertEqual(pass_verdict, "PASS")
        self.assertEqual(pass_reasons, [])

        warn_metrics = compute_metrics(
            text="A" * 2101,
            avg_word_conf=92.0,
            line_count=2,
        )
        warn_verdict, warn_reasons = decide_verdict(warn_metrics, thresholds)
        self.assertEqual(warn_verdict, "WARN")
        self.assertTrue(any("char_count" in reason for reason in warn_reasons))

        fail_metrics = compute_metrics(
            text="|{}[]<>^^",
            avg_word_conf=40.0,
            line_count=1,
        )
        fail_verdict, fail_reasons = decide_verdict(fail_metrics, thresholds)
        self.assertEqual(fail_verdict, "FAIL")
        self.assertTrue(any("alpha_ratio" in reason for reason in fail_reasons))
        self.assertTrue(any("avg_word_conf" in reason for reason in fail_reasons))

    def test_run_sweep_writes_reports(self) -> None:
        root = Path.cwd() / f".tmp_test_sweep_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=False)
        try:
            corpus_dir = root / "corpus"
            sidecars_dir = root / "sidecars"
            notes_dir = root / "notes"
            out_dir = root / "qa"

            (corpus_dir / "books" / "book1").mkdir(parents=True, exist_ok=True)
            sidecars_dir.mkdir(parents=True, exist_ok=True)
            notes_dir.mkdir(parents=True, exist_ok=True)

            pages_path = corpus_dir / "books" / "book1" / "pages.jsonl"
            page_record = _fake_page_record()
            pages_path.write_text(json.dumps(page_record) + "\n", encoding="utf-8")

            sidecar_payload = {
                "book_id": "book1",
                "page_num": 1,
                "span_id": "p1_s1",
                "line_ids": ["p1_l1", "p1_l2"],
                "scan_relpath": "page_0001.png",
                "printed_page": "v",
            }
            (sidecars_dir / "note_a.span.json").write_text(
                json.dumps(sidecar_payload, indent=2),
                encoding="utf-8",
            )
            (notes_dir / "note_a.md").write_text("# Note A\n", encoding="utf-8")

            args = SimpleNamespace(
                corpus_dir=corpus_dir,
                sidecars_dir=sidecars_dir,
                notes_dir=notes_dir,
                glob="*.span.json",
                out_dir=out_dir,
                max_items=None,
                dry_run=False,
                overwrite="never",
                thresholds=None,
                fail_alpha_min=None,
                fail_conf_min=None,
                fail_garbage_max=None,
                warn_line_max=None,
                warn_char_max=None,
                warn_pipe_max=None,
            )

            result = run_sweep(args)
            self.assertEqual(result, 0)

            report_json = out_dir / "qa_report.json"
            report_md = out_dir / "qa_report.md"
            self.assertTrue(report_json.exists())
            self.assertTrue(report_md.exists())

            records = json.loads(report_json.read_text(encoding="utf-8"))
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record["book_id"], "book1")
            self.assertEqual(record["span_id"], "p1_s1")
            self.assertEqual(record["printed_page"], "v")
            self.assertEqual(record["verdict"], "PASS")
            self.assertIsNotNone(record["note_path"])
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
