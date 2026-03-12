# OCR Obsidian Ingest

Minimal, machine-first ingestion pipeline with one OCR pass per page, then two downstream products:

- Highlight-triggered excerpt notes for Obsidian.
- Canonical full-book corpus in JSONL for future retrieval/RAG tooling.

The canonical source of truth is `corpus/books/<book_id>/pages.jsonl`. Downstream outputs are derived from this corpus and run artifacts. No re-OCR per excerpt.

## Install

1. Use Python 3.11+.
2. Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Install external OCR binary:
- Tesseract must be installed and available on `PATH`.
- Verify:

```powershell
tesseract --version
```

If `pytesseract` or `opencv-python-headless` is missing, the CLI fails with explicit dependency errors.

## Configuration

- Pipeline defaults: `configs/pipeline.yaml`
- Book config: `configs/books/sample_book.yaml`
- Note template: `templates/obsidian_note.md`

`sample_book.yaml` includes:
- `book_id`
- bibliographic metadata (`title`, `creator`, `year`, `publisher_studio`, `format`)
- `scans_path`
- `vault_out_path`
- note defaults (`note_type`, `note_status`, `note_version`, `YAML_schema_version`, `register`, `tags`)

## Commands

CLI entrypoint:

```powershell
python -m ingest --help
```

Phase 1: OCR spine (canonical corpus)

```powershell
python -m ingest ocr `
  --book configs/books/sample_book.yaml `
  --out corpus `
  --runs runs `
  --max-pages 3
```

Phase 2: Highlight detection

```powershell
python -m ingest detect-highlights `
  --book configs/books/sample_book.yaml `
  --runs runs `
  --run-id <run_id> `
  --max-pages 3
```

Phase 3: Span selection (highlights -> OCR lines + context)

```powershell
python -m ingest make-spans `
  --book configs/books/sample_book.yaml `
  --runs runs `
  --run-id <run_id> `
  --corpus corpus `
  --k-before 2 `
  --k-after 2 `
  --max-pages 3
```

Phase 4: Emit Obsidian notes (+ optional sidecar JSON)

```powershell
python -m ingest emit-obsidian `
  --book configs/books/sample_book.yaml `
  --runs runs `
  --run-id <run_id> `
  --corpus corpus `
  --vault runs/<run_id>/obsidian_staging `
  --sidecar-json `
  --max-pages 3
```

`emit-obsidian` and `export-book-text` both use the same deterministic token+layout-aware text renderer by default.
This rendering does not modify canonical raw OCR evidence in `pages.jsonl`.
Use `--no-clean-text` on either command to keep raw OCR line breaks.

Optional corpus export:

```powershell
python -m ingest export-book-text `
  --book configs/books/sample_book.yaml `
  --out corpus `
  --format txt
```

Markdown export:

```powershell
python -m ingest export-book-text `
  --book configs/books/sample_book.yaml `
  --out corpus `
  --format md
```

## Safety Flags

Relevant commands support:

- `--dry-run`
- `--overwrite {never|if_same_run|always}` (default: `never`)
- `--max-pages N`
- `--run-id <id>`
- `--no-clean-text` (`emit-obsidian`, `export-book-text`)

Overwrite semantics are fail-closed by default:

- `never`: fail if output already exists.
- `if_same_run`: allow overwrite only under `runs/<run_id>/...`.
- `always`: allow replacement.

Canonical corpus (`corpus/.../pages.jsonl`) remains fail-closed unless `--overwrite always`.

## Output Layout

```text
corpus/
  books/
    <book_id>/
      pages.jsonl
      book.txt
      book.md

runs/
  <run_id>/
    book_<book_id>/
      page_0001/
        page_text.json
        page_overlay.png
        highlight_mask.png
        highlight_candidates.json
        highlights_overlay.png
        spans.json
        spans_overlay.png
    obsidian_staging/
      <book_id>/
        <note>.md
        <note>.span.json
```

## Obsidian Frontmatter Constraint

No new YAML frontmatter keys are introduced.

Emitter uses only existing schema keys:

- `uuid`
- `note_version`
- `YAML_schema_version`
- `note_type`
- `note_status`
- `tags`
- `format`
- `title`
- `creator`
- `year`
- `publisher_studio`
- `register`

Provenance fields such as page number, line ids, run id, bbox coordinates, and config hash are written in:

- note body under `## Source`
- optional sidecar JSON `<note>.span.json`

## Smoke Test Script

Use:

```powershell
scripts/dev_smoke_test.ps1 -BookConfig configs/books/sample_book.yaml
```

The script runs OCR -> highlight detection -> span generation -> note emission with `--max-pages 3`.

Frontmatter rendering smoke check:

```powershell
python scripts/frontmatter_smoke_check.py
```

## Tuning Workflow

The pipeline supports tuning OCR, highlight detection, QA, and span selection settings through:

1. **Named Scenarios** - Pre-configured tuning profiles for common use cases
2. **CLI Overrides** - Direct command-line control of individual settings
3. **YAML Configuration** - Base settings in `configs/pipeline.yaml`

Configuration precedence: `built-in defaults < pipeline YAML < CLI overrides`.

### Running Multiple Tuning Scenarios

To compare OCR results across multiple scenarios on the same sample pages:

```powershell
scripts/tune_sample.ps1 -MaxPages 3
```

This runs all default scenarios (baseline, conservative_text, messy_scan_rescue, highlight_sensitive) and outputs separate runs for comparison.

Run specific scenarios:

```powershell
scripts/tune_sample.ps1 -Scenarios @("baseline", "messy_scan_rescue") -MaxPages 5
```

Each scenario produces complete artifacts in `runs/<timestamp>_<scenario>/`:
- `page_text.json` - OCR results
- `page_overlay.png` - OCR visualization
- `highlight_mask.png` - Highlight detection mask
- `highlights_overlay.png` - Highlight visualization
- `spans.json` - Text spans with context
- `spans_overlay.png` - Span visualization

### Available Scenarios

- **baseline**: Current defaults (no overrides)
- **conservative_text**: Tighter QA thresholds, fewer false positives
- **messy_scan_rescue**: More forgiving for noisy/low-confidence scans
- **highlight_sensitive**: Permissive highlight detection for faint markers

### Using Scenarios in Commands

Apply a scenario to any command with `--scenario`:

```powershell
python -m ingest ocr `
  --book configs/books/sample_book.yaml `
  --scenario messy_scan_rescue `
  --out corpus `
  --runs runs `
  --max-pages 3
```

### CLI Override Flags

Override individual settings directly on the command line. CLI overrides take precedence over both scenario and YAML settings.

**OCR Settings:**
- `--ocr-psm <int>` - Page segmentation mode (default: 6)
- `--ocr-language <str>` - Language code (default: "eng")
- `--ocr-line-y-tolerance-px <int>` - Line grouping Y tolerance (default: 14)

**Highlight Settings:**
- `--highlight-min-area <int>` - Minimum area threshold (default: 120)
- `--highlight-kernel-size <int>` - Morphological kernel size (default: 5)
- `--highlight-edge-margin-px <int>` - Edge margin filter (default: 25)
- `--highlight-max-hw-ratio <float>` - Max height/width ratio (default: 3.0)
- `--highlight-max-height-frac <float>` - Max height fraction (default: 0.15)

**QA Settings:**
- `--qa-min-avg-word-conf <float>` - Min average word confidence (default: 58.0)
- `--qa-max-garbage-ratio <float>` - Max garbage ratio (default: 0.22)
- `--qa-max-pipe-ratio <float>` - Max pipe character ratio (default: 0.04)
- `--qa-min-alpha-ratio <float>` - Min alpha character ratio (default: 0.45)

**Span Settings:**
- `--span-min-overlap-frac <float>` - Min overlap fraction (default: 0.02)
- `--span-min-x-overlap-px <int>` - Min X overlap in pixels (default: 40)
- `--span-max-overlap-lines <int>` - Max overlap lines (default: 8)

### Combining Scenarios and CLI Overrides

Start with a scenario and tweak individual settings:

```powershell
python -m ingest ocr `
  --book configs/books/sample_book.yaml `
  --scenario messy_scan_rescue `
  --ocr-psm 3 `
  --qa-min-avg-word-conf 45.0 `
  --out corpus `
  --runs runs `
  --max-pages 3
```

Or override without a scenario:

```powershell
python -m ingest detect-highlights `
  --book configs/books/sample_book.yaml `
  --highlight-min-area 80 `
  --highlight-kernel-size 3 `
  --runs runs `
  --run-id <run_id> `
  --max-pages 3
```

## Current TODOs

- PDF ingestion support (v0 currently supports image folders only).
- Better paragraph/reading-order line grouping.
- More robust highlight color modeling for difficult scans.
