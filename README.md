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

Optional corpus export:

```powershell
python -m ingest export-book-text `
  --book configs/books/sample_book.yaml `
  --out corpus
```

## Safety Flags

Relevant commands support:

- `--dry-run`
- `--overwrite {never|if_same_run|always}` (default: `never`)
- `--max-pages N`
- `--run-id <id>`

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

## Current TODOs

- PDF ingestion support (v0 currently supports image folders only).
- Better paragraph/reading-order line grouping.
- More robust highlight color modeling for difficult scans.
