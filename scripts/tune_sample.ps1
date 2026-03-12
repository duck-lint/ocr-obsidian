param(
  [string]$BookConfig = "configs/books/sample_book.yaml",
  [string]$PipelineConfig = "configs/pipeline.yaml",
  [int]$MaxPages = 3,
  [string[]]$Scenarios = @("baseline", "conservative_text", "messy_scan_rescue", "highlight_sensitive"),
  [switch]$DryRun
)

<#
.SYNOPSIS
    Run OCR pipeline through multiple tuning scenarios for comparison.

.DESCRIPTION
    This script runs the same sample pages through multiple named tuning scenarios,
    generating separate run outputs for each scenario so you can compare artifacts
    (page_text.json, overlays, highlights, spans) and dial in settings.

.PARAMETER BookConfig
    Path to the book config YAML (default: configs/books/sample_book.yaml)

.PARAMETER PipelineConfig
    Path to the pipeline config YAML (default: configs/pipeline.yaml)

.PARAMETER MaxPages
    Number of pages to process per scenario (default: 3)

.PARAMETER Scenarios
    Array of scenario names to run (default: baseline, conservative_text, messy_scan_rescue, highlight_sensitive)
    Available scenarios: baseline, conservative_text, messy_scan_rescue, highlight_sensitive

.PARAMETER DryRun
    Print actions without writing files

.EXAMPLE
    # Run all default scenarios on 3 pages
    scripts/tune_sample.ps1

.EXAMPLE
    # Run specific scenarios on 5 pages
    scripts/tune_sample.ps1 -MaxPages 5 -Scenarios @("baseline", "messy_scan_rescue")

.EXAMPLE
    # Dry run to preview what will happen
    scripts/tune_sample.ps1 -DryRun
#>

$ErrorActionPreference = "Stop"

function Invoke-PipelineStep {
  param(
    [string]$Scenario,
    [string]$RunId,
    [string]$Command,
    [string[]]$BaseArgs,
    [switch]$IsDryRun
  )

  $stepName = "$Command ($Scenario)"
  Write-Host ""
  Write-Host "=== $stepName ===" -ForegroundColor Cyan

  $args = $BaseArgs + @(
    "--scenario", $Scenario,
    "--run-id", $RunId
  )

  if ($IsDryRun) {
    $args += @("--dry-run")
  }

  & python $args
  if ($LASTEXITCODE -ne 0) {
    throw "Step failed: $stepName (exit code $LASTEXITCODE)"
  }
}

Write-Host "============================================" -ForegroundColor Green
Write-Host "OCR Tuning Sample Runner" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host "Book Config: $BookConfig"
Write-Host "Pipeline Config: $PipelineConfig"
Write-Host "Max Pages: $MaxPages"
Write-Host "Scenarios: $($Scenarios -join ', ')"
Write-Host "Dry Run: $DryRun"
Write-Host ""

$timestamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")

foreach ($scenario in $Scenarios) {
  $runId = "${timestamp}_${scenario}"

  Write-Host ""
  Write-Host "##########################################" -ForegroundColor Yellow
  Write-Host "# Scenario: $scenario" -ForegroundColor Yellow
  Write-Host "# Run ID: $runId" -ForegroundColor Yellow
  Write-Host "##########################################" -ForegroundColor Yellow

  # OCR phase
  $ocrArgs = @(
    "-m", "ingest", "ocr",
    "--book", $BookConfig,
    "--pipeline", $PipelineConfig,
    "--out", "corpus",
    "--runs", "runs",
    "--max-pages", "$MaxPages",
    "--overwrite", "if_same_run"
  )
  Invoke-PipelineStep -Scenario $scenario -RunId $runId -Command "OCR" -BaseArgs $ocrArgs -IsDryRun:$DryRun

  # Highlight detection phase
  $highlightArgs = @(
    "-m", "ingest", "detect-highlights",
    "--book", $BookConfig,
    "--pipeline", $PipelineConfig,
    "--runs", "runs",
    "--max-pages", "$MaxPages",
    "--overwrite", "if_same_run"
  )
  Invoke-PipelineStep -Scenario $scenario -RunId $runId -Command "Detect Highlights" -BaseArgs $highlightArgs -IsDryRun:$DryRun

  # Span generation phase
  $spanArgs = @(
    "-m", "ingest", "make-spans",
    "--book", $BookConfig,
    "--pipeline", $PipelineConfig,
    "--runs", "runs",
    "--corpus", "corpus",
    "--k-before", "2",
    "--k-after", "2",
    "--max-pages", "$MaxPages",
    "--overwrite", "if_same_run"
  )
  Invoke-PipelineStep -Scenario $scenario -RunId $runId -Command "Make Spans" -BaseArgs $spanArgs -IsDryRun:$DryRun

  # Emit Obsidian notes phase
  $vaultStaging = "runs/$runId/obsidian_staging"
  $emitArgs = @(
    "-m", "ingest", "emit-obsidian",
    "--book", $BookConfig,
    "--pipeline", $PipelineConfig,
    "--runs", "runs",
    "--corpus", "corpus",
    "--vault", $vaultStaging,
    "--sidecar-json",
    "--max-pages", "$MaxPages",
    "--overwrite", "if_same_run"
  )
  Invoke-PipelineStep -Scenario $scenario -RunId $runId -Command "Emit Obsidian" -BaseArgs $emitArgs -IsDryRun:$DryRun

  Write-Host ""
  Write-Host "Scenario '$scenario' completed: runs/$runId" -ForegroundColor Green
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "All scenarios completed!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Compare outputs in the 'runs' directory:"
foreach ($scenario in $Scenarios) {
  $runId = "${timestamp}_${scenario}"
  Write-Host "  runs/$runId"
}
Write-Host ""
Write-Host "Artifacts to compare per scenario:"
Write-Host "  - page_text.json (OCR results)"
Write-Host "  - page_overlay.png (OCR visualization)"
Write-Host "  - highlight_mask.png (highlight detection)"
Write-Host "  - highlights_overlay.png (highlight visualization)"
Write-Host "  - spans.json (text spans)"
Write-Host "  - spans_overlay.png (span visualization)"
Write-Host ""
