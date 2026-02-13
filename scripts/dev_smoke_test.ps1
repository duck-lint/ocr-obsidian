param(
  [string]$BookConfig = "configs/books/sample_book.yaml",
  [string]$PipelineConfig = "configs/pipeline.yaml",
  [int]$MaxPages = 3,
  [string]$RunId = "",
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RunId)) {
  $RunId = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
}

$dryFlag = @()
if ($DryRun) {
  $dryFlag = @("--dry-run")
}

function Invoke-Step {
  param(
    [string]$Name,
    [string[]]$CommandArgs
  )
  Write-Host "=== $Name ==="
  & python @CommandArgs
  if ($LASTEXITCODE -ne 0) {
    throw "Step failed: $Name (exit code $LASTEXITCODE)"
  }
}

$vaultStaging = "runs/$RunId/obsidian_staging"

$ocrArgs = @(
  "-m", "ingest", "ocr",
  "--book", $BookConfig,
  "--pipeline", $PipelineConfig,
  "--out", "corpus",
  "--runs", "runs",
  "--run-id", $RunId,
  "--max-pages", "$MaxPages"
) 
$ocrArgs += $dryFlag
Invoke-Step -Name "OCR" -CommandArgs $ocrArgs

$highlightArgs = @(
  "-m", "ingest", "detect-highlights",
  "--book", $BookConfig,
  "--pipeline", $PipelineConfig,
  "--runs", "runs",
  "--run-id", $RunId,
  "--max-pages", "$MaxPages"
)
$highlightArgs += $dryFlag
Invoke-Step -Name "Detect Highlights" -CommandArgs $highlightArgs

$spanArgs = @(
  "-m", "ingest", "make-spans",
  "--book", $BookConfig,
  "--pipeline", $PipelineConfig,
  "--runs", "runs",
  "--run-id", $RunId,
  "--corpus", "corpus",
  "--k-before", "2",
  "--k-after", "2",
  "--max-pages", "$MaxPages"
)
$spanArgs += $dryFlag
Invoke-Step -Name "Make Spans" -CommandArgs $spanArgs

$emitArgs = @(
  "-m", "ingest", "emit-obsidian",
  "--book", $BookConfig,
  "--pipeline", $PipelineConfig,
  "--runs", "runs",
  "--run-id", $RunId,
  "--corpus", "corpus",
  "--vault", $vaultStaging,
  "--sidecar-json",
  "--max-pages", "$MaxPages"
)
$emitArgs += $dryFlag
Invoke-Step -Name "Emit Obsidian" -CommandArgs $emitArgs

$exportArgs = @(
  "-m", "ingest", "export-book-text",
  "--book", $BookConfig,
  "--pipeline", $PipelineConfig,
  "--out", "corpus"
)
$exportArgs += $dryFlag
Invoke-Step -Name "Export Book Text" -CommandArgs $exportArgs

Write-Host "Smoke test finished. run_id=$RunId"
