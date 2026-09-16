
$ErrorActionPreference = "Stop"
$oldRepo = "C:\Users\dell\rag-adversarial-robustness"

Write-Host ""
Write-Host "=== 1. Remove the garbled junk file from the failed manual copy ===" -ForegroundColor Cyan
$junk = Get-ChildItem -Force | Where-Object { $_.Name -like "*Destination*" -or $_.Name -like "*ersdell*" }
if ($junk) {
    foreach ($f in $junk) {
        Write-Host "Removing: $($f.Name)" -ForegroundColor Yellow
        Remove-Item -LiteralPath $f.FullName -Force
    }
} else {
    Write-Host "No junk file found - already clean or named differently. Continuing." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== 2. Confirm the source folder actually exists before copying ===" -ForegroundColor Cyan
if (-not (Test-Path $oldRepo)) {
    Write-Host "Source repo not found at $oldRepo - stopping." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=== 3. Copy phase1_results_complete ===" -ForegroundColor Cyan
robocopy "$oldRepo\phase1_results_complete" ".\phase1_results_complete" /E /NFL /NDL /NJH
$roboCode1 = $LASTEXITCODE

Write-Host ""
Write-Host "=== 4. Copy phase1_results_partial ===" -ForegroundColor Cyan
robocopy "$oldRepo\phase1_results_partial" ".\phase1_results_partial" /E /NFL /NDL /NJH
$roboCode2 = $LASTEXITCODE

Write-Host ""
Write-Host "=== 5. Copy baseline_raw_qwen3.jsonl and PHASE2_PREP_LOG.md ===" -ForegroundColor Cyan
Copy-Item -LiteralPath "$oldRepo\baseline_raw_qwen3.jsonl" -Destination ".\baseline_raw_qwen3.jsonl" -Force
Copy-Item -LiteralPath "$oldRepo\PHASE2_PREP_LOG.md" -Destination ".\PHASE2_PREP_LOG.md" -Force

Write-Host ""
Write-Host "=== 6. Verify ===" -ForegroundColor Cyan
$completeCount = (Get-ChildItem .\phase1_results_complete -File -ErrorAction SilentlyContinue | Measure-Object).Count
$sourceCompleteCount = (Get-ChildItem "$oldRepo\phase1_results_complete" -File -ErrorAction SilentlyContinue | Measure-Object).Count
Write-Host "phase1_results_complete: $completeCount files here vs $sourceCompleteCount in source"

if (Test-Path ".\PHASE2_PREP_LOG.md") {
    Write-Host "PHASE2_PREP_LOG.md copied successfully. First 3 lines:" -ForegroundColor Green
    Get-Content ".\PHASE2_PREP_LOG.md" -TotalCount 3
} else {
    Write-Host "PHASE2_PREP_LOG.md did NOT copy - check for errors above." -ForegroundColor Red
}

if (Test-Path ".\baseline_raw_qwen3.jsonl") {
    Write-Host "baseline_raw_qwen3.jsonl copied successfully." -ForegroundColor Green
} else {
    Write-Host "baseline_raw_qwen3.jsonl did NOT copy - check for errors above." -ForegroundColor Red
}

Write-Host ""
Write-Host "=== 7. Robocopy exit codes (0-7 = success, 8+ = real failure) ===" -ForegroundColor Cyan
Write-Host "phase1_results_complete robocopy exit code: $roboCode1"
Write-Host "phase1_results_partial robocopy exit code: $roboCode2"

Write-Host ""
Write-Host "=== Done. Review the counts and messages above before proceeding. ===" -ForegroundColor Yellow
