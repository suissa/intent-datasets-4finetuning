$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot
$Venv = Join-Path $RepoRoot ".venv-tev"
$Python = Join-Path $Venv "Scripts\python.exe"

Write-Host ""
Write-Host "=== Tev1 Legal Fine-tuning / Windows ===" -ForegroundColor Cyan
Write-Host "Repo: $RepoRoot"
Write-Host ""

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python was not found in PATH. Install Python 3.12+ and reopen PowerShell."
}

if (-not (Test-Path $Python)) {
    Write-Host "[1/4] Creating Python environment..." -ForegroundColor Yellow
    python -m venv $Venv
}

Write-Host "[2/4] Installing/updating training dependencies..." -ForegroundColor Yellow
& $Python -m pip install --upgrade pip
& $Python -m pip install -r (Join-Path $PSScriptRoot "requirements-tev.txt")

Write-Host "[3/4] Preparing Tev1 instruction dataset..." -ForegroundColor Yellow
& $Python (Join-Path $PSScriptRoot "tev_prepare.py")

$epochs = if ($env:TEV_EPOCHS) { [double]$env:TEV_EPOCHS } else { 1.0 }
$maxLength = if ($env:TEV_MAX_LENGTH) { [int]$env:TEV_MAX_LENGTH } else { 768 }
$gradAccum = if ($env:TEV_GRAD_ACCUM) { [int]$env:TEV_GRAD_ACCUM } else { 16 }
$batchSize = if ($env:TEV_BATCH_SIZE) { [int]$env:TEV_BATCH_SIZE } else { 1 }
$lr = if ($env:TEV_LR) { [double]$env:TEV_LR } else { 5e-5 }

Write-Host "[4/4] Starting LoRA training..." -ForegroundColor Yellow
Write-Host "epochs=$epochs batch=$batchSize grad_accum=$gradAccum max_length=$maxLength lr=$lr"

$trainScript = Join-Path $PSScriptRoot "tev_train.py"
& $Python $trainScript --epochs $epochs --batch-size $batchSize --grad-accum $gradAccum --max-length $maxLength --learning-rate $lr

Write-Host ""
Write-Host "Training finished." -ForegroundColor Green
Write-Host "Adapter: outputs\tev1-legal\final"
Write-Host ""
Write-Host "Evaluate with:"
Write-Host "  $Python scripts\finetuning\tev_eval.py"
