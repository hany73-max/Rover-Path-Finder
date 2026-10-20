# run_all.ps1 — reproduces all rover path-planning runs.
# Run from the repo root: .\run_all.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---- 0. Ensure venv is up to date ----
Write-Host "`n=== uv sync ===" -ForegroundColor Cyan
uv sync

New-Item -ItemType Directory -Force -Path data, outputs | Out-Null

# ---- 1. Synthetic demo (no download needed) ----
Write-Host "`n=== Synthetic Mars demo ===" -ForegroundColor Cyan
uv run python plan_on_heightmap.py --palette mars --out outputs/demo

# ---- 2. Kasei Valles (HiRISE DTM, 2 m/px, ~78 MB) ----
if (-not (Test-Path "data/kasei.IMG")) {
    Write-Host "`n=== Downloading Kasei Valles DEM (~78 MB) ===" -ForegroundColor Cyan
    curl.exe -L -o data/kasei.IMG `
        "https://www.uahirise.org/PDS/DTM/ESP/ORB_032100_032199/ESP_032193_1990_ESP_031837_1990/DTEED_032193_1990_031837_1990_A01.IMG" `
        --progress-bar
} else {
    Write-Host "`n=== kasei.IMG already present ===" -ForegroundColor Yellow
}

# Look pass — generates ticked axes so you can read off pixel coords
Write-Host "`n=== Kasei: look pass (--ticks) ===" -ForegroundColor Cyan
uv run python plan_on_heightmap.py data/kasei.IMG `
    --ticks --palette mars --out outputs/look

# Route planning
# NOTE: --max-slope 20 returns NO SAFE ROUTE (channel walls ~25 deg).
#       Raised to 25 deg per task instructions.
Write-Host "`n=== Kasei: route planning (max-slope 25 deg) ===" -ForegroundColor Cyan
uv run python plan_on_heightmap.py data/kasei.IMG `
    --start 250 500 `
    --goal 2400 5800 `
    --max-slope 25 `
    --rover-radius 20 `
    --palette mars `
    --out outputs/kasei

# ---- 3. Opportunity Landing Site (HiRISE DTM, 1 m/px, ~9 MB) ----
if (-not (Test-Path "data/opportunity.IMG")) {
    Write-Host "`n=== Downloading Opportunity Landing Site DEM (~9 MB) ===" -ForegroundColor Cyan
    curl.exe -L -o data/opportunity.IMG `
        "https://www.uahirise.org/PDS/DTM/PSP/ORB_001400_001499/PSP_001414_1780_PSP_001612_1780/DTEEC_001414_1780_001612_1780_U01.IMG" `
        --progress-bar
} else {
    Write-Host "`n=== opportunity.IMG already present ===" -ForegroundColor Yellow
}

Write-Host "`n=== Opportunity: route planning (max-slope 20 deg) ===" -ForegroundColor Cyan
uv run python plan_on_heightmap.py data/opportunity.IMG `
    --max-slope 20 `
    --rover-radius 5 `
    --palette mars `
    --out outputs/opportunity

Write-Host "`n=== All runs complete. Outputs in outputs/ ===" -ForegroundColor Green
