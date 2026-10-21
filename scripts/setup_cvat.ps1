# Installs and starts a self-hosted CVAT next to this project (needs Docker Desktop + git).
# CVAT is run from its own official compose files so upgrades stay simple.
param(
    [string]$Version = "v2.25.0",          # CVAT release tag; check https://github.com/cvat-ai/cvat/releases
    [string]$Dir = (Join-Path $PSScriptRoot "..\cvat")
)
$ErrorActionPreference = "Stop"

if (-not (Test-Path $Dir)) {
    git clone --depth 1 --branch $Version https://github.com/cvat-ai/cvat.git $Dir
}
Push-Location $Dir
try {
    $env:CVAT_VERSION = $Version
    docker compose up -d
    Write-Host "`nCVAT is starting at http://localhost:8080 (first start takes a few minutes)."
    Write-Host "Create the admin user with:"
    Write-Host "  docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'"
} finally {
    Pop-Location
}
