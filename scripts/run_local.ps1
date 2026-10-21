# Run without Docker on a Windows desktop: creates a venv, installs CUDA PyTorch, starts the API.
param([ValidateSet("cu124", "cu121", "cpu")][string]$Torch = "cu124")
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Test-Path .venv)) { python -m venv .venv }
& .\.venv\Scripts\Activate.ps1
if (-not (Test-Path .env)) { Copy-Item .env.example .env; Write-Host "Created .env - edit API_KEYS first." }

pip install torch torchvision --index-url "https://download.pytorch.org/whl/$Torch"
pip install -r requirements.txt -r requirements-ml.txt -r requirements-azure.txt
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
uvicorn app.main:app --host 127.0.0.1 --port 8000
