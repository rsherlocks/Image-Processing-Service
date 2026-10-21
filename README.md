# AI Image Processing Service

Self-hosted FastAPI server: upload images → YOLOv11 (Ultralytics, PyTorch, NVIDIA CUDA) →
annotated image + JSON detections saved on your own disk. Optional Azure Blob backup and
CVAT dataset round-trip.

```
client ──HTTP──▶ FastAPI (API-key auth, validation, rate limit)
                   ├─▶ YOLOv11 on GPU/CPU
                   ├─▶ Local disk: inputs/ processed/ results/ + SQLite index
                   ├─▶ (optional) Azure Blob Storage backup
                   └─▶ YOLO-1.1 dataset export ⇄ self-hosted CVAT
```

## Quick start (Docker, NVIDIA GPU)

Prerequisites: Docker Desktop (WSL2 backend) and a recent NVIDIA driver. Verify GPU access with
`docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`.

```powershell
copy .env.example .env
# edit .env: set API_KEYS (python -c "import secrets; print(secrets.token_urlsafe(32))")
# and HOST_STORAGE_DIR (e.g. D:/AIImageData) for where files are stored
docker compose up --build
```

- Swagger UI: <http://localhost:8000/docs> (click **Authorize**, paste your API key)
- Health: <http://localhost:8000/health> (shows device, `cuda_available`)
- No GPU: `docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build`

## Quick start (no Docker, Windows)

Requires Python 3.10–3.12. `./scripts/run_local.ps1` creates a venv, installs CUDA PyTorch
(`-Torch cu124|cu121|cpu`) and starts the server on <http://127.0.0.1:8000>.

## API

All `/api/v1/*` routes need header `X-API-Key`.

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/images` | Upload one image (`file`, optional `conf`) → detections |
| POST | `/api/v1/images/batch` | Up to 20 images; per-file errors |
| GET | `/api/v1/images` | List (`limit`, `offset`) |
| GET | `/api/v1/images/{id}` | Record + detections |
| GET | `/api/v1/images/{id}/original` · `/processed` | Download images |
| DELETE | `/api/v1/images/{id}` | Delete from disk and index |
| POST | `/api/v1/sync/azure` | Back up everything not yet in Azure |
| GET | `/api/v1/dataset/export` | CVAT-importable YOLO 1.1 zip |
| GET | `/health` | Public status |

```bash
curl -H "X-API-Key: $KEY" -F "file=@photo.jpg" http://localhost:8000/api/v1/images
```

## Storage layout

```
<HOST_STORAGE_DIR>/
  inputs/2026/10/21/<id>.jpg      original bytes, untouched
  processed/2026/10/21/<id>.jpg   boxes drawn
  results/2026/10/21/<id>.json    detections (source of truth)
  index.sqlite3                   listing index (rebuildable from results/)
```

Back up the whole folder, or enable Azure.

## Azure Blob backup (optional)

Set in `.env`: `AZURE_ENABLED=true`, `AZURE_STORAGE_CONNECTION_STRING=...`, `AZURE_CONTAINER=...`.
With `AZURE_AUTO_UPLOAD=true` each result uploads in the background after processing; failures are
logged and never fail the request. `POST /api/v1/sync/azure` retries anything not yet synced.
Blob names mirror the local relative paths. This is one-way backup (local → Azure).

## CVAT

See [docs/CVAT.md](docs/CVAT.md): install with `scripts/setup_cvat.ps1`, import exported
pre-annotations, correct them, export labels and fine-tune; point `YOLO_MODEL` at the new weights.

## Configuration

Everything is an environment variable or `.env` entry; see [.env.example](.env.example) for the
full list (model, device, thresholds, limits, CORS, Azure).

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Tests use a fake detector, so they need neither PyTorch nor model weights. Postman: import
`postman/AI_Image_Service.postman_collection.json`, set `baseUrl`/`apiKey`, pick an image in the
Upload request, and run the collection (it chains the image id automatically).

## Security notes

- API-key auth (constant-time compare); no keys configured = everything rejected.
- Uploads are decoded with Pillow: real format checked (JPEG/PNG/WEBP/BMP), size and pixel limits,
  client filename/content-type never used for paths; ids are validated hex; paths are confined to
  the storage root.
- Container runs as non-root; the port is bound to localhost by default. Use a TLS reverse proxy
  (Caddy, nginx, Traefik) before exposing it beyond your machine.
- Built-in rate limiter is per process; use a gateway for multi-replica setups.

## Scaling

Inference is serialised per process (one GPU job at a time). To scale: run several uvicorn
workers/containers (each loads the model; mind VRAM), put a load balancer in front, and move
`STORAGE_DIR` to shared storage and the SQLite index to Postgres (isolated in
`app/services/storage.py`).

## Layout

```
app/  main.py config.py security.py schemas.py deps.py
      api/images.py
      services/ detector.py processor.py storage.py azure_sync.py dataset.py
tests/  postman/  docs/  scripts/  Dockerfile  docker-compose*.yml
```
