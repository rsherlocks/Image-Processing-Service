# CVAT (self-hosted annotation) integration

CVAT runs as its own Docker Compose stack (official images), separate from this API.
This project feeds it with data and takes corrected labels back for retraining.

## 1. Install CVAT

Requires Docker Desktop and git.

```powershell
./scripts/setup_cvat.ps1          # Linux/macOS: ./scripts/setup_cvat.sh
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

Open <http://localhost:8080>. CVAT's UI is on port 8080, so it does not clash with this API on
port 8000. The release tag in the script is a default I could not verify offline; check it
against <https://github.com/cvat-ai/cvat/releases> and pass the one you want.

## 2. Send processed data to CVAT

1. Upload images to the API as usual (model detections are saved with each image).
2. Download a dataset: `GET /api/v1/dataset/export` (optionally `?ids=<id>&ids=<id>`) in Swagger
   or Postman. The zip is in **YOLO 1.1** format (`obj.names`, `obj.data`, `train.txt`,
   `obj_train_data/`).
3. In CVAT: *Projects → Create from backup / Create new project*, then create a task using the
   same images (you can use the originals in `<HOST_STORAGE_DIR>/inputs/`), then
   *Actions → Upload annotations → YOLO 1.1* and select the zip.
   The detections show up as editable boxes: correct them, add missing ones.

## 3. Get corrected labels back

In CVAT: *Actions → Export task dataset → YOLO 1.1* (or *Ultralytics YOLO Detection 1.0* if
your CVAT version offers it). Use that dataset to fine-tune:

```bash
yolo detect train model=yolo11n.pt data=data.yaml epochs=50 imgsz=640 device=0
```

Put the resulting `best.pt` in `./models/` and set `YOLO_MODEL=/models/best.pt` in `.env`.

## 4. Optional: let CVAT read the image folder directly

Mount the storage folder into CVAT's `cvat_server` and `cvat_worker_*` services with a
`docker-compose.override.yml` in the `cvat/` folder (CVAT "share" feature):

```yaml
services:
  cvat_server:
    volumes:
      - cvat_share:/home/django/share:ro
volumes:
  cvat_share:
    driver_opts:
      type: none
      device: D:/AIImageData/inputs     # your HOST_STORAGE_DIR
      o: bind
```

Add the same `cvat_share` volume to the `cvat_worker_import` / `cvat_worker_chunks` services, then
restart. Images then appear under *Connected file share* when creating a task.
