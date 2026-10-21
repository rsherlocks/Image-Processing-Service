import os
from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Path,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from app.config import Settings, get_settings
from app.deps import get_azure, get_detector, get_storage
from app.schemas import BatchItemError, BatchResult, ImageList, ImageRecord, SyncResult
from app.security import rate_limit, require_api_key
from app.services.azure_sync import AzureSync
from app.services.dataset import build_yolo_zip
from app.services.detector import Detector
from app.services.processor import process_upload
from app.services.storage import Storage

router = APIRouter(
    prefix="/api/v1", dependencies=[Depends(require_api_key), Depends(rate_limit)]
)

ImageId = Annotated[str, Path(pattern=r"^[0-9a-f]{32}$", description="Image id (32 hex chars)")]
Conf = Annotated[float | None, Form(ge=0, le=1, description="Override confidence threshold")]


async def _read_limited(file: UploadFile, limit: int) -> bytes:
    """Read at most limit+1 bytes so oversized uploads are rejected without buffering them all."""
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "File too large")
    return data


@router.post("/images", response_model=ImageRecord, status_code=status.HTTP_201_CREATED, tags=["images"])
async def upload_image(
    background: BackgroundTasks,
    file: UploadFile = File(..., description="JPEG, PNG, WEBP or BMP"),
    conf: Conf = None,
    settings: Settings = Depends(get_settings),
    storage: Storage = Depends(get_storage),
    detector: Detector = Depends(get_detector),
    azure: AzureSync = Depends(get_azure),
):
    """Upload one image: run YOLOv11, save input/processed/result to disk, return detections."""
    data = await _read_limited(file, settings.max_upload_bytes)
    record = await run_in_threadpool(
        process_upload, data, os.path.basename(file.filename or ""), conf, settings, storage, detector
    )
    if azure.enabled and settings.azure_auto_upload:
        background.add_task(azure.upload_record, storage, record)
    return record


@router.post("/images/batch", response_model=BatchResult, tags=["images"])
async def upload_batch(
    background: BackgroundTasks,
    files: list[UploadFile] = File(..., description="Up to 20 images"),
    conf: Conf = None,
    settings: Settings = Depends(get_settings),
    storage: Storage = Depends(get_storage),
    detector: Detector = Depends(get_detector),
    azure: AzureSync = Depends(get_azure),
):
    """Upload several images; failures are reported per file instead of failing the batch."""
    if len(files) > 20:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Max 20 files per batch")
    results, errors = [], []
    for f in files:
        name = os.path.basename(f.filename or "")
        try:
            data = await _read_limited(f, settings.max_upload_bytes)
            record = await run_in_threadpool(
                process_upload, data, name, conf, settings, storage, detector
            )
            results.append(record)
            if azure.enabled and settings.azure_auto_upload:
                background.add_task(azure.upload_record, storage, record)
        except HTTPException as exc:
            errors.append(BatchItemError(filename=name, error=str(exc.detail)))
    return BatchResult(results=results, errors=errors)


@router.get("/images", response_model=ImageList, tags=["images"])
def list_images(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    storage: Storage = Depends(get_storage),
):
    total, items = storage.list_records(limit, offset)
    return ImageList(total=total, limit=limit, offset=offset, items=items)


def _get_or_404(storage: Storage, image_id: str) -> ImageRecord:
    record = storage.get_record(image_id)
    if not record:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Image not found")
    return record


@router.get("/images/{image_id}", response_model=ImageRecord, tags=["images"])
def get_image(image_id: ImageId, storage: Storage = Depends(get_storage)):
    return _get_or_404(storage, image_id)


@router.get("/images/{image_id}/original", response_class=FileResponse, tags=["images"])
def download_original(image_id: ImageId, storage: Storage = Depends(get_storage)):
    record = _get_or_404(storage, image_id)
    return FileResponse(storage.abs_path(record.input_file))


@router.get("/images/{image_id}/processed", response_class=FileResponse, tags=["images"])
def download_processed(image_id: ImageId, storage: Storage = Depends(get_storage)):
    record = _get_or_404(storage, image_id)
    return FileResponse(storage.abs_path(record.processed_file), media_type="image/jpeg")


@router.delete("/images/{image_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["images"])
def delete_image(image_id: ImageId, storage: Storage = Depends(get_storage)):
    if not storage.delete_record(image_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Image not found")


@router.post("/sync/azure", response_model=SyncResult, tags=["azure"])
def sync_azure(storage: Storage = Depends(get_storage), azure: AzureSync = Depends(get_azure)):
    """Upload every record not yet backed up to Azure Blob Storage."""
    if not azure.enabled:
        return SyncResult(uploaded=0, failed=0, azure_enabled=False)
    uploaded, failed = azure.sync_all(storage)
    return SyncResult(uploaded=uploaded, failed=failed, azure_enabled=True)


@router.get("/dataset/export", response_class=FileResponse, tags=["cvat"])
def export_dataset(
    background: BackgroundTasks,
    ids: list[str] | None = Query(None, description="Limit to these image ids"),
    storage: Storage = Depends(get_storage),
    detector: Detector = Depends(get_detector),
):
    """Download images + detections as a CVAT 'YOLO 1.1' zip (see docs/CVAT.md)."""
    records = storage.all_records(ids)
    if not records:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No images to export")
    zip_path = build_yolo_zip(storage, records, detector.names)
    background.add_task(os.unlink, zip_path)
    return FileResponse(zip_path, media_type="application/zip", filename="dataset_yolo.zip")
