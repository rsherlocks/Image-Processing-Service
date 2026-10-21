"""Upload validation and the process-and-store pipeline."""

import io
import uuid

from fastapi import HTTPException, status
from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import Settings
from app.schemas import ImageRecord
from app.services.detector import Detector
from app.services.storage import Storage, utcnow

# Real format detected by Pillow (not the client-supplied content type) -> file extension
ALLOWED_FORMATS = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp", "BMP": "bmp"}


def validate_image(data: bytes, settings: Settings) -> tuple[Image.Image, str]:
    """Return (RGB image, extension) or raise HTTP 4xx. Never trusts filename/content-type."""
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty file")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"File exceeds {settings.max_upload_mb} MB"
        )
    Image.MAX_IMAGE_PIXELS = settings.max_image_pixels  # decompression-bomb guard
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            probe.verify()
        if fmt not in ALLOWED_FORMATS:
            raise HTTPException(
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                f"Unsupported format {fmt}. Allowed: {', '.join(ALLOWED_FORMATS)}",
            )
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Invalid image: {exc.__class__.__name__}")
    return image, ALLOWED_FORMATS[fmt]


def process_upload(
    data: bytes,
    filename: str,
    conf: float | None,
    settings: Settings,
    storage: Storage,
    detector: Detector,
) -> ImageRecord:
    image, ext = validate_image(data, settings)
    output = detector.predict(image, conf)

    image_id = uuid.uuid4().hex
    created = utcnow()
    paths = storage.relative_paths(image_id, ext, created)

    buf = io.BytesIO()
    output.annotated.save(buf, format="JPEG", quality=92)

    record = ImageRecord(
        id=image_id,
        original_filename=(filename or "upload")[:255],
        created_at=created,
        width=image.width,
        height=image.height,
        model=detector.model_name,
        device=detector.device,
        inference_ms=output.inference_ms,
        conf_threshold=settings.yolo_conf if conf is None else conf,
        detections=output.detections,
        **paths,
    )
    storage.write_bytes(record.input_file, data)  # keep the original bytes untouched
    storage.write_bytes(record.processed_file, buf.getvalue())
    storage.save_record(record)
    return record
