import base64
import io
from collections import Counter
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.images import Conf, _read_limited
from app.config import Settings, get_settings
from app.deps import get_detector
from app.schemas import DetectResponse
from app.security import rate_limit, require_api_key
from app.services.detector import Detector
from app.services.processor import validate_image

router = APIRouter(
    prefix="/api/v1", dependencies=[Depends(require_api_key), Depends(rate_limit)], tags=["detect"]
)


@router.post("/detect", response_model=DetectResponse)
async def detect_objects(
    file: UploadFile = File(..., description="JPEG, PNG, WEBP or BMP"),
    conf: Conf = None,
    classes: Annotated[
        str | None, Form(description="Comma-separated class names to keep, e.g. 'person,car'")
    ] = None,
    include_image: Annotated[bool, Form(description="Also return the annotated JPEG as base64")] = False,
    settings: Settings = Depends(get_settings),
    detector: Detector = Depends(get_detector),
):
    """Send an image, get the objects found in it. Nothing is saved to disk.

    Use `POST /api/v1/images` instead if you want the image and result stored.
    """
    data = await _read_limited(file, settings.max_upload_bytes)
    image, _ = validate_image(data, settings)
    output = await run_in_threadpool(detector.predict, image, conf)

    detections = output.detections
    if classes:
        wanted = {c.strip().lower() for c in classes.split(",") if c.strip()}
        detections = [d for d in detections if d.class_name.lower() in wanted]

    encoded = None
    if include_image:
        buf = io.BytesIO()
        output.annotated.save(buf, format="JPEG", quality=90)
        encoded = base64.b64encode(buf.getvalue()).decode()

    return DetectResponse(
        width=image.width,
        height=image.height,
        model=detector.model_name,
        device=detector.device,
        inference_ms=output.inference_ms,
        count=len(detections),
        summary=dict(Counter(d.class_name for d in detections)),
        detections=detections,
        annotated_image_base64=encoded,
    )
