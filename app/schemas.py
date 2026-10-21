from datetime import datetime

from pydantic import BaseModel, Field


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(ge=0, le=1)
    bbox_xyxy: list[float] = Field(description="[x1, y1, x2, y2] in pixels")


class ImageRecord(BaseModel):
    id: str
    original_filename: str
    created_at: datetime
    width: int
    height: int
    model: str
    device: str
    inference_ms: float
    conf_threshold: float
    detections: list[Detection]
    input_file: str = Field(description="Path relative to the storage root")
    processed_file: str
    result_file: str
    azure_synced: bool = False

    @property
    def detection_count(self) -> int:
        return len(self.detections)


class ImageList(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[ImageRecord]


class BatchItemError(BaseModel):
    filename: str
    error: str


class BatchResult(BaseModel):
    results: list[ImageRecord]
    errors: list[BatchItemError]


class SyncResult(BaseModel):
    uploaded: int
    failed: int
    azure_enabled: bool


class Health(BaseModel):
    status: str
    version: str
    model: str
    model_loaded: bool
    device: str
    cuda_available: bool
    azure_enabled: bool
