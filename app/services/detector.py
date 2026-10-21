"""YOLOv11 inference (Ultralytics + PyTorch, CUDA when available)."""

import threading
import time
from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from app.config import Settings
from app.schemas import Detection


@dataclass
class DetectionOutput:
    detections: list[Detection]
    annotated: Image.Image
    inference_ms: float


class Detector(Protocol):
    model_name: str
    device: str
    loaded: bool
    names: dict[int, str]

    def load(self) -> None: ...

    def predict(self, image: Image.Image, conf: float | None = None) -> DetectionOutput: ...


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


class YoloDetector:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.model_name = settings.yolo_model
        self.device = "unloaded"
        self.loaded = False
        self.names: dict[int, str] = {}
        self._model = None
        self._lock = threading.Lock()  # one GPU inference at a time; load once

    def _resolve_device(self) -> str:
        wanted = self._settings.yolo_device.strip().lower()
        if wanted in ("", "auto"):
            return "cuda:0" if cuda_available() else "cpu"
        return wanted

    def load(self) -> None:
        with self._lock:
            if self.loaded:
                return
            from ultralytics import YOLO  # heavy import, keep lazy

            self.device = self._resolve_device()
            self._model = YOLO(self._settings.yolo_model)
            self._model.to(self.device)
            self.names = dict(self._model.names)
            self.loaded = True

    def predict(self, image: Image.Image, conf: float | None = None) -> DetectionOutput:
        if not self.loaded:
            self.load()
        threshold = self._settings.yolo_conf if conf is None else conf
        use_half = self._settings.yolo_half and self.device.startswith("cuda")
        with self._lock:
            start = time.perf_counter()
            results = self._model.predict(
                image,
                conf=threshold,
                iou=self._settings.yolo_iou,
                imgsz=self._settings.yolo_imgsz,
                device=self.device,
                half=use_half,
                verbose=False,
            )
            elapsed = (time.perf_counter() - start) * 1000
            result = results[0]
            plotted = result.plot()  # BGR ndarray with boxes drawn
        annotated = Image.fromarray(plotted[..., ::-1])
        detections = [
            Detection(
                class_id=int(box.cls.item()),
                class_name=self.names.get(int(box.cls.item()), str(int(box.cls.item()))),
                confidence=round(float(box.conf.item()), 4),
                bbox_xyxy=[round(float(v), 1) for v in box.xyxy[0].tolist()],
            )
            for box in result.boxes
        ]
        return DetectionOutput(detections, annotated, round(elapsed, 2))
