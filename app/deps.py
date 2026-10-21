"""Shared singletons exposed as FastAPI dependencies (override them in tests)."""

from functools import lru_cache

from fastapi import Depends

from app.config import Settings, get_settings
from app.services.azure_sync import AzureSync
from app.services.detector import Detector, YoloDetector
from app.services.storage import Storage


@lru_cache
def _storage(root: str) -> Storage:
    return Storage(root)  # type: ignore[arg-type]


def get_storage(settings: Settings = Depends(get_settings)) -> Storage:
    return _storage(str(settings.storage_dir))


@lru_cache
def _detector() -> Detector:
    return YoloDetector(get_settings())


def get_detector() -> Detector:
    return _detector()


@lru_cache
def _azure() -> AzureSync:
    return AzureSync(get_settings())


def get_azure() -> AzureSync:
    return _azure()
