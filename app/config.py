from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Security
    api_keys: str = ""
    cors_origins: str = ""
    max_upload_mb: int = 20
    max_image_pixels: int = 50_000_000
    rate_limit_per_minute: int = 120

    # Storage
    storage_dir: Path = Path("./storage")

    # Model
    yolo_model: str = "yolo11n.pt"
    yolo_device: str = "auto"
    yolo_conf: float = 0.25
    yolo_iou: float = 0.45
    yolo_imgsz: int = 640
    yolo_half: bool = True
    preload_model: bool = True

    # Azure (optional)
    azure_enabled: bool = False
    azure_storage_connection_string: str = ""
    azure_container: str = "image-processing"
    azure_auto_upload: bool = True

    @property
    def api_key_list(self) -> list[str]:
        return _split(self.api_keys)

    @property
    def cors_origin_list(self) -> list[str]:
        return _split(self.cors_origins)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
