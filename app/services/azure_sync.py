"""Optional Azure Blob Storage backup. Disabled unless AZURE_ENABLED=true."""

import logging

from app.config import Settings
from app.schemas import ImageRecord
from app.services.storage import Storage

log = logging.getLogger(__name__)


class AzureSync:
    def __init__(self, settings: Settings) -> None:
        self.enabled = settings.azure_enabled and bool(settings.azure_storage_connection_string)
        self._settings = settings
        self._container = None

    def _get_container(self):
        if self._container is None:
            from azure.storage.blob import BlobServiceClient  # optional dependency

            service = BlobServiceClient.from_connection_string(
                self._settings.azure_storage_connection_string
            )
            container = service.get_container_client(self._settings.azure_container)
            if not container.exists():
                container.create_container()
            self._container = container
        return self._container

    def upload_record(self, storage: Storage, record: ImageRecord) -> bool:
        """Upload input, processed image and result JSON. Returns True on success."""
        if not self.enabled:
            return False
        try:
            container = self._get_container()
            for rel in (record.input_file, record.processed_file):
                with storage.abs_path(rel).open("rb") as fh:
                    container.upload_blob(rel, fh, overwrite=True)
            storage.mark_synced(record.id)  # rewrites the JSON with azure_synced=true first
            updated = storage.get_record(record.id)
            container.upload_blob(
                record.result_file, updated.model_dump_json(indent=2).encode(), overwrite=True
            )
            return True
        except Exception:  # network/credential problems must never break the API
            log.exception("Azure upload failed for %s", record.id)
            return False

    def sync_all(self, storage: Storage) -> tuple[int, int]:
        uploaded = failed = 0
        for record in storage.unsynced_records():
            if self.upload_record(storage, record):
                uploaded += 1
            else:
                failed += 1
        return uploaded, failed
