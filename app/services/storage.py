"""Local disk storage (SSD/HDD) plus a small SQLite index for fast listing.

Layout under STORAGE_DIR:
    inputs/YYYY/MM/DD/<id>.<ext>      original uploads
    processed/YYYY/MM/DD/<id>.jpg     annotated images
    results/YYYY/MM/DD/<id>.json      detection results (source of truth)
    index.sqlite3                     listing index
"""

import contextlib
import sqlite3
import threading
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from app.schemas import ImageRecord


class Storage:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        for sub in ("inputs", "processed", "results"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self._db_path = self.root / "index.sqlite3"
        self._lock = threading.Lock()
        with self._conn() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS images ("
                "id TEXT PRIMARY KEY, created_at TEXT NOT NULL, "
                "azure_synced INTEGER NOT NULL DEFAULT 0, data TEXT NOT NULL)"
            )
            db.execute("CREATE INDEX IF NOT EXISTS idx_created ON images(created_at DESC)")

    @contextlib.contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            db = sqlite3.connect(self._db_path)
            try:
                yield db
                db.commit()
            finally:
                db.close()

    # ---- paths -------------------------------------------------------------
    @staticmethod
    def relative_paths(image_id: str, ext: str, when: datetime) -> dict[str, str]:
        day = when.strftime("%Y/%m/%d")
        return {
            "input_file": f"inputs/{day}/{image_id}.{ext}",
            "processed_file": f"processed/{day}/{image_id}.jpg",
            "result_file": f"results/{day}/{image_id}.json",
        }

    def abs_path(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):  # defence in depth
            raise ValueError("Path escapes storage root")
        return path

    def write_bytes(self, relative: str, data: bytes) -> None:
        path = self.abs_path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)  # atomic: no half-written files

    # ---- records -----------------------------------------------------------
    def save_record(self, record: ImageRecord) -> None:
        payload = record.model_dump_json(indent=2)
        self.write_bytes(record.result_file, payload.encode())
        with self._conn() as db:
            db.execute(
                "INSERT OR REPLACE INTO images (id, created_at, azure_synced, data) VALUES (?, ?, ?, ?)",
                (record.id, record.created_at.isoformat(), int(record.azure_synced), payload),
            )

    def get_record(self, image_id: str) -> ImageRecord | None:
        with self._conn() as db:
            row = db.execute("SELECT data FROM images WHERE id = ?", (image_id,)).fetchone()
        return ImageRecord.model_validate_json(row[0]) if row else None

    def list_records(
        self, limit: int, offset: int, object_name: str | None = None, min_confidence: float = 0
    ) -> tuple[int, list[ImageRecord]]:
        if object_name:
            # Detections live in the JSON blob, so filter in Python (fine at desktop scale).
            wanted = object_name.strip().lower()
            with self._conn() as db:
                rows = db.execute("SELECT data FROM images ORDER BY created_at DESC").fetchall()
            matches = [
                r
                for r in (ImageRecord.model_validate_json(row[0]) for row in rows)
                if any(
                    d.class_name.lower() == wanted and d.confidence >= min_confidence
                    for d in r.detections
                )
            ]
            return len(matches), matches[offset : offset + limit]
        with self._conn() as db:
            total = db.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            rows = db.execute(
                "SELECT data FROM images ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset)
            ).fetchall()
        return total, [ImageRecord.model_validate_json(r[0]) for r in rows]

    def unsynced_records(self) -> list[ImageRecord]:
        with self._conn() as db:
            rows = db.execute("SELECT data FROM images WHERE azure_synced = 0").fetchall()
        return [ImageRecord.model_validate_json(r[0]) for r in rows]

    def all_records(self, ids: list[str] | None = None) -> list[ImageRecord]:
        with self._conn() as db:
            rows = db.execute("SELECT data FROM images ORDER BY created_at").fetchall()
        records = [ImageRecord.model_validate_json(r[0]) for r in rows]
        return [r for r in records if ids is None or r.id in ids]

    def mark_synced(self, image_id: str) -> None:
        record = self.get_record(image_id)
        if record:
            record.azure_synced = True
            self.save_record(record)

    def delete_record(self, image_id: str) -> bool:
        record = self.get_record(image_id)
        if not record:
            return False
        for rel in (record.input_file, record.processed_file, record.result_file):
            with contextlib.suppress(FileNotFoundError):
                self.abs_path(rel).unlink()
        with self._conn() as db:
            db.execute("DELETE FROM images WHERE id = ?", (image_id,))
        return True


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
