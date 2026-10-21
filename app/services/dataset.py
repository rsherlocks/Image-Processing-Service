"""Export stored images + detections as a CVAT-importable "YOLO 1.1" dataset zip.

Import in CVAT: create a task with the same images, then Actions > Upload annotations >
format "YOLO 1.1" and pick the zip. Or create a project from the zip. Detections become
editable pre-annotations you can correct and later use for retraining.
"""

import tempfile
import zipfile

from app.schemas import ImageRecord
from app.services.storage import Storage


def build_yolo_zip(storage: Storage, records: list[ImageRecord], names: dict[int, str]) -> str:
    """Write the zip to a temp file and return its path (caller deletes it)."""
    max_id = max([*names.keys(), *(d.class_id for r in records for d in r.detections)], default=-1)
    ordered = [names.get(i, f"class_{i}") for i in range(max_id + 1)]

    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    tmp.close()
    with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("obj.names", "\n".join(ordered) + "\n")
        zf.writestr(
            "obj.data",
            f"classes = {len(ordered)}\ntrain = train.txt\nnames = obj.names\nbackup = backup/\n",
        )
        train_lines = []
        for r in records:
            ext = r.input_file.rsplit(".", 1)[-1]
            stem = f"obj_train_data/{r.id}"
            train_lines.append(f"data/{stem}.{ext}")
            zf.write(storage.abs_path(r.input_file), f"{stem}.{ext}")
            lines = []
            for d in r.detections:
                x1, y1, x2, y2 = d.bbox_xyxy
                cx, cy = (x1 + x2) / 2 / r.width, (y1 + y2) / 2 / r.height
                w, h = (x2 - x1) / r.width, (y2 - y1) / r.height
                lines.append(f"{d.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            zf.writestr(f"{stem}.txt", "\n".join(lines) + ("\n" if lines else ""))
        zf.writestr("train.txt", "\n".join(train_lines) + "\n")
    return tmp.name
