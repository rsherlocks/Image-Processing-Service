import io
import os
import tempfile

# Configure the app BEFORE importing it (settings are cached on first use).
_TMP = tempfile.mkdtemp(prefix="aiimg-test-")
os.environ.update(
    API_KEYS="test-key,other-key",
    STORAGE_DIR=_TMP,
    PRELOAD_MODEL="false",
    RATE_LIMIT_PER_MINUTE="0",
    MAX_UPLOAD_MB="1",
    AZURE_ENABLED="false",
    CORS_ORIGINS="",
)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from app.deps import get_detector  # noqa: E402
from app.main import app  # noqa: E402
from app.schemas import Detection  # noqa: E402
from app.services.detector import DetectionOutput  # noqa: E402


class FakeDetector:
    """Stands in for YOLO so tests need neither torch nor model weights."""

    model_name = "fake.pt"
    device = "cpu"
    loaded = True
    names = {0: "person", 1: "car"}

    def load(self) -> None:
        pass

    def predict(self, image, conf=None):
        w, h = image.size
        det = Detection(class_id=0, class_name="person", confidence=0.9, bbox_xyxy=[0, 0, w / 2, h / 2])
        return DetectionOutput([det], image.copy(), 1.5)


@pytest.fixture(scope="session")
def client():
    app.dependency_overrides[get_detector] = lambda: FakeDetector()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def auth():
    return {"X-API-Key": "test-key"}


def make_image(fmt="PNG", size=(64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 30, 30)).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def png_bytes():
    return make_image("PNG")


@pytest.fixture
def uploaded(client, auth, png_bytes):
    r = client.post("/api/v1/images", headers=auth, files={"file": ("cat.png", png_bytes, "image/png")})
    assert r.status_code == 201, r.text
    return r.json()
