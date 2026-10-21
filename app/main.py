import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app import __version__
from app.api.images import router as images_router
from app.config import Settings, get_settings
from app.deps import get_azure, get_detector, get_storage
from app.schemas import Health
from app.services.detector import cuda_available

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    get_storage(settings)  # creates folders + index
    if not settings.api_key_list:
        log.warning("API_KEYS is empty: every authenticated request will be rejected")
    if settings.preload_model:
        detector = app.dependency_overrides.get(get_detector, get_detector)()
        detector.load()
        log.info("Model %s loaded on %s", detector.model_name, detector.device)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Image Processing Service",
        version=__version__,
        description=(
            "Upload images, run YOLOv11 object detection (PyTorch, CUDA), store results locally, "
            "optionally back up to Azure Blob Storage and export datasets for CVAT.\n\n"
            "Click **Authorize** and enter an API key from `API_KEYS` in your `.env`."
        ),
        lifespan=lifespan,
    )
    settings = get_settings()
    if settings.cors_origin_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origin_list,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["X-API-Key", "Content-Type"],
        )
    app.include_router(images_router)

    index_html = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index():
        return index_html

    @app.get("/health", response_model=Health, tags=["system"])
    def health(
        settings: Settings = Depends(get_settings),
        detector=Depends(get_detector),
        azure=Depends(get_azure),
    ):
        return Health(
            status="ok",
            version=__version__,
            model=detector.model_name,
            model_loaded=detector.loaded,
            device=detector.device,
            cuda_available=cuda_available(),
            azure_enabled=azure.enabled,
        )

    return app


app = create_app()
