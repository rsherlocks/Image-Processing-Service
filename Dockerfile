FROM python:3.11-slim

# cu124 = CUDA 12.4 wheels (work with any recent NVIDIA driver). For CPU-only builds use:
#   --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt requirements-ml.txt requirements-azure.txt ./
RUN pip install torch torchvision --index-url ${TORCH_INDEX_URL} \
    && pip install -r requirements.txt -r requirements-ml.txt -r requirements-azure.txt

COPY app ./app

# Run as an unprivileged user
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /data /models && chown -R appuser /data /models /srv
USER appuser
WORKDIR /models
# Model weights are downloaded into /models (a volume) on first start and reused afterwards.

ENV STORAGE_DIR=/data PYTHONPATH=/srv
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
