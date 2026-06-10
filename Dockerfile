# ── Stage 1: dependency install ──────────────────────────────────
FROM python:3.11-slim AS base

# System libs required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1-mesa-glx \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        libgstreamer1.0-0 \
        wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLOv8n weights so the container starts instantly
RUN python - <<'EOF'
from ultralytics import YOLO
YOLO("yolov8n.pt")          # downloads ~6 MB to /root/.config/Ultralytics/
EOF

# ── Stage 2: app ─────────────────────────────────────────────────
FROM base AS app

WORKDIR /app

COPY . .

# Persistent directories
RUN mkdir -p uploads static

EXPOSE 5000

# Disable Python output buffering so logs appear immediately
ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]
