FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=5001 \
    FLASK_APP=app.py \
    FLASK_ENV=production

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt ./

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn

# Copy application files
COPY app.py ./
COPY inference_server.py ./
COPY pose_analyzer.py ./
COPY poseformer_model.py ./
COPY pose_3d_visualizer.py ./
COPY vitpose_detector.py ./
COPY vitpose_base_coco.py ./
COPY deficiency.py ./
COPY supabase_db.py ./
COPY config.py ./
COPY video_overlay.py ./
COPY bigru_model.py ./
COPY static/ ./static/

# Create necessary directories
RUN mkdir -p uploads outputs models

EXPOSE 5001

# Run the application (Render uses $PORT)
CMD ["sh", "-c", "gunicorn -w 2 -b 0.0.0.0:${PORT} app:app"]
