# Production image for the Brain Tumour Segmentation app.
# CPU-only by default. Build:  docker build -t brats-seg .
# Run:    docker run --rm -p 7860:7860 -v /path/to/artifacts:/app/artifacts brats-seg
# GPU:    use --gpus all with an onnxruntime-gpu enabled base (see README).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=7860 \
    BRATSIFY=1

WORKDIR /app

# System deps: libgomp (onnxruntime/torch), unzip not needed at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code only — model weights, sample data and notebooks stay OUT of the
# image. Mount them at runtime: -v /host/artifacts:/app/artifacts
COPY app.py .
COPY src/ ./src/
COPY README.md LICENSE ./

# Non-root user for hospital deployment hygiene.
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

# Gradio reads APP_HOST/APP_PORT via app.py env handling.
CMD ["python", "app.py"]
