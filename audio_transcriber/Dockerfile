# Umbrella Transcriber - Doctrine v1.0 Compliant
FROM nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04

# Doctrine-required environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV WHISPER_MODEL=large-v3
ENV PYANNOTE_TOKEN=${PYANNOTE_TOKEN}
ENV MAX_CONCURRENT_JOBS=4
ENV GPU_MEMORY_FRACTION=0.9
ENV S3_BUCKET=umbrella-audio-processing

# System dependencies per doctrine
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download models at build time per doctrine
RUN python -c "import whisper; whisper.load_model('large-v3')"

# Copy application
COPY *.py ./
COPY strategies/ ./strategies/

# Create required directories
RUN mkdir -p /data/input /data/output /app/logs

# Set permissions for data directories
RUN chown -R 1000:1000 /data /app/logs

# API server per doctrine
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run as non-root
RUN useradd -m -u 1000 umbrella && chown -R umbrella:umbrella /app
USER umbrella

# Start API server
CMD ["uvicorn", "api_doctrine:app", "--host", "0.0.0.0", "--port", "8000"]