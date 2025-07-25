# Umbrella Transcriber Deployment Guide

## Prerequisites

1. **Docker Desktop** with WSL2 backend (Windows) or Docker Engine (Linux)
2. **NVIDIA Container Toolkit** for GPU support
3. **NVIDIA GPU** with CUDA 12.1+ support
4. **Hugging Face account** for speaker diarization

## Quick Start

### 1. Configure Environment

```bash
cp .env.template .env
# Edit .env with your configuration
```

**Required configuration:**
- `PYANNOTE_TOKEN` - Get from https://huggingface.co/settings/tokens

### 2. Build Docker Image

Windows:
```cmd
build.bat
```

Linux/Mac:
```bash
chmod +x build.sh
./build.sh
```

### 3. Start Services

```bash
docker-compose -f docker-compose.prod.yml up -d
```

### 4. Verify Deployment

Check health:
```bash
curl http://localhost:8000/health
```

View logs:
```bash
docker-compose -f docker-compose.prod.yml logs -f
```

## Usage

### API Endpoints

- **Submit job:** `POST http://localhost:8000/jobs`
- **Check status:** `GET http://localhost:8000/jobs/{job_id}/status`
- **Get result:** `GET http://localhost:8000/jobs/{job_id}/result`
- **API docs:** `http://localhost:8000/docs`

### File Processing

Place audio files in `./input/` directory:
```bash
cp your_audio.wav ./input/
```

Results appear in `./output/` directory.

### Example API Call

```bash
curl -X POST http://localhost:8000/jobs \
  -F "file=@audio.wav" \
  -F "priority=normal" \
  -F "expected_speakers=2"
```

## Production Considerations

### GPU Memory

Adjust `GPU_MEMORY_FRACTION` in `.env` if running multiple services:
- Single service: 0.9 (90%)
- Multiple services: 0.5-0.7

### Scaling

Increase concurrent jobs in `.env`:
```
MAX_CONCURRENT_JOBS=8
```

### SSL/HTTPS

1. Add certificates to `./ssl/` directory
2. Update `nginx.conf` with SSL configuration
3. Set `NGINX_SSL_PORT=443` in `.env`

### Monitoring

Prometheus metrics available at:
```
http://localhost:8000/metrics
```

### Backup

Regular backups recommended for:
- `./output/` - Transcription results
- Redis data volume - Job queue

## Troubleshooting

### GPU not detected
```bash
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi
```

### Out of memory
- Reduce `GPU_MEMORY_FRACTION`
- Decrease `MAX_CONCURRENT_JOBS`
- Use smaller Whisper model

### Slow processing
- Ensure GPU is being used (check logs)
- Verify CUDA version compatibility
- Check available GPU memory

## Security

1. Change default `API_SECRET_KEY` in production
2. Use HTTPS for external access
3. Implement rate limiting in nginx
4. Regular security updates:
   ```bash
   docker-compose -f docker-compose.prod.yml pull
   docker-compose -f docker-compose.prod.yml up -d
   ```