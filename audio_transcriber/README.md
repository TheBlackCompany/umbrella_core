# Audio Transcriber

High-accuracy audio transcription service with speaker diarization, part of the Umbrella Core processing suite.

## Features

- **OpenAI Whisper** (large-v3 model) for transcription
- **Speaker diarization** using pyannote.audio
- **GPU acceleration** with CUDA support
- **Chunked processing** for long audio files
- **Priority queue** system (Emergency/Urgent/Normal/Batch/Citizen)
- **Legislative optimization** with bill and legislator extraction
- **Forensic chain of custody** for legal compliance
- **REST API** with job management

## Quick Start

### Standalone Docker

```bash
cd audio_transcriber
docker build -t umbrella-audio-transcriber .
docker run -p 8000:8000 --gpus all umbrella-audio-transcriber
```

### With Docker Compose (from root)

```bash
docker-compose up audio-transcriber
```

### Local Development

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
python cli.py audio.wav --output-dir output/
```

## API Usage

### Submit Job
```bash
curl -X POST http://localhost:8000/jobs \
  -F "file=@audio.wav" \
  -F "priority=normal" \
  -F "expected_speakers=2" \
  -F "context_type=meeting"
```

### Check Status
```bash
curl http://localhost:8000/jobs/{job_id}/status
```

### Get Result
```bash
curl http://localhost:8000/jobs/{job_id}/result
```

## Configuration

Environment variables:
- `PYANNOTE_TOKEN` - Required for speaker diarization
- `WHISPER_MODEL` - Model size (tiny/base/small/medium/large)
- `MAX_CONCURRENT_JOBS` - Parallel processing limit
- `GPU_MEMORY_FRACTION` - GPU memory allocation

## Output Format

Doctrine-compliant JSON with:
```json
{
  "job_id": "20250724_164141_ca6ba565",
  "transcript": {
    "segments": [...],
    "full_text": "...",
    "speaker_count": 2
  },
  "processing_metadata": {
    "models_used": {"whisper": "large"},
    "processing_speed": 3.2
  },
  "chain_of_custody": {...}
}
```

## Performance

- Speed: 3-4x realtime on RTX 3080
- Accuracy: 95%+ with large model
- Memory: ~8GB GPU RAM

## Supported Formats

WAV, MP3, M4A, FLAC, OGG, WebM

## See Also

- [Main README](../README.md)
- [API Documentation](http://localhost:8000/docs)
- [Deployment Guide](./deploy.md)