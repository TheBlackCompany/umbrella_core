# Umbrella Core

Central processing engine for the Umbrella Project, containing various data processing tools and services.

## Overview

Umbrella Core provides a suite of processing tools designed for government, legislative, and organizational data processing needs. Each tool is containerized and can be deployed independently or as part of a larger processing pipeline.

## Tools

### 🎙️ [Audio Transcriber](./audio_transcriber/)
High-accuracy audio transcription service with speaker diarization, optimized for legislative and government audio processing.

- **Features**: Whisper large-v3, speaker diarization, priority queue, forensic chain of custody
- **Performance**: 3-4x realtime on GPU
- **API**: REST API with job management
- **Status**: Production ready ✅

### 📄 Document Processor (Coming Soon)
OCR and document analysis for government documents, reports, and legislative texts.

### 📊 Data Aggregator (Coming Soon)
Collect and normalize data from various government sources and APIs.

### 🔍 Entity Extractor (Coming Soon)
NLP-based extraction of people, organizations, bills, and other entities from text.

## Architecture

```
umbrella_core/
├── audio_transcriber/      # Audio transcription service
│   ├── api_doctrine.py     # REST API
│   ├── engine.py          # Core processing
│   └── Dockerfile         # Container definition
├── document_processor/     # Document processing (future)
├── data_aggregator/       # Data collection (future)
├── entity_extractor/      # NLP extraction (future)
└── shared/               # Shared utilities
    ├── schemas/          # Common data schemas
    └── utils/           # Shared functions
```

## Quick Start

### Using Docker Compose

Deploy all services:
```bash
docker-compose up -d
```

Deploy specific service:
```bash
docker-compose up -d audio_transcriber
```

### Development

1. Clone repository:
```bash
git clone <repo-url>
cd umbrella_core
```

2. Choose a tool:
```bash
cd audio_transcriber
```

3. Follow tool-specific README for setup

## API Gateway

All services are accessible through the API gateway:

- Audio Transcriber: `http://gateway/audio/*`
- Document Processor: `http://gateway/documents/*`
- Data Aggregator: `http://gateway/data/*`

## Configuration

Global configuration in `.env`:
```env
# API Gateway
GATEWAY_PORT=8080

# Service Ports
AUDIO_TRANSCRIBER_PORT=8001
DOCUMENT_PROCESSOR_PORT=8002
DATA_AGGREGATOR_PORT=8003

# Shared Resources
REDIS_URL=redis://redis:6379
S3_BUCKET=umbrella-processing
```

## Deployment

### Kubernetes

Deploy to Kubernetes cluster:
```bash
kubectl apply -f k8s/
```

### Docker Swarm

Deploy as Docker Swarm stack:
```bash
docker stack deploy -c docker-compose.yml umbrella
```

## Contributing

Each tool should:
1. Have its own directory
2. Include a Dockerfile
3. Expose a REST API
4. Follow the common schema format
5. Include comprehensive tests

## License

[Your License]

## Support

For issues and feature requests, use the GitHub issue tracker.