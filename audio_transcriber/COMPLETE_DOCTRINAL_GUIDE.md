# Umbrella Audio Transcriber - Complete Doctrinal Guide

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [API Reference](#api-reference)
5. [Processing Workflow](#processing-workflow)
6. [Docker Deployment](#docker-deployment)
7. [Configuration Management](#configuration-management)
8. [Security & Compliance](#security-compliance)
9. [Performance Optimization](#performance-optimization)
10. [Integration Patterns](#integration-patterns)
11. [Troubleshooting](#troubleshooting)
12. [Appendices](#appendices)

---

## 1. Overview {#overview}

The Umbrella Audio Transcriber is a high-performance, GPU-accelerated audio transcription service designed for government, legislative, and enterprise use cases. It provides accurate speech-to-text conversion with speaker diarization, entity extraction, and forensic chain of custody.

### Key Capabilities
- **Multi-format audio processing** (WAV, MP3, M4A, FLAC, OGG, WebM)
- **Speaker diarization** with consolidation algorithms
- **Legislative optimization** for government proceedings
- **Priority-based job scheduling** (Emergency to Citizen levels)
- **Forensic chain of custody** for legal compliance
- **RESTful API** with async job management
- **MCP compatibility** for AI assistant integration

### Design Principles
1. **Accuracy First**: Uses OpenAI Whisper large-v3 model for maximum accuracy
2. **Scalability**: Async job queue with priority scheduling
3. **Security**: Multi-level classification support, audit trails
4. **Flexibility**: Configurable processing strategies
5. **Compliance**: Doctrine-based implementation with forensic tracking

---

## 2. System Architecture {#system-architecture}

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     External Clients                         │
│  (Web Apps, CLI, MCP Agents, n8n, Citizen Portals)         │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    API Gateway (FastAPI)                     │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│  │ Auth    │  │ Rate     │  │ Request  │  │ Response   │  │
│  │ Layer   │  │ Limiter  │  │ Valid.   │  │ Transform  │  │
│  └─────────┘  └──────────┘  └──────────┘  └────────────┘  │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    Core Processing Layer                     │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Priority   │  │ Transcription │  │    Speaker       │  │
│  │   Queue     │  │    Engine     │  │  Diarization     │  │
│  └─────────────┘  └──────────────┘  └──────────────────┘  │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Strategy   │  │   Context     │  │    Entity        │  │
│  │  Selector   │  │   System      │  │   Extraction     │  │
│  └─────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                     Storage Layer                            │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Redis     │  │  PostgreSQL  │  │   S3/Minio       │  │
│  │  (Queue)    │  │  (Metadata)  │  │   (Audio/JSON)   │  │
│  └─────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Component Interactions

```
Client Request → API Gateway → Job Queue → Processing Engine
                     ↓              ↓              ↓
                 Validation    Priority Sort    Transcribe
                     ↓              ↓              ↓
                 Response      Schedule      Diarization
                                               ↓
                                         Post-Processing
                                               ↓
                                           Storage
                                               ↓
                                        Webhook/Response
```

---

## 3. Core Components {#core-components}

### 3.1 Transcription Engine (`transcribe.py`)

The heart of the system, responsible for audio-to-text conversion.

```python
class UmbrellaTranscriber:
    """
    Main transcription orchestrator
    
    Responsibilities:
    - Audio file validation and loading
    - Model initialization (Whisper + Pyannote)
    - Strategy selection based on duration
    - Result formatting per doctrine schema
    """
    
    def __init__(self, 
                 model_size: str = "large",
                 device: str = "cuda",
                 compute_type: str = "float16"):
        # Initialize Whisper model
        self.model = whisper.load_model(model_size, device=device)
        
        # Initialize speaker diarization
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=os.getenv("PYANNOTE_TOKEN")
        )
```

**Key Methods:**
- `transcribe()`: Main entry point for transcription
- `_detect_audio_duration()`: Determines processing strategy
- `_select_strategy()`: Chooses between standard/chunked processing
- `format_segments()`: Combines transcription with speaker labels

### 3.2 Speaker Diarization (`speaker_consolidation.py`)

Post-processes speaker segments to reduce over-segmentation.

```python
class SpeakerConsolidator:
    """
    Reduces speaker over-segmentation using:
    - Voice similarity analysis
    - Temporal proximity
    - Context-based constraints
    """
    
    def consolidate_speakers(self, 
                           segments: List[Dict],
                           expected_speakers: Optional[int] = None,
                           context: Optional[ConversationContext] = None):
        # Apply consolidation algorithm
        if context and context.type == ConversationType.PHONE_CALL:
            expected_speakers = 2
```

**Consolidation Strategies:**
- **Phone calls**: Force 2 speakers
- **Interviews**: Typically 2 speakers (interviewer/interviewee)
- **Meetings**: Use similarity threshold
- **Legislative**: Maintain all speakers (formal proceedings)

### 3.3 Priority Queue System (`priority_queue.py`)

Manages job scheduling based on priority levels.

```python
class Priority(IntEnum):
    EMERGENCY = 0  # Live legislative sessions
    URGENT = 1     # Time-sensitive research
    NORMAL = 2     # Regular processing
    BATCH = 3      # Archive processing
    CITIZEN = 4    # Public submissions

class PriorityJobQueue:
    """
    Thread-safe priority queue with resource allocation
    
    Emergency jobs get:
    - 8x processing speed
    - Dedicated GPU resources
    - Preemption capability
    """
```

**Resource Allocation:**
| Priority | GPU Allocation | CPU Threads | Max Concurrent |
|----------|---------------|-------------|----------------|
| Emergency | 90% | 16 | 1 |
| Urgent | 70% | 12 | 2 |
| Normal | 50% | 8 | 4 |
| Batch | 30% | 4 | 8 |
| Citizen | 20% | 2 | 16 |

### 3.4 Processing Strategies (`strategies/`)

Different approaches based on audio characteristics.

#### Standard Strategy (`standard.py`)
- For files < 30 minutes
- Single-pass processing
- Full file loaded into memory

#### Chunked Strategy (`chunked.py`)
- For files > 30 minutes
- 20-minute chunks with 1-minute overlap
- Streaming processing to manage memory

```python
class ChunkedStrategy(ProcessingStrategy):
    def process(self, audio_path: Path, transcriber, **kwargs):
        chunks = self._split_audio(audio_path)
        results = []
        
        for i, chunk in enumerate(chunks):
            # Process with overlap handling
            result = transcriber.transcribe(chunk)
            results.append(self._adjust_timestamps(result, i))
            
        return self._merge_results(results)
```

### 3.5 Legislative Processor (`legislative_processor.py`)

Specialized processing for government audio.

```python
class LegislativeProcessor:
    """
    Extracts legislative entities:
    - Bill numbers (SB/HB patterns)
    - Legislator names
    - Committee references
    - Procedural markers
    """
    
    BILL_PATTERNS = [
        r'\b(SB|HB|SR|HR)\s*\d+\b',  # Senate/House Bills
        r'\bBill\s+\d+\b',           # Generic bills
    ]
    
    LEGISLATIVE_MARKERS = [
        "roll call", "motion to", "second the motion",
        "all in favor", "opposed", "motion carries"
    ]
```

### 3.6 Context System (`context_system.py`)

Provides processing hints based on audio type.

```python
class ConversationType(Enum):
    UNKNOWN = "unknown"
    PHONE_CALL = "phone_call"
    MEETING = "meeting"
    INTERVIEW = "interview"
    LEGISLATIVE = "legislative"
    PRESENTATION = "presentation"

class ConversationContext:
    """
    Influences:
    - Speaker count expectations
    - Vocabulary optimization
    - Entity extraction focus
    """
```

### 3.7 Security Manager (`security.py`)

Handles access control and classification.

```python
class SecurityManager:
    """
    Manages:
    - User authentication/authorization
    - Content classification
    - Audit logging
    - Encryption at rest
    """
    
    CLASSIFICATION_LEVELS = {
        "public": 0,
        "confidential": 1,
        "secret": 2,
        "top_secret": 3
    }
```

### 3.8 Schema System (`schema.py`)

Ensures doctrine-compliant output format.

```python
class DoctrineSchema:
    """
    Generates compliant output with:
    - Job identification
    - Source/transcript hashing
    - Processing metadata
    - Quality metrics
    - Chain of custody
    - Extracted entities
    """
    
    @staticmethod
    def create_output(transcription_result, metadata):
        return {
            "job_id": DoctrineSchema.create_job_id(file_hash, timestamp),
            "source_hash": file_hash,
            "transcript_hash": transcript_hash,
            "processing_metadata": {...},
            "transcript": {...},
            "quality_metrics": {...},
            "chain_of_custody": {...},
            "extracted_entities": {...}
        }
```

---

## 4. API Reference {#api-reference}

### 4.1 Core Endpoints

#### POST /jobs - Submit Transcription Job
```http
POST /jobs
Content-Type: application/json

{
  "audio_url": "s3://bucket/path/to/audio.wav",
  "source_metadata": {
    "source_type": "legislative_hearing",
    "expected_speakers": 15,
    "environment": "clean",
    "priority": "urgent",
    "project_code": "SB123-HEARING",
    "classification": "public",
    "speaker_hints": ["Senator Smith", "Representative Jones"]
  },
  "processing_options": {
    "diarization": true,
    "language": "en",
    "output_format": "full",
    "max_processing_minutes": 120
  },
  "callback_webhook": "https://callback.example.com/webhook"
}

Response:
{
  "job_id": "audio_20250207_143022_a1b2c3d4",
  "status": "accepted"
}
```

#### GET /jobs/{job_id}/status - Check Job Status
```http
GET /jobs/audio_20250207_143022_a1b2c3d4/status

Response:
{
  "job_id": "audio_20250207_143022_a1b2c3d4",
  "status": "processing",
  "progress": 0.45,
  "estimated_completion": "2025-02-07T14:35:00Z",
  "priority": "urgent",
  "submitted_at": "2025-02-07T14:30:22Z",
  "processing_started": "2025-02-07T14:30:25Z"
}
```

#### GET /jobs/{job_id}/result - Get Transcript
```http
GET /jobs/audio_20250207_143022_a1b2c3d4/result

Response:
{
  "job_id": "audio_20250207_143022_a1b2c3d4",
  "source_hash": "sha256:a1b2c3d4...",
  "transcript_hash": "sha256:e5f6g7h8...",
  "processing_metadata": {
    "start_time": "2025-02-07T14:30:25Z",
    "end_time": "2025-02-07T14:34:52Z",
    "duration_seconds": 267,
    "models_used": {
      "whisper": "large-v3",
      "pyannote": "3.1"
    },
    "processing_speed": 3.2,
    "strategies_used": ["chunked"]
  },
  "source_metadata": {
    "filename": "senate_hearing_sb123.wav",
    "duration_seconds": 854.3,
    "sample_rate": 16000,
    "channels": 1,
    "format": "wav"
  },
  "transcript": {
    "segments": [
      {
        "speaker": "SPEAKER_1",
        "text": "Good morning, this hearing will come to order.",
        "start": 0.0,
        "end": 3.5,
        "confidence": 0.95
      }
    ],
    "full_text": "Complete transcript text...",
    "speaker_count": 15,
    "speaker_map": {
      "SPEAKER_1": "speaker_abc123",
      "SPEAKER_2": "speaker_def456"
    }
  },
  "quality_metrics": {
    "audio_quality": "high",
    "silence_ratio": 0.12,
    "overlap_ratio": 0.05,
    "confidence_avg": 0.92,
    "confidence_min": 0.78
  },
  "extracted_entities": {
    "bills": ["SB 123", "HB 456"],
    "legislators": ["Senator Smith", "Representative Jones"],
    "committees": ["Finance Committee", "Education Subcommittee"],
    "organizations": ["State Treasury", "Department of Education"],
    "votes": [
      {
        "motion": "Motion to approve SB 123",
        "result": "passed",
        "timestamp": 523.4
      }
    ]
  },
  "chain_of_custody": {
    "submitted_by": "user123",
    "submitted_at": "2025-02-07T14:30:22Z",
    "processed_by": "worker-gpu-01",
    "integrity_checks": {
      "input_hash_verified": true,
      "output_hash_computed": true
    }
  }
}
```

#### DELETE /jobs/{job_id} - Cancel Job
```http
DELETE /jobs/audio_20250207_143022_a1b2c3d4

Response:
{
  "job_id": "audio_20250207_143022_a1b2c3d4",
  "status": "cancelled"
}
```

### 4.2 Webhook Management

#### POST /jobs/{job_id}/webhook - Register Webhook
```http
POST /jobs/audio_20250207_143022_a1b2c3d4/webhook
Content-Type: application/json

{
  "webhook_url": "https://callback.example.com/transcription-complete"
}

Response:
{
  "job_id": "audio_20250207_143022_a1b2c3d4",
  "webhook_registered": true
}
```

#### DELETE /jobs/{job_id}/webhook - Remove Webhook
```http
DELETE /jobs/audio_20250207_143022_a1b2c3d4/webhook

Response:
{
  "job_id": "audio_20250207_143022_a1b2c3d4",
  "webhook_removed": true
}
```

### 4.3 Utility Endpoints

#### GET /health - Service Health
```http
GET /health

Response:
{
  "status": "healthy",
  "version": "1.0.0",
  "models_loaded": true,
  "gpu_available": true,
  "gpu_memory_free": 8294,
  "queue_depth": {
    "emergency": 0,
    "urgent": 2,
    "normal": 5,
    "batch": 12,
    "citizen": 3
  }
}
```

#### GET /metrics - Processing Statistics
```http
GET /metrics

Response:
{
  "jobs_processed": 1547,
  "jobs_in_progress": 8,
  "jobs_failed": 23,
  "average_processing_time": 287.4,
  "average_processing_speed": 3.2,
  "queue_depth_by_priority": {
    "emergency": 0,
    "urgent": 2,
    "normal": 5,
    "batch": 12,
    "citizen": 3
  },
  "success_rate": 0.985,
  "uptime_seconds": 864000
}
```

#### POST /estimate - Cost/Time Estimation
```http
POST /estimate
Content-Type: application/json

{
  "duration_seconds": 3600,
  "priority": "normal",
  "diarization": true
}

Response:
{
  "estimated_cost_usd": 1.25,
  "estimated_time_seconds": 1200,
  "recommended_priority": "normal"
}
```

### 4.4 Error Responses

All errors follow RFC 7807 Problem Details format:

```json
{
  "type": "/errors/job-not-found",
  "title": "Job Not Found",
  "status": 404,
  "detail": "No job found with ID audio_20250207_143022_a1b2c3d4",
  "instance": "/jobs/audio_20250207_143022_a1b2c3d4/status"
}
```

Common error codes:
- `400` - Invalid request format
- `401` - Authentication required
- `403` - Insufficient permissions
- `404` - Resource not found
- `409` - Conflict (job already processing)
- `413` - File too large
- `429` - Rate limit exceeded
- `500` - Internal server error
- `503` - Service unavailable

---

## 5. Processing Workflow {#processing-workflow}

### 5.1 Job Submission Flow

```
1. Client submits audio URL via POST /jobs
   ↓
2. API validates request format and permissions
   ↓
3. Job created with unique ID (timestamp + hash)
   ↓
4. Job enqueued based on priority
   ↓
5. Response returned with job_id
   ↓
6. Background processing begins
```

### 5.2 Processing Pipeline

```
1. Job dequeued by worker
   ↓
2. Audio downloaded/validated
   ↓
3. Duration detection
   ↓
4. Strategy selection (standard/chunked)
   ↓
5. Whisper transcription
   ↓
6. Pyannote diarization (if enabled)
   ↓
7. Speaker consolidation
   ↓
8. Entity extraction (if legislative)
   ↓
9. Quality metrics calculation
   ↓
10. Result formatting per schema
    ↓
11. Storage and webhook notification
```

### 5.3 Chunked Processing Detail

For files > 30 minutes:

```
Audio File (2 hours)
    ↓
Split into 20-minute chunks with 1-minute overlap
    ↓
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│ Chunk 1 │ │ Chunk 2 │ │ Chunk 3 │ │ Chunk 4 │
│ 0-20min │ │19-39min │ │38-58min │ │57-77min │
└─────────┘ └─────────┘ └─────────┘ └─────────┘
    ↓           ↓           ↓           ↓
Process each chunk independently
    ↓           ↓           ↓           ↓
Merge results with overlap resolution
    ↓
Final transcript with adjusted timestamps
```

### 5.4 Speaker Consolidation Algorithm

```python
def consolidate_speakers(segments, expected_speakers=None):
    # 1. Build speaker similarity matrix
    similarity_matrix = compute_voice_similarity(segments)
    
    # 2. Apply clustering based on similarity
    clusters = hierarchical_clustering(similarity_matrix)
    
    # 3. Apply context constraints
    if expected_speakers:
        clusters = merge_to_n_speakers(clusters, expected_speakers)
    
    # 4. Create speaker mapping
    speaker_map = create_consolidated_mapping(clusters)
    
    return apply_mapping(segments, speaker_map)
```

---

## 6. Docker Deployment {#docker-deployment}

### 6.1 Dockerfile Structure

```dockerfile
# Multi-stage build for efficiency
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS base

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
FROM base AS builder
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Final stage
FROM base
COPY --from=builder /root/.local /root/.local
COPY . /app
WORKDIR /app

# Pre-download models at build time
RUN python -c "import whisper; whisper.load_model('large')"

# Security: Run as non-root
RUN useradd -m -u 1000 umbrella
USER umbrella

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD curl -f http://localhost:8000/health || exit 1

# Start API server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 6.2 Docker Compose Configuration

```yaml
version: '3.8'

services:
  audio-transcriber:
    build: 
      context: ./audio_transcriber
      dockerfile: Dockerfile
    image: umbrella-audio-transcriber:latest
    container_name: umbrella-transcriber
    ports:
      - "8000:8000"
    environment:
      - PYANNOTE_TOKEN=${PYANNOTE_TOKEN}
      - WHISPER_MODEL=large
      - GPU_MEMORY_FRACTION=0.9
      - MAX_CONCURRENT_JOBS=4
      - REDIS_URL=redis://redis:6379/0
      - DATABASE_URL=postgresql://umbrella:password@postgres:5432/transcriber
    volumes:
      - ./input:/data/input:ro
      - ./output:/data/output
      - model-cache:/root/.cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    depends_on:
      - redis
      - postgres
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    container_name: umbrella-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    container_name: umbrella-postgres
    environment:
      - POSTGRES_USER=umbrella
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=transcriber
    volumes:
      - postgres-data:/var/lib/postgresql/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    container_name: umbrella-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - audio-transcriber
    restart: unless-stopped

volumes:
  model-cache:
  redis-data:
  postgres-data:
```

### 6.3 Production Deployment

#### Build and Deploy Commands
```bash
# Build image with GPU support
docker build -t umbrella-audio-transcriber:latest \
  --build-arg CUDA_VERSION=12.1 \
  --build-arg PYTHON_VERSION=3.10 \
  ./audio_transcriber

# Run with GPU access
docker run -d \
  --name umbrella-transcriber \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/input:/data/input \
  -v $(pwd)/output:/data/output \
  -e PYANNOTE_TOKEN=$PYANNOTE_TOKEN \
  umbrella-audio-transcriber:latest

# Or use docker-compose
docker-compose -f docker-compose.prod.yml up -d
```

#### Environment Variables
```bash
# Required
PYANNOTE_TOKEN=hf_xxxxxxxxxxxx  # From HuggingFace

# Optional
WHISPER_MODEL=large              # Model size
GPU_MEMORY_FRACTION=0.9          # GPU memory limit
MAX_CONCURRENT_JOBS=4            # Parallel jobs
REDIS_URL=redis://localhost:6379 # Job queue
DATABASE_URL=postgresql://...    # Metadata storage
API_SECRET_KEY=your-secret-key   # API authentication
LOG_LEVEL=INFO                   # Logging verbosity
```

### 6.4 Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: umbrella-transcriber
  namespace: umbrella-core
spec:
  replicas: 3
  selector:
    matchLabels:
      app: audio-transcriber
  template:
    metadata:
      labels:
        app: audio-transcriber
    spec:
      containers:
      - name: transcriber
        image: umbrella-audio-transcriber:latest
        ports:
        - containerPort: 8000
        env:
        - name: PYANNOTE_TOKEN
          valueFrom:
            secretKeyRef:
              name: transcriber-secrets
              key: pyannote-token
        resources:
          requests:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
          limits:
            memory: "16Gi"
            cpu: "8"
            nvidia.com/gpu: 1
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: transcriber-service
  namespace: umbrella-core
spec:
  selector:
    app: audio-transcriber
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

## 7. Configuration Management {#configuration-management}

### 7.1 Configuration Hierarchy

```
1. Default values (hardcoded)
   ↓
2. Configuration files (config.yaml)
   ↓
3. Environment variables
   ↓
4. Command-line arguments
   ↓
5. Runtime API calls
```

### 7.2 Configuration File Format

```yaml
# config.yaml
transcription:
  model: large
  device: cuda
  compute_type: float16
  language: en
  
diarization:
  enabled: true
  min_speakers: 1
  max_speakers: 20
  
processing:
  chunk_duration: 1200  # 20 minutes
  chunk_overlap: 60     # 1 minute
  max_duration: 14400   # 4 hours
  
api:
  host: 0.0.0.0
  port: 8000
  workers: 4
  cors_origins:
    - http://localhost:3000
    - https://app.example.com
    
security:
  require_auth: true
  api_key_header: X-API-Key
  encryption_at_rest: true
  
storage:
  type: s3  # or 'local', 'azure', 'gcs'
  bucket: umbrella-transcripts
  region: us-east-1
  
redis:
  host: localhost
  port: 6379
  db: 0
  
monitoring:
  metrics_enabled: true
  metrics_port: 9090
  log_level: INFO
  sentry_dsn: null
```

### 7.3 Dynamic Configuration

```python
class ConfigManager:
    """
    Manages runtime configuration with hot-reload support
    """
    
    def __init__(self):
        self.config = self._load_config()
        self._watch_for_changes()
    
    def get(self, key: str, default=None):
        """Get config value with dot notation"""
        # "api.port" -> config['api']['port']
        return self._traverse(self.config, key.split('.'), default)
    
    def set(self, key: str, value):
        """Update config value at runtime"""
        self._update(key, value)
        self._notify_listeners()
```

---

## 8. Security & Compliance {#security-compliance}

### 8.1 Authentication & Authorization

```python
# API Key Authentication
@app.middleware("http")
async def authenticate(request: Request, call_next):
    if request.url.path in PUBLIC_ENDPOINTS:
        return await call_next(request)
    
    api_key = request.headers.get("X-API-Key")
    if not api_key or not validate_api_key(api_key):
        return JSONResponse(
            status_code=401,
            content={"error": "Invalid or missing API key"}
        )
    
    # Add user context
    request.state.user = get_user_from_api_key(api_key)
    return await call_next(request)

# Role-based access
def require_role(role: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            if not has_role(request.state.user, role):
                raise HTTPException(403, "Insufficient permissions")
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator
```

### 8.2 Data Classification

```python
class ClassificationLevel(Enum):
    PUBLIC = "public"
    CONFIDENTIAL = "confidential"  
    SECRET = "secret"
    TOP_SECRET = "top_secret"

def enforce_classification(user_clearance: str, content_classification: str):
    """Ensure user has clearance for content"""
    clearance_levels = {
        "public": 0,
        "confidential": 1,
        "secret": 2,
        "top_secret": 3
    }
    
    if clearance_levels[user_clearance] < clearance_levels[content_classification]:
        raise SecurityException("Insufficient clearance level")
```

### 8.3 Audit Logging

```python
class AuditLogger:
    """
    Logs all security-relevant events
    """
    
    def log_access(self, user_id: str, resource: str, action: str, result: str):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "resource": resource,
            "action": action,
            "result": result,
            "ip_address": get_client_ip(),
            "user_agent": get_user_agent()
        }
        
        # Log to secure audit trail
        self._write_to_audit_log(entry)
```

### 8.4 Encryption

```python
# At-rest encryption for sensitive data
class EncryptionManager:
    def __init__(self, key: bytes):
        self.cipher = Fernet(key)
    
    def encrypt_file(self, file_path: Path) -> Path:
        """Encrypt audio/transcript files"""
        with open(file_path, 'rb') as f:
            encrypted = self.cipher.encrypt(f.read())
        
        encrypted_path = file_path.with_suffix('.enc')
        with open(encrypted_path, 'wb') as f:
            f.write(encrypted)
        
        return encrypted_path

# In-transit encryption handled by TLS/HTTPS
```

### 8.5 Compliance Features

#### GDPR Compliance
- Right to deletion (DELETE /jobs/{id})
- Data portability (export in JSON)
- Consent tracking
- Privacy by design

#### HIPAA Compliance
- Audit trails
- Encryption at rest and in transit
- Access controls
- Data retention policies

#### Chain of Custody
```python
def create_chain_of_custody(file_path: Path, user_id: str):
    return {
        "file_hash": calculate_sha256(file_path),
        "submitted_by": user_id,
        "submitted_at": datetime.utcnow().isoformat(),
        "processing_node": socket.gethostname(),
        "integrity_verified": verify_file_integrity(file_path),
        "tamper_evidence": None
    }
```

---

## 9. Performance Optimization {#performance-optimization}

### 9.1 GPU Optimization

```python
# Optimal batch sizes per GPU
GPU_BATCH_SIZES = {
    "RTX 3080": 8,
    "RTX 3090": 12,
    "RTX 4090": 16,
    "A100": 24,
    "H100": 32
}

# Memory management
def optimize_gpu_memory():
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.9)
    
    # Enable TF32 for Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
```

### 9.2 Caching Strategy

```python
class TranscriptionCache:
    """
    Multi-level caching for performance
    """
    
    def __init__(self):
        self.memory_cache = LRUCache(maxsize=100)
        self.redis_cache = Redis()
        self.disk_cache = DiskCache("/tmp/transcription_cache")
    
    async def get_or_compute(self, audio_hash: str, compute_func):
        # L1: Memory cache
        if result := self.memory_cache.get(audio_hash):
            return result
        
        # L2: Redis cache
        if result := await self.redis_cache.get(audio_hash):
            self.memory_cache.set(audio_hash, result)
            return result
        
        # L3: Disk cache
        if result := self.disk_cache.get(audio_hash):
            await self.redis_cache.set(audio_hash, result)
            self.memory_cache.set(audio_hash, result)
            return result
        
        # Compute and cache
        result = await compute_func()
        await self._cache_result(audio_hash, result)
        return result
```

### 9.3 Performance Metrics

```python
class PerformanceMonitor:
    """
    Track and optimize performance metrics
    """
    
    metrics = {
        "transcription_speed": [],  # Words per second
        "gpu_utilization": [],      # Percentage
        "memory_usage": [],         # GB
        "queue_wait_time": [],      # Seconds
        "processing_time": []       # Seconds
    }
    
    def analyze_performance(self):
        return {
            "avg_speed": np.mean(self.metrics["transcription_speed"]),
            "gpu_efficiency": np.mean(self.metrics["gpu_utilization"]),
            "memory_pressure": max(self.metrics["memory_usage"]) / TOTAL_GPU_MEMORY,
            "queue_efficiency": np.mean(self.metrics["queue_wait_time"]),
            "processing_efficiency": self._calculate_efficiency()
        }
```

### 9.4 Optimization Techniques

1. **Model Quantization**
   ```python
   # Use INT8 quantization for faster inference
   model = whisper.load_model("large", device="cuda")
   model = torch.quantization.quantize_dynamic(
       model, {torch.nn.Linear}, dtype=torch.qint8
   )
   ```

2. **Batch Processing**
   ```python
   # Process multiple audio segments in parallel
   def batch_transcribe(audio_segments: List[np.ndarray]):
       with torch.cuda.amp.autocast():
           return model.transcribe_batch(audio_segments)
   ```

3. **Streaming Processing**
   ```python
   # Process audio as it arrives
   async def stream_transcribe(audio_stream):
       buffer = AudioBuffer(size=30)  # 30-second buffer
       async for chunk in audio_stream:
           buffer.add(chunk)
           if buffer.is_full():
               yield await transcribe_buffer(buffer)
               buffer.clear()
   ```

---

## 10. Integration Patterns {#integration-patterns}

### 10.1 REST API Integration

```python
# Python client example
import httpx
import asyncio

class UmbrellaTranscriberClient:
    def __init__(self, base_url: str, api_key: str):
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={"X-API-Key": api_key}
        )
    
    async def transcribe(self, audio_url: str, **options):
        # Submit job
        response = await self.client.post("/jobs", json={
            "audio_url": audio_url,
            "source_metadata": options.get("metadata", {}),
            "processing_options": options.get("processing", {})
        })
        job_id = response.json()["job_id"]
        
        # Poll for completion
        while True:
            status = await self.get_status(job_id)
            if status["status"] == "completed":
                return await self.get_result(job_id)
            elif status["status"] == "failed":
                raise Exception(f"Transcription failed: {status}")
            await asyncio.sleep(5)
```

### 10.2 Webhook Integration

```python
# Webhook receiver example
from fastapi import FastAPI, Request

app = FastAPI()

@app.post("/webhook/transcription-complete")
async def handle_transcription_webhook(request: Request):
    payload = await request.json()
    
    # Process completed transcription
    job_id = payload["job_id"]
    status = payload["status"]
    
    if status == "completed":
        # Fetch and process result
        result = await fetch_transcription_result(job_id)
        await process_transcription(result)
    else:
        # Handle failure
        await handle_transcription_failure(job_id, payload.get("error"))
    
    return {"status": "received"}
```

### 10.3 MCP Integration

```python
# MCP server implementation
from mcp import MCPServer, Tool

class TranscriberMCPServer(MCPServer):
    def __init__(self):
        super().__init__("umbrella-transcriber")
        self.client = UmbrellaTranscriberClient(
            base_url="http://localhost:8000",
            api_key=os.getenv("UMBRELLA_API_KEY")
        )
    
    @Tool(
        name="transcribe_audio",
        description="Transcribe audio with speaker diarization",
        parameters={
            "audio_url": "URL or path to audio file",
            "source_type": "Type of audio content",
            "expected_speakers": "Number of expected speakers"
        }
    )
    async def transcribe_audio(self, audio_url: str, **kwargs):
        return await self.client.transcribe(audio_url, **kwargs)
```

### 10.4 n8n Workflow Integration

```json
{
  "nodes": [
    {
      "name": "HTTP Request",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "method": "POST",
        "url": "http://umbrella-transcriber:8000/jobs",
        "authentication": "headerAuth",
        "headerAuth": {
          "name": "X-API-Key",
          "value": "={{$credentials.umbrellaApiKey}}"
        },
        "jsonParameters": true,
        "bodyParametersJson": {
          "audio_url": "={{$node['S3'].json['url']}}",
          "source_metadata": {
            "source_type": "meeting",
            "priority": "normal"
          }
        }
      }
    },
    {
      "name": "Wait for Completion",
      "type": "n8n-nodes-base.wait",
      "parameters": {
        "amount": 30,
        "unit": "seconds"
      }
    },
    {
      "name": "Get Result",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "method": "GET",
        "url": "={{$node['HTTP Request'].json['job_id']}}/result"
      }
    }
  ]
}
```

### 10.5 CLI Integration

```bash
#!/bin/bash
# Shell script integration

# Submit transcription
JOB_RESPONSE=$(curl -s -X POST http://localhost:8000/jobs \
  -H "X-API-Key: $UMBRELLA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_url": "file:///path/to/audio.wav",
    "source_metadata": {
      "source_type": "meeting",
      "priority": "normal"
    }
  }')

JOB_ID=$(echo $JOB_RESPONSE | jq -r '.job_id')

# Poll for completion
while true; do
  STATUS=$(curl -s http://localhost:8000/jobs/$JOB_ID/status | jq -r '.status')
  
  if [ "$STATUS" = "completed" ]; then
    # Get result
    curl -s http://localhost:8000/jobs/$JOB_ID/result | jq '.'
    break
  elif [ "$STATUS" = "failed" ]; then
    echo "Transcription failed"
    exit 1
  fi
  
  sleep 5
done
```

---

## 11. Troubleshooting {#troubleshooting}

### 11.1 Common Issues

#### GPU Not Detected
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA availability in Python
python -c "import torch; print(torch.cuda.is_available())"

# Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi
```

#### Out of Memory Errors
```python
# Reduce batch size
os.environ["BATCH_SIZE"] = "4"

# Limit GPU memory
torch.cuda.set_per_process_memory_fraction(0.8)

# Use smaller model
model = whisper.load_model("medium")
```

#### Slow Processing
```python
# Check GPU utilization
nvidia-smi dmon -s u

# Profile performance
with torch.profiler.profile() as prof:
    result = transcriber.transcribe(audio)
print(prof.key_averages())

# Enable optimizations
torch.backends.cudnn.benchmark = True
```

### 11.2 Debugging Tools

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Trace processing pipeline
@trace_execution
def process_audio(audio_path):
    # Processing steps logged automatically
    pass

# Memory profiling
from memory_profiler import profile

@profile
def transcribe_audio(audio_path):
    # Memory usage tracked line by line
    pass
```

### 11.3 Health Checks

```python
class HealthChecker:
    """
    Comprehensive health monitoring
    """
    
    async def check_all(self):
        checks = {
            "api": await self._check_api(),
            "gpu": await self._check_gpu(),
            "models": await self._check_models(),
            "storage": await self._check_storage(),
            "queue": await self._check_queue()
        }
        
        return {
            "status": "healthy" if all(checks.values()) else "unhealthy",
            "checks": checks,
            "timestamp": datetime.utcnow().isoformat()
        }
```

### 11.4 Recovery Procedures

```python
# Automatic recovery from failures
class RecoveryManager:
    def __init__(self):
        self.recovery_strategies = {
            "gpu_oom": self._recover_from_oom,
            "model_corruption": self._reload_models,
            "queue_deadlock": self._reset_queue,
            "storage_failure": self._switch_storage_backend
        }
    
    async def recover_from_error(self, error_type: str):
        if strategy := self.recovery_strategies.get(error_type):
            await strategy()
        else:
            # Manual intervention required
            await self._alert_operators(error_type)
```

---

## 12. Appendices {#appendices}

### Appendix A: Complete API Schema

```yaml
openapi: 3.0.0
info:
  title: Umbrella Audio Transcription API
  version: 1.0.0
  description: Doctrine-compliant audio transcription service

paths:
  /jobs:
    post:
      summary: Submit transcription job
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/JobSubmissionRequest'
      responses:
        '200':
          description: Job accepted
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/JobSubmissionResponse'

components:
  schemas:
    JobSubmissionRequest:
      type: object
      required:
        - audio_url
        - source_metadata
      properties:
        audio_url:
          type: string
          format: uri
        source_metadata:
          $ref: '#/components/schemas/SourceMetadata'
        processing_options:
          $ref: '#/components/schemas/ProcessingOptions'
        callback_webhook:
          type: string
          format: uri
```

### Appendix B: Performance Benchmarks

| Audio Duration | Model | GPU | Processing Time | Speed | Memory |
|---------------|-------|-----|-----------------|-------|---------|
| 10 min | Large | RTX 3080 | 3 min | 3.3x | 6GB |
| 30 min | Large | RTX 3080 | 9 min | 3.3x | 7GB |
| 60 min | Large | RTX 3080 | 18 min | 3.3x | 8GB |
| 120 min | Large | RTX 3080 | 38 min | 3.2x | 8GB |
| 10 min | Large | A100 | 2 min | 5.0x | 12GB |
| 60 min | Large | A100 | 12 min | 5.0x | 14GB |

### Appendix C: Error Codes Reference

| Code | Type | Description | Resolution |
|------|------|-------------|------------|
| E001 | GPU_OOM | GPU out of memory | Reduce batch size or use smaller model |
| E002 | AUDIO_CORRUPT | Cannot read audio file | Verify file integrity |
| E003 | MODEL_LOAD | Failed to load model | Check model files and GPU drivers |
| E004 | DIARIZATION_FAIL | Speaker diarization failed | Verify Pyannote token |
| E005 | STORAGE_FULL | Output storage full | Clear old results or expand storage |
| E006 | QUEUE_TIMEOUT | Job timeout in queue | Increase timeout or priority |

### Appendix D: Configuration Examples

#### Minimal Configuration
```yaml
transcription:
  model: large
api:
  port: 8000
```

#### High-Performance Configuration
```yaml
transcription:
  model: large
  device: cuda
  compute_type: float16
  
processing:
  batch_size: 16
  num_workers: 8
  
cache:
  enabled: true
  size: 10GB
  
gpu:
  memory_fraction: 0.95
  allow_growth: false
```

#### Legislative-Optimized Configuration
```yaml
transcription:
  model: large
  language: en
  
legislative:
  enabled: true
  extract_bills: true
  identify_speakers: true
  track_votes: true
  
processing:
  priority_boost: emergency
  max_speakers: 50
```

### Appendix E: Monitoring Queries

#### Prometheus Queries
```promql
# Average processing speed
rate(transcription_words_total[5m]) / rate(transcription_duration_seconds[5m])

# GPU utilization
avg(gpu_utilization_percent) by (gpu_index)

# Queue depth by priority
transcription_queue_depth by (priority)

# Error rate
rate(transcription_errors_total[5m]) / rate(transcription_requests_total[5m])
```

#### Grafana Dashboard JSON
```json
{
  "dashboard": {
    "title": "Umbrella Transcriber Monitoring",
    "panels": [
      {
        "title": "Processing Speed",
        "targets": [
          {
            "expr": "rate(transcription_words_total[5m])"
          }
        ]
      }
    ]
  }
}
```

---

This completes the comprehensive doctrinal guide for the Umbrella Audio Transcriber module. The system is designed for high-accuracy, high-performance audio transcription with enterprise-grade features and compliance capabilities.