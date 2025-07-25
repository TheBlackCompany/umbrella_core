# UmbrellaOps Audio Transcription Service Doctrine v1.0

**Status**: Draft for Operator review (2025-01-22)  
**Maintainers**: Transcription-GPT · Pipeline-GPT · Sean Steele  
**Purpose**: Multi-tenant audio processing service for all Umbrella Project streams

---

## 1. Executive Summary

This doctrine defines a containerized audio transcription service that processes audio from legislative hearings, citizen submissions, and internal communications into standardized, searchable text with speaker attribution. The service operates as a tributary feeding the main UmbrellaOps data river.

---

## 2. Core Principles

1. **Universal Processor** – One service, many consumers (legislative, citizen, internal)
2. **Evidence-Grade Output** – Every transcript includes forensic chain of custody
3. **Graceful Degradation** – Better partial results than processing failure
4. **Pipeline Agnostic** – Standardized output works with any downstream consumer
5. **Attribution First** – WHO said WHAT matters more than perfect transcription

---

## 3. Standardized Output Schema

```json
{
  "job_id": "audio_20250122_135847_2f3a",
  "source_hash": "sha256:abc123...",
  "transcript_hash": "sha256:def456...",
  "processing_metadata": {
    "pipeline_version": "1.0.0",
    "models_used": {
      "whisper": "large-v3",
      "pyannote": "3.1"
    },
    "processing_strategy": "chunked",
    "chunks_processed": 6,
    "total_duration_seconds": 9091.6,
    "processing_time_seconds": 1832.5,
    "cost_estimate_usd": 0.31
  },
  "source_metadata": {
    "source_type": "legislative_hearing",
    "expected_speakers": 12,
    "environment": "clean",
    "priority": "urgent",
    "project_code": "CO_LEGISLATURE_2025",
    "classification": "public",
    "original_filename": "SB25-003_committee_hearing.mp3",
    "duration_seconds": 9091.6,
    "sample_rate": 44100,
    "channels": 2
  },
  "transcript": {
    "full_text": "SENATOR BRIDGES: The committee will come to order...",
    "language": "en",
    "segments": [
      {
        "start": 0.0,
        "end": 12.5,
        "speaker": "SPEAKER_00",
        "text": "The committee will come to order. We're here today to discuss Senate Bill 25-003.",
        "confidence": 0.95
      }
    ],
    "speakers": {
      "SPEAKER_00": {
        "label": "Sen. Bridges",
        "confidence": 0.87,
        "total_duration": 4521.3,
        "turn_count": 234,
        "identified_by": "voice_print_match"
      },
      "SPEAKER_01": {
        "label": "Unknown_Female_01",
        "confidence": 0.62,
        "total_duration": 1832.7,
        "turn_count": 89
      }
    },
    "speaker_count": 12
  },
  "quality_metrics": {
    "transcription_confidence": 0.94,
    "diarization_confidence": 0.81,
    "audio_quality_score": 0.88,
    "processing_speed_ratio": 4.96,
    "memory_peak_gb": 11.2,
    "requires_human_review": false
  },
  "chain_of_custody": {
    "submitted_by": "discovery_engine_001",
    "submitted_at": "2025-01-22T10:00:00Z",
    "processing_started": "2025-01-22T10:01:15Z",
    "processing_completed": "2025-01-22T10:31:47Z",
    "processor_id": "transcriber-gpu-node-03",
    "retention_policy": "permanent",
    "access_log": []
  },
  "extracted_entities": {
    "bills_mentioned": ["SB25-003", "HB24-1353"],
    "legislators_identified": ["Sen. Bridges", "Rep. Sirota"],
    "timestamps_extracted": {
      "roll_call": [125.3, 8976.2],
      "recess": [3601.0, 7203.5]
    }
  }
}
```

---

## 4. Service Architecture

### 4.1 API Endpoints

```yaml
# Core Endpoints
POST   /jobs                    # Submit new transcription job
GET    /jobs/{id}/status       # Check job status
GET    /jobs/{id}/result       # Retrieve completed transcript
DELETE /jobs/{id}              # Cancel in-progress job

# Webhook Management
POST   /jobs/{id}/webhook      # Register completion webhook
DELETE /jobs/{id}/webhook      # Remove webhook

# Utility Endpoints
GET    /health                 # Service health check
GET    /metrics                # Processing statistics
GET    /models                 # Available models/versions
POST   /estimate               # Cost/time estimation

# Future: WebSocket for real-time progress
WS     /jobs/{id}/stream       # Live progress updates
```

### 4.2 Job Submission Request

```json
POST /jobs
{
  "audio_url": "s3://umbrella-evidence/audio/hearing_20250122.mp3",
  "source_metadata": {
    "source_type": "legislative_hearing",
    "expected_speakers": 12,
    "environment": "clean",
    "priority": "urgent",
    "project_code": "CO_LEGISLATURE_2025",
    "classification": "public"
  },
  "processing_options": {
    "diarization": true,
    "language": "en",
    "speaker_hints": ["Sen. Bridges", "Rep. Sirota"],
    "output_format": "full",
    "max_processing_minutes": 120
  },
  "callback_webhook": "https://umbrella-api.com/webhooks/audio/complete"
}
```

---

## 5. Processing Pipeline

### 5.1 Preprocessing Module

```python
class AudioPreprocessor:
    """
    Forensic intake and intelligent segmentation
    """
    
    def process(self, job: TranscriptionJob) -> ProcessingManifest:
        # 1. Download and hash
        audio_path = self.download_audio(job.audio_url)
        source_hash = self.calculate_hash(audio_path)
        
        # 2. Analyze audio characteristics
        analysis = self.analyze_audio(audio_path)
        
        # 3. Determine processing strategy
        strategy = self.select_strategy(analysis, job.source_metadata)
        
        # 4. Create processing manifest
        manifest = ProcessingManifest(
            job_id=job.id,
            source_hash=source_hash,
            audio_path=audio_path,
            strategy=strategy,
            segments=self.segment_audio(audio_path, strategy),
            parameters=self.optimize_parameters(analysis, job.source_metadata)
        )
        
        return manifest
    
    def select_strategy(self, analysis: AudioAnalysis, metadata: Dict) -> str:
        """
        Strategy selection based on duration and type
        """
        duration = analysis.duration_seconds
        source_type = metadata.get('source_type', 'unknown')
        
        if duration < 1800:  # < 30 minutes
            return "standard"
        elif duration < 7200:  # < 2 hours
            return "chunked"
        elif source_type == "legislative_hearing":
            return "legislative_optimized"
        else:
            return "distributed"
    
    def segment_audio(self, audio_path: str, strategy: str) -> List[Segment]:
        """
        Intelligent segmentation by strategy
        """
        if strategy == "legislative_optimized":
            # Detect gavels, roll calls, recess announcements
            return self.segment_by_legislative_markers(audio_path)
        elif strategy == "chunked":
            # Find natural breaks (silence, speaker changes)
            return self.segment_by_natural_boundaries(audio_path)
        else:
            # Time-based chunks with overlap
            return self.segment_by_duration(audio_path, chunk_minutes=30)
```

### 5.2 Transcription Processor

```python
class TranscriptionProcessor:
    """
    Core transcription with speaker attribution
    """
    
    def process(self, manifest: ProcessingManifest) -> TranscriptionResult:
        # 1. Load appropriate models
        whisper_model = self.load_whisper(manifest.parameters.whisper_model)
        diarization_pipeline = self.load_pyannote()
        
        # 2. Process based on strategy
        if manifest.strategy == "standard":
            result = self.process_standard(manifest, whisper_model, diarization_pipeline)
        elif manifest.strategy == "chunked":
            result = self.process_chunked(manifest, whisper_model, diarization_pipeline)
        elif manifest.strategy == "legislative_optimized":
            result = self.process_legislative(manifest, whisper_model, diarization_pipeline)
        else:
            result = self.process_distributed(manifest, whisper_model, diarization_pipeline)
        
        # 3. Post-processing
        result = self.reconcile_speakers(result)
        result = self.extract_entities(result)
        result = self.calculate_confidence_scores(result)
        
        return result
    
    def process_legislative(self, manifest: ProcessingManifest, 
                          whisper_model, diarization_pipeline) -> TranscriptionResult:
        """
        Special handling for government audio
        """
        # Load known speaker voice prints
        voice_prints = self.load_legislator_voiceprints(manifest.metadata.get('state', 'CO'))
        
        # Process with enhanced attribution
        transcript = whisper_model.transcribe(manifest.audio_path)
        diarization = diarization_pipeline(manifest.audio_path)
        
        # Match speakers to known legislators
        speaker_mapping = self.match_speakers_to_voiceprints(
            diarization, 
            voice_prints,
            manifest.metadata.get('speaker_hints', [])
        )
        
        # Extract legislative markers
        markers = self.extract_legislative_markers(transcript)
        
        return TranscriptionResult(
            transcript=transcript,
            diarization=diarization,
            speaker_mapping=speaker_mapping,
            markers=markers
        )
```

---

## 6. Priority Queue Management

```python
PRIORITY_LEVELS = {
    "emergency": 0,     # Live legislative session
    "urgent": 1,        # Time-sensitive opposition research  
    "normal": 2,        # Regular meeting minutes
    "batch": 3,         # Historical archive processing
    "citizen": 4        # Public submissions
}

class JobQueue:
    """
    Priority-based job scheduling
    """
    
    def enqueue(self, job: TranscriptionJob):
        priority = PRIORITY_LEVELS.get(job.metadata.get('priority', 'normal'), 2)
        
        # Emergency jobs preempt current processing
        if priority == 0 and self.current_job and self.current_job.priority > 0:
            self.suspend_current()
            
        self.queue.put((priority, job.submitted_at, job))
    
    def get_resource_allocation(self, job: TranscriptionJob) -> ResourceAllocation:
        """
        Allocate resources based on priority
        """
        if job.priority == "emergency":
            return ResourceAllocation(
                gpu_memory_gb=24,
                cpu_cores=16,
                timeout_minutes=None
            )
        elif job.priority == "urgent":
            return ResourceAllocation(
                gpu_memory_gb=16,
                cpu_cores=8,
                timeout_minutes=120
            )
        else:
            return ResourceAllocation(
                gpu_memory_gb=12,
                cpu_cores=4,
                timeout_minutes=240
            )
```

---

## 7. Error Handling & Fallbacks

```python
class ProcessingFallbacks:
    """
    Graceful degradation strategies
    """
    
    def handle_diarization_failure(self, job: TranscriptionJob, 
                                  transcript: WhisperResult) -> TranscriptionResult:
        """
        If diarization fails, still deliver value
        """
        return TranscriptionResult(
            transcript=transcript,
            speakers={"SPEAKER_00": {"label": "Unknown", "confidence": 0.0}},
            quality_metrics={
                "diarization_confidence": 0.0,
                "requires_human_review": True,
                "fallback_reason": "diarization_timeout"
            }
        )
    
    def handle_memory_pressure(self, job: TranscriptionJob) -> ProcessingManifest:
        """
        Switch to chunked processing if memory constrained
        """
        logger.warning(f"Memory pressure detected for job {job.id}, switching to chunked mode")
        
        # Re-segment with smaller chunks
        new_manifest = self.preprocessor.create_manifest(
            job, 
            force_strategy="chunked",
            chunk_size_minutes=15
        )
        
        return new_manifest
```

---

## 8. Deployment Configuration

### 8.1 Container Specification

```dockerfile
FROM nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    libsndfile1

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Download models at build time
RUN python -c "import whisper; whisper.load_model('large-v3')"

# Copy application
COPY src/ /app/src/
WORKDIR /app

# API server
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 8.2 Resource Requirements

```yaml
# Kubernetes deployment
resources:
  requests:
    memory: "16Gi"
    cpu: "4"
    nvidia.com/gpu: "1"
  limits:
    memory: "32Gi"
    cpu: "8"
    nvidia.com/gpu: "1"

# Environment variables
env:
  - name: WHISPER_MODEL
    value: "large-v3"
  - name: PYANNOTE_TOKEN
    valueFrom:
      secretKeyRef:
        name: huggingface-token
        key: token
  - name: MAX_CONCURRENT_JOBS
    value: "4"
  - name: GPU_MEMORY_FRACTION
    value: "0.9"
  - name: S3_BUCKET
    value: "umbrella-audio-processing"
```

---

## 9. Monitoring & Metrics

```python
METRICS = {
    # Performance metrics
    "processing_speed_ratio": Histogram("audio_processing_speed", 
                                       "Realtime speed multiplier"),
    "job_duration_seconds": Histogram("audio_job_duration", 
                                     "Total processing time"),
    "queue_depth": Gauge("audio_queue_depth", 
                        "Number of pending jobs by priority"),
    
    # Quality metrics
    "transcription_confidence": Histogram("audio_transcript_confidence",
                                         "Whisper confidence scores"),
    "diarization_confidence": Histogram("audio_diarization_confidence",
                                       "Speaker attribution confidence"),
    
    # Resource metrics
    "gpu_memory_usage": Gauge("audio_gpu_memory_bytes", 
                             "GPU memory consumption"),
    "cost_per_minute": Histogram("audio_cost_per_minute_usd",
                                "Processing cost per audio minute")
}
```

---

## 10. Cost Model

| Audio Type | Duration | Strategy | Cost Estimate |
|------------|----------|----------|---------------|
| Phone call | < 30 min | Standard | $0.18 |
| Meeting | 1-2 hours | Chunked | $0.72 |
| Hearing | 3-6 hours | Legislative | $2.16 |
| Marathon | 6+ hours | Distributed | $0.36/hour |

**Cost Optimization Rules**:
- Batch processing gets 50% discount
- Emergency priority gets 2x surcharge
- Failed diarization credits 30% back
- Citizen submissions subsidized by revenue streams

---

## 11. Integration Examples

### 11.1 n8n Workflow Node

```javascript
// Audio Transcription Node
{
  "name": "Transcribe Audio Evidence",
  "type": "umbrella.audio.transcribe",
  "parameters": {
    "audio_source": "={{$node['Download_Hearing'].data.file_path}}",
    "source_type": "legislative_hearing",
    "priority": "normal",
    "speaker_hints": "={{$node['Get_Committee_Members'].data.names}}",
    "project_code": "CO_LEGISLATURE_2025"
  },
  "webhook": {
    "url": "={{$env.WEBHOOK_URL}}/audio/complete",
    "headers": {
      "X-Umbrella-Token": "={{$env.UMBRELLA_TOKEN}}"
    }
  }
}
```

### 11.2 Python Client

```python
from umbrella_audio import TranscriptionClient

client = TranscriptionClient(base_url="https://audio.umbrellaops.com")

# Submit job
job = client.submit_job(
    audio_url="s3://bucket/hearing.mp3",
    source_type="legislative_hearing",
    priority="urgent",
    speaker_hints=["Sen. Bridges", "Rep. Sirota"]
)

# Poll for completion
while job.status != "completed":
    time.sleep(30)
    job.refresh()

# Get results
transcript = job.get_result()
print(f"Identified {len(transcript['speakers'])} speakers")
print(f"Transcription confidence: {transcript['quality_metrics']['transcription_confidence']}")
```

---

## 12. Security & Compliance

### 12.1 Access Control

```python
ACCESS_LEVELS = {
    "public": ["legislative_hearing", "public_meeting"],
    "confidential": ["internal_comm", "strategy_session"],
    "secret": ["executive_session", "classified_briefing"]
}

def validate_access(user: User, job: TranscriptionJob) -> bool:
    """
    Ensure user can access transcript based on classification
    """
    classification = job.metadata.get('classification', 'public')
    return user.clearance_level >= CLEARANCE_REQUIRED[classification]
```

### 12.2 Data Retention

```python
RETENTION_POLICIES = {
    "legislative_hearing": "permanent",
    "citizen_submission": "7_years",
    "internal_comm": "90_days",
    "phone_call": "1_year"
}
```

---

## 13. Future Enhancements

### Phase 2 (With Revenue)
- Real-time streaming transcription
- Multi-language support (Spanish for testimony)
- Speaker identification training interface
- Custom vocabulary for domain terms

### Phase 3 (At Scale)
- Federated processing across regions
- On-premise deployment option
- API rate limiting and quotas
- Advanced analytics dashboard

---

## 14. Success Metrics

- **Throughput**: Process 500+ hours/day
- **Accuracy**: >95% transcription, >85% speaker attribution
- **Latency**: Results within 30% of audio duration
- **Availability**: 99.9% uptime for urgent/emergency
- **Cost**: <$0.01/minute average processing cost

---

**Operator Signature**  
Sean Steele | 2025-01-22

*"Every word spoken in the arena of power must be captured, attributed, and preserved. This is how memory defeats rhetoric."*