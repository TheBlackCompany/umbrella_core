# MCP Integration Guide for Umbrella Transcriber

## Overview

The Umbrella Transcriber service is fully compatible with the Model Context Protocol (MCP), allowing Claude and other AI assistants to use the transcription service seamlessly.

## Quick Start for AI Assistants

### 1. Using the REST API Directly

The service exposes a RESTful API at `http://localhost:8000` with the following key endpoints:

```bash
# Submit a transcription job
POST /jobs
{
  "audio_url": "file:///path/to/audio.wav",
  "source_metadata": {
    "source_type": "meeting",
    "expected_speakers": 2,
    "priority": "normal"
  },
  "processing_options": {
    "diarization": true,
    "language": "en"
  }
}

# Check job status
GET /jobs/{job_id}/status

# Get completed transcript
GET /jobs/{job_id}/result
```

### 2. Using the MCP Wrapper

```python
from mcp_wrapper import UmbrellaTranscriberMCP

# Initialize client
mcp = UmbrellaTranscriberMCP(base_url="http://localhost:8000")

# Submit and wait for transcription
result = await mcp.transcribe_and_wait(
    audio_url="path/to/audio.wav",
    source_type="meeting",
    expected_speakers=2
)

# Access transcript segments
for segment in result['transcript']['segments']:
    print(f"[{segment['speaker']}] {segment['text']}")
```

## MCP Server Configuration

The service can be registered as an MCP server using the provided `mcp_server.json`:

```json
{
  "name": "umbrella-transcriber",
  "version": "1.0.0",
  "type": "http",
  "config": {
    "baseUrl": "http://localhost:8000"
  },
  "tools": [
    {
      "name": "transcribe_audio",
      "description": "Submit audio for transcription with speaker diarization"
    },
    {
      "name": "check_transcription_status",
      "description": "Check status of transcription job"
    },
    {
      "name": "get_transcript",
      "description": "Get completed transcript with speaker labels"
    }
  ]
}
```

## Available Tools for AI Assistants

### 1. transcribe_audio
Submit an audio file for transcription with automatic speaker diarization.

**Parameters:**
- `audio_url` (required): Path or URL to the audio file
- `source_type` (required): Type of audio (legislative_hearing, meeting, phone_call, interview, citizen_submission)
- `expected_speakers` (optional): Expected number of speakers
- `priority` (optional): Processing priority (emergency, urgent, normal, batch, citizen)
- `language` (optional): Language code (default: en)

**Returns:** Job ID and status

### 2. check_transcription_status
Check the status of a submitted transcription job.

**Parameters:**
- `job_id` (required): The job ID returned from transcribe_audio

**Returns:** Status information including progress and estimated completion

### 3. get_transcript
Retrieve the completed transcript with speaker labels and timestamps.

**Parameters:**
- `job_id` (required): The job ID of a completed transcription

**Returns:** Full transcript with:
- Speaker-labeled segments
- Timestamps for each segment
- Quality metrics
- Extracted entities (for legislative content)
- Processing metadata

### 4. estimate_transcription_cost
Estimate the cost and processing time before submitting.

**Parameters:**
- `duration_seconds` (required): Audio duration in seconds
- `priority` (optional): Processing priority

**Returns:** Cost estimate and recommended priority

## Example Workflows

### Basic Transcription
```python
# 1. Submit audio
job = await mcp.transcribe_audio(
    audio_url="interview.wav",
    source_type="interview",
    expected_speakers=2
)

# 2. Wait for completion
while True:
    status = await mcp.check_status(job["job_id"])
    if status["status"] == "completed":
        break
    await asyncio.sleep(5)

# 3. Get transcript
transcript = await mcp.get_transcript(job["job_id"])
```

### Legislative Hearing Processing
```python
# Submit with legislative optimization
job = await mcp.transcribe_audio(
    audio_url="senate_hearing.mp3",
    source_type="legislative_hearing",
    priority="urgent"
)

# Get result with bill references and speaker identification
result = await mcp.wait_for_completion(job["job_id"])

# Access extracted legislative entities
bills = result["extracted_entities"]["bills"]
legislators = result["extracted_entities"]["legislators"]
```

## Authentication

If the service requires authentication, set the API key:

```python
# Via environment variable
os.environ["UMBRELLA_API_KEY"] = "your-api-key"

# Or directly
mcp = UmbrellaTranscriberMCP(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)
```

## Output Schema

The service returns transcripts in a structured format:

```json
{
  "job_id": "audio_20241207_143022_a1b2",
  "source_hash": "sha256:...",
  "transcript_hash": "sha256:...",
  "processing_metadata": {
    "model": "whisper-large",
    "processing_time": 125.3,
    "strategies_used": ["chunked"]
  },
  "transcript": {
    "segments": [
      {
        "speaker": "SPEAKER_1",
        "text": "Hello, how are you?",
        "start": 0.0,
        "end": 2.5,
        "confidence": 0.95
      }
    ],
    "full_text": "Complete transcript text...",
    "speaker_map": {
      "SPEAKER_1": "speaker_abc123",
      "SPEAKER_2": "speaker_def456"
    }
  },
  "quality_metrics": {
    "audio_quality": "high",
    "confidence_avg": 0.92,
    "speaker_overlap_ratio": 0.05
  },
  "extracted_entities": {
    "bills": ["SB 123", "HB 456"],
    "legislators": ["Senator Smith", "Representative Jones"],
    "organizations": ["Committee on Finance"]
  }
}
```

## Best Practices for AI Assistants

1. **Check service health** before submitting jobs:
   ```python
   health = await mcp.get_health()
   if health["status"] != "healthy":
       print("Service unavailable")
   ```

2. **Use appropriate source types** for better accuracy:
   - `legislative_hearing`: Optimizes for formal speech, bill references
   - `phone_call`: Expects 2 speakers, optimizes for conversational speech
   - `meeting`: Multi-speaker optimization
   - `interview`: Typically 2 speakers with Q&A format

3. **Set expected speakers** when known:
   - Improves speaker consolidation accuracy
   - Reduces over-segmentation

4. **Handle long audio files**:
   - Files over 30 minutes are automatically chunked
   - Use `batch` priority for files over 3 hours

5. **Monitor job progress**:
   - Check status periodically
   - Use webhooks for async notification

## Error Handling

Common errors and solutions:

- **404 Job not found**: Job ID is invalid or expired
- **400 Job not completed**: Wait for completion before getting results
- **503 Service unavailable**: Check service health and GPU availability
- **413 File too large**: Use URL instead of direct upload for large files

## Performance Notes

- GPU acceleration provides 3-8x speedup
- Priority levels affect processing speed:
  - Emergency: 8x speed (highest GPU allocation)
  - Urgent: 5x speed
  - Normal: 3x speed
  - Batch: 2x speed
  - Citizen: 1.5x speed

## Docker Deployment

For AI assistants running in containers:

```bash
# Access service from another container
docker run --network umbrella_network \
  my-ai-assistant \
  --transcriber-url http://umbrella-transcriber:8000
```

## Support

- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health
- Metrics: http://localhost:8000/metrics