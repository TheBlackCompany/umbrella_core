#!/usr/bin/env python3
"""
Umbrella Transcriber API - Doctrine Compliant
All endpoints per audio-transcription-doctrine.md
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uuid
import redis
import json
from datetime import datetime
from pathlib import Path

from engine import TranscriptionEngine
from context_system import ConversationType, ConversationContext
from priority_queue import Priority, TranscriptionJob, PriorityJobQueue
from schema import DoctrineSchema


# Request/Response models per doctrine
class SourceMetadata(BaseModel):
    source_type: str = Field(..., description="legislative_hearing, citizen_submission, etc.")
    expected_speakers: Optional[int] = None
    environment: str = "clean"
    priority: str = "normal"
    project_code: Optional[str] = None
    classification: str = "public"
    speaker_hints: Optional[List[str]] = None


class ProcessingOptions(BaseModel):
    diarization: bool = True
    language: str = "en"
    speaker_hints: Optional[List[str]] = []
    output_format: str = "full"
    max_processing_minutes: Optional[int] = 120


class JobSubmissionRequest(BaseModel):
    """Doctrine-compliant job submission"""
    audio_url: str
    source_metadata: SourceMetadata
    processing_options: ProcessingOptions = ProcessingOptions()
    callback_webhook: Optional[str] = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: Optional[float] = None
    estimated_completion: Optional[str] = None
    priority: str
    submitted_at: str
    processing_started: Optional[str] = None


class EstimateRequest(BaseModel):
    duration_seconds: float
    priority: str = "normal"
    diarization: bool = True


class EstimateResponse(BaseModel):
    estimated_cost_usd: float
    estimated_time_seconds: float
    recommended_priority: str


# Initialize components
app = FastAPI(
    title="Umbrella Audio Transcription Service",
    version="1.0.0",
    description="Doctrine-compliant audio processing per v1.0 specification"
)

# In production, use Redis or database
redis_client = None  # redis.Redis(host='localhost', port=6379, db=0)
jobs_db = {}  # Fallback to in-memory
queue = PriorityJobQueue()
engine = TranscriptionEngine()


# Dependency for job storage
def get_job_store():
    if redis_client:
        return redis_client
    return jobs_db


# Core Endpoints per doctrine
@app.post("/jobs", response_model=Dict[str, str])
async def submit_job(request: JobSubmissionRequest, background_tasks: BackgroundTasks):
    """Submit new transcription job"""
    job_id = f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
    
    # Create job
    job = TranscriptionJob(
        id=job_id,
        audio_url=request.audio_url,
        priority=Priority[request.source_metadata.priority.upper()],
        source_metadata=request.source_metadata.dict(),
        processing_options=request.processing_options.dict(),
        submitted_at=datetime.now(),
        webhook_url=request.callback_webhook
    )
    
    # Store job
    job_data = {
        "status": "queued",
        "job": job.__dict__,
        "submitted_at": job.submitted_at.isoformat()
    }
    
    if redis_client:
        redis_client.set(f"job:{job_id}", json.dumps(job_data))
    else:
        jobs_db[job_id] = job_data
    
    # Enqueue for processing
    queue.enqueue(job)
    background_tasks.add_task(process_job_async, job)
    
    return {"job_id": job_id, "status": "accepted"}


@app.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Check job status"""
    # Get from storage
    if redis_client:
        job_data = redis_client.get(f"job:{job_id}")
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found")
        job_data = json.loads(job_data)
    else:
        if job_id not in jobs_db:
            raise HTTPException(status_code=404, detail="Job not found")
        job_data = jobs_db[job_id]
    
    return JobStatusResponse(
        job_id=job_id,
        status=job_data["status"],
        progress=job_data.get("progress"),
        estimated_completion=job_data.get("estimated_completion"),
        priority=job_data["job"]["source_metadata"]["priority"],
        submitted_at=job_data["submitted_at"],
        processing_started=job_data.get("processing_started")
    )


@app.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    """Retrieve completed transcript"""
    # Get from storage
    if redis_client:
        job_data = redis_client.get(f"job:{job_id}")
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found")
        job_data = json.loads(job_data)
    else:
        if job_id not in jobs_db:
            raise HTTPException(status_code=404, detail="Job not found")
        job_data = jobs_db[job_id]
    
    if job_data["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed. Status: {job_data['status']}")
    
    return job_data.get("result", {})


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel in-progress job"""
    # Update status
    if redis_client:
        job_data = redis_client.get(f"job:{job_id}")
        if job_data:
            job_data = json.loads(job_data)
            job_data["status"] = "cancelled"
            redis_client.set(f"job:{job_id}", json.dumps(job_data))
    else:
        if job_id in jobs_db:
            jobs_db[job_id]["status"] = "cancelled"
        else:
            raise HTTPException(status_code=404, detail="Job not found")
    
    return {"job_id": job_id, "status": "cancelled"}


# Webhook Management
@app.post("/jobs/{job_id}/webhook")
async def register_webhook(job_id: str, webhook_url: str):
    """Register completion webhook"""
    # Update job with webhook
    if redis_client:
        job_data = redis_client.get(f"job:{job_id}")
        if job_data:
            job_data = json.loads(job_data)
            job_data["webhook_url"] = webhook_url
            redis_client.set(f"job:{job_id}", json.dumps(job_data))
    else:
        if job_id in jobs_db:
            jobs_db[job_id]["webhook_url"] = webhook_url
        else:
            raise HTTPException(status_code=404, detail="Job not found")
    
    return {"job_id": job_id, "webhook_registered": True}


# Utility Endpoints
@app.get("/health")
async def health_check():
    """Service health check"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "models_loaded": True,
        "gpu_available": True,  # Check actual GPU
        "queue_depth": queue.get_queue_depth()
    }


@app.get("/metrics")
async def get_metrics():
    """Processing statistics"""
    return {
        "jobs_processed": 0,  # Track in production
        "average_processing_time": 0,
        "queue_depth_by_priority": queue.get_queue_depth(),
        "success_rate": 1.0
    }


@app.get("/models")
async def get_models():
    """Available models/versions"""
    return {
        "whisper_models": ["tiny", "base", "small", "medium", "large"],
        "default_whisper": "large",
        "pyannote_version": "3.1",
        "legislative_optimization": True
    }


@app.post("/estimate", response_model=EstimateResponse)
async def estimate_job(request: EstimateRequest):
    """Cost/time estimation"""
    cost = DoctrineSchema.estimate_cost(request.duration_seconds, request.priority)
    
    # Time estimate based on priority
    speed_multipliers = {
        "emergency": 8.0,
        "urgent": 5.0,
        "normal": 3.0,
        "batch": 2.0,
        "citizen": 1.5
    }
    
    speed = speed_multipliers.get(request.priority, 3.0)
    estimated_time = request.duration_seconds / speed
    
    # Recommend priority based on duration
    if request.duration_seconds > 10800:  # > 3 hours
        recommended = "batch"
    elif request.duration_seconds > 3600:  # > 1 hour
        recommended = "normal"
    else:
        recommended = request.priority
    
    return EstimateResponse(
        estimated_cost_usd=cost,
        estimated_time_seconds=estimated_time,
        recommended_priority=recommended
    )


# Background processing
async def process_job_async(job: TranscriptionJob):
    """Process job in background"""
    job_id = job.id
    
    # Update status
    update_job_status(job_id, "processing", processing_started=datetime.now().isoformat())
    
    try:
        # Create context from metadata
        context = None
        if job.source_metadata.get("source_type") == "legislative_hearing":
            context = ConversationContext(
                type=ConversationType.LEGISLATIVE,
                expected_speakers=job.source_metadata.get("expected_speakers"),
                description=f"Legislative hearing - {job.source_metadata.get('project_code', '')}"
            )
        
        # Process with engine
        # In production, download from audio_url first
        audio_path = Path(job.audio_url.replace("s3://", "/data/"))  # Placeholder
        
        result = engine.process(
            audio_path,
            context=context,
            expected_speakers=job.source_metadata.get("expected_speakers")
        )
        
        # Update with result
        update_job_status(job_id, "completed", result=result)
        
        # Call webhook if registered
        if job.webhook_url:
            # In production, make actual HTTP call
            pass
            
    except Exception as e:
        update_job_status(job_id, "failed", error=str(e))


def update_job_status(job_id: str, status: str, **kwargs):
    """Update job status in storage"""
    if redis_client:
        job_data = redis_client.get(f"job:{job_id}")
        if job_data:
            job_data = json.loads(job_data)
            job_data["status"] = status
            job_data.update(kwargs)
            redis_client.set(f"job:{job_id}", json.dumps(job_data))
    else:
        if job_id in jobs_db:
            jobs_db[job_id]["status"] = status
            jobs_db[job_id].update(kwargs)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)