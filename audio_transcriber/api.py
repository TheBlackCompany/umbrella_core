#!/usr/bin/env python3
"""
Umbrella Transcriber API
Machine-callable REST interface
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
from typing import Optional, List, Dict
import uuid

from engine import TranscriptionEngine
from context_system import ConversationType, ConversationContext


# Request/Response models
class TranscriptionRequest(BaseModel):
    """API request schema"""
    file_path: str
    output_dir: Optional[str] = None
    model_size: str = "large"
    expected_speakers: Optional[int] = None
    context_type: Optional[str] = None
    context_description: Optional[str] = None


class TranscriptionResponse(BaseModel):
    """API response schema"""
    job_id: str
    status: str
    result: Optional[Dict] = None
    error: Optional[str] = None


# API setup
app = FastAPI(title="Umbrella Transcriber API", version="1.0.0")
engine = TranscriptionEngine()
jobs = {}  # In production, use Redis or database


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(request: TranscriptionRequest, background_tasks: BackgroundTasks):
    """Submit transcription job"""
    job_id = str(uuid.uuid4())
    
    # Validate input
    input_path = Path(request.file_path)
    if not input_path.exists():
        raise HTTPException(status_code=404, detail="Input file not found")
    
    # Create context if provided
    context = None
    if request.context_type:
        context = ConversationContext(
            type=ConversationType(request.context_type),
            expected_speakers=request.expected_speakers,
            description=request.context_description
        )
    
    # Submit job
    jobs[job_id] = {"status": "processing"}
    background_tasks.add_task(
        process_job,
        job_id,
        input_path,
        Path(request.output_dir) if request.output_dir else None,
        context,
        request.expected_speakers
    )
    
    return TranscriptionResponse(job_id=job_id, status="processing")


@app.get("/status/{job_id}", response_model=TranscriptionResponse)
async def get_status(job_id: str):
    """Get job status"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return TranscriptionResponse(
        job_id=job_id,
        status=job["status"],
        result=job.get("result"),
        error=job.get("error")
    )


def process_job(job_id: str, input_path: Path, output_dir: Optional[Path],
                context: Optional[ConversationContext], expected_speakers: Optional[int]):
    """Process transcription job"""
    try:
        result = engine.process(
            input_path,
            output_dir,
            context,
            expected_speakers
        )
        jobs[job_id] = {
            "status": "complete",
            "result": result
        }
    except Exception as e:
        jobs[job_id] = {
            "status": "failed",
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)