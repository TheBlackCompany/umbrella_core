#!/usr/bin/env python3
"""
MCP Server Wrapper for Umbrella Transcriber
Allows Claude and other AIs to use the transcription service via MCP
"""

import os
import json
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path
import httpx
from datetime import datetime

class UmbrellaTranscriberMCP:
    """MCP-compatible wrapper for the Umbrella Transcriber API"""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("UMBRELLA_API_KEY")
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"X-API-Key": self.api_key} if self.api_key else {}
        )
    
    async def transcribe_audio(
        self,
        audio_url: str,
        source_type: str = "meeting",
        expected_speakers: Optional[int] = None,
        priority: str = "normal",
        language: str = "en",
        webhook_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Submit an audio file for transcription
        
        Args:
            audio_url: URL or file path to audio
            source_type: Type of audio (legislative_hearing, meeting, phone_call, etc.)
            expected_speakers: Expected number of speakers
            priority: Processing priority (emergency, urgent, normal, batch, citizen)
            language: Audio language code
            webhook_url: Optional webhook for completion notification
            
        Returns:
            Job submission response with job_id
        """
        # Convert local file paths to file:// URLs if needed
        if os.path.exists(audio_url):
            audio_url = f"file://{Path(audio_url).absolute()}"
        
        payload = {
            "audio_url": audio_url,
            "source_metadata": {
                "source_type": source_type,
                "priority": priority,
                "environment": "clean",
                "classification": "public"
            },
            "processing_options": {
                "diarization": True,
                "language": language,
                "output_format": "full"
            }
        }
        
        if expected_speakers:
            payload["source_metadata"]["expected_speakers"] = expected_speakers
        
        if webhook_url:
            payload["callback_webhook"] = webhook_url
        
        response = await self.client.post("/jobs", json=payload)
        response.raise_for_status()
        return response.json()
    
    async def check_status(self, job_id: str) -> Dict[str, Any]:
        """Check the status of a transcription job"""
        response = await self.client.get(f"/jobs/{job_id}/status")
        response.raise_for_status()
        return response.json()
    
    async def get_transcript(self, job_id: str) -> Dict[str, Any]:
        """Retrieve completed transcript"""
        response = await self.client.get(f"/jobs/{job_id}/result")
        response.raise_for_status()
        return response.json()
    
    async def wait_for_completion(
        self, 
        job_id: str, 
        check_interval: int = 5,
        max_wait: int = 3600
    ) -> Dict[str, Any]:
        """
        Wait for a job to complete and return the transcript
        
        Args:
            job_id: Job ID to monitor
            check_interval: Seconds between status checks
            max_wait: Maximum seconds to wait
            
        Returns:
            Completed transcript
        """
        start_time = datetime.now()
        
        while (datetime.now() - start_time).total_seconds() < max_wait:
            status = await self.check_status(job_id)
            
            if status["status"] == "completed":
                return await self.get_transcript(job_id)
            elif status["status"] == "failed":
                raise Exception(f"Job failed: {status.get('error', 'Unknown error')}")
            elif status["status"] == "cancelled":
                raise Exception("Job was cancelled")
            
            await asyncio.sleep(check_interval)
        
        raise TimeoutError(f"Job did not complete within {max_wait} seconds")
    
    async def transcribe_and_wait(
        self,
        audio_url: str,
        source_type: str = "meeting",
        expected_speakers: Optional[int] = None,
        priority: str = "normal",
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Submit transcription and wait for completion
        
        This is a convenience method that combines submission and polling
        """
        # Submit job
        job_response = await self.transcribe_audio(
            audio_url=audio_url,
            source_type=source_type,
            expected_speakers=expected_speakers,
            priority=priority,
            language=language
        )
        
        job_id = job_response["job_id"]
        
        # Wait for completion
        return await self.wait_for_completion(job_id)
    
    async def estimate_cost(
        self,
        duration_seconds: float,
        priority: str = "normal"
    ) -> Dict[str, Any]:
        """Estimate cost and time for transcription"""
        response = await self.client.post("/estimate", json={
            "duration_seconds": duration_seconds,
            "priority": priority,
            "diarization": True
        })
        response.raise_for_status()
        return response.json()
    
    async def get_health(self) -> Dict[str, Any]:
        """Check service health"""
        response = await self.client.get("/health")
        response.raise_for_status()
        return response.json()
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


# MCP tool definitions for registration
MCP_TOOLS = [
    {
        "name": "transcribe_audio",
        "description": "Submit audio for transcription with speaker diarization",
        "parameters": {
            "audio_url": "Path or URL to audio file",
            "source_type": "Type: legislative_hearing, meeting, phone_call, interview",
            "expected_speakers": "Expected number of speakers (optional)",
            "priority": "Priority: emergency, urgent, normal, batch, citizen",
            "language": "Language code (default: en)"
        }
    },
    {
        "name": "check_transcription_status",
        "description": "Check status of transcription job",
        "parameters": {
            "job_id": "Job ID from transcribe_audio"
        }
    },
    {
        "name": "get_transcript",
        "description": "Get completed transcript with speaker labels",
        "parameters": {
            "job_id": "Job ID of completed transcription"
        }
    },
    {
        "name": "transcribe_and_wait",
        "description": "Submit transcription and wait for completion",
        "parameters": {
            "audio_url": "Path or URL to audio file",
            "source_type": "Type of audio source",
            "expected_speakers": "Expected speakers (optional)",
            "priority": "Processing priority",
            "language": "Language code"
        }
    }
]


async def main():
    """Example usage"""
    mcp = UmbrellaTranscriberMCP()
    
    try:
        # Check health
        health = await mcp.get_health()
        print(f"Service health: {health}")
        
        # Example transcription
        result = await mcp.transcribe_and_wait(
            audio_url="path/to/audio.wav",
            source_type="meeting",
            expected_speakers=2,
            priority="normal"
        )
        
        print(f"Transcript: {json.dumps(result, indent=2)}")
        
    finally:
        await mcp.close()


if __name__ == "__main__":
    asyncio.run(main())