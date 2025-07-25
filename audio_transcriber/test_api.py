#!/usr/bin/env python3
"""Test the API (doctrine-compliant implementation)"""

import os
import sys
import time
import requests
import json
from pathlib import Path

# Add FFmpeg to PATH
if os.name == 'nt':
    ffmpeg_paths = [
        r"C:\ffmpeg\bin",
        r"C:\Program Files\ffmpeg\bin",
        r"C:\ProgramData\chocolatey\bin"
    ]
    for path in ffmpeg_paths:
        if os.path.exists(path) and path not in os.environ["PATH"]:
            os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]
            break

API_BASE = "http://localhost:8000"

# Test endpoints
def test_health():
    """Test health endpoint"""
    try:
        resp = requests.get(f"{API_BASE}/health")
        if resp.status_code == 200:
            print("[OK] Health check passed")
            print(f"    Response: {resp.json()}")
        else:
            print(f"[FAIL] Health check failed: {resp.status_code}")
    except Exception as e:
        print(f"[FAIL] Could not connect to API: {e}")
        return False
    return True

def test_metrics():
    """Test metrics endpoint"""
    try:
        resp = requests.get(f"{API_BASE}/metrics")
        if resp.status_code == 200:
            print("[OK] Metrics endpoint working")
            print(f"    Jobs processed: {resp.json().get('jobs_processed', 0)}")
        else:
            print(f"[FAIL] Metrics failed: {resp.status_code}")
    except Exception as e:
        print(f"[FAIL] Metrics error: {e}")

def test_job_submission():
    """Test job submission"""
    audio_file = Path("raw/talk with josh of comm committee.m4a")
    if not audio_file.exists():
        print(f"[FAIL] Test audio file not found: {audio_file}")
        return None
        
    # Read file
    with open(audio_file, 'rb') as f:
        file_data = f.read()
    
    # Submit job
    try:
        files = {'file': (audio_file.name, file_data, 'audio/mp4')}
        data = {
            'priority': 'normal',
            'expected_speakers': 2,
            'context_type': 'phone_call',
            'description': 'Committee discussion test'
        }
        
        resp = requests.post(f"{API_BASE}/jobs", files=files, data=data)
        if resp.status_code == 200:
            job_info = resp.json()
            print(f"[OK] Job submitted successfully")
            print(f"    Job ID: {job_info['job_id']}")
            return job_info['job_id']
        else:
            print(f"[FAIL] Job submission failed: {resp.status_code}")
            print(f"    Response: {resp.text}")
    except Exception as e:
        print(f"[FAIL] Job submission error: {e}")
    return None

def test_job_status(job_id):
    """Check job status"""
    try:
        resp = requests.get(f"{API_BASE}/jobs/{job_id}/status")
        if resp.status_code == 200:
            status = resp.json()
            print(f"[OK] Job status retrieved")
            print(f"    Status: {status['status']}")
            print(f"    Progress: {status.get('progress', 0)}%")
            return status
        else:
            print(f"[FAIL] Status check failed: {resp.status_code}")
    except Exception as e:
        print(f"[FAIL] Status check error: {e}")
    return None

def test_job_result(job_id):
    """Get job result"""
    try:
        resp = requests.get(f"{API_BASE}/jobs/{job_id}/result")
        if resp.status_code == 200:
            result = resp.json()
            print(f"[OK] Job result retrieved")
            print(f"    Transcript segments: {len(result['transcript']['segments'])}")
            print(f"    Speakers: {result['transcript']['speaker_count']}")
            return result
        else:
            print(f"[FAIL] Result retrieval failed: {resp.status_code}")
    except Exception as e:
        print(f"[FAIL] Result retrieval error: {e}")
    return None

def main():
    print("Testing Doctrine-Compliant API")
    print("===============================\n")
    
    # Check if API is running
    if not test_health():
        print("\n[INFO] API server not running. Start it with:")
        print("cd umbrella_transcriber && ./venv/Scripts/python.exe -m uvicorn api_doctrine:app --reload")
        return
    
    # Test other endpoints
    print("\nTesting endpoints...")
    test_metrics()
    
    # Submit a job
    print("\nSubmitting transcription job...")
    job_id = test_job_submission()
    
    if job_id:
        # Poll for completion
        print("\nWaiting for job to complete...")
        max_wait = 300  # 5 minutes
        start_time = time.time()
        
        while (time.time() - start_time) < max_wait:
            status = test_job_status(job_id)
            if status and status['status'] == 'completed':
                print("\nJob completed! Getting result...")
                result = test_job_result(job_id)
                
                # Save result
                if result:
                    output_file = Path("doctrine_test_result.json")
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2)
                    print(f"\n[OK] Result saved to: {output_file}")
                break
            elif status and status['status'] == 'failed':
                print(f"\n[FAIL] Job failed: {status.get('error', 'Unknown error')}")
                break
            
            time.sleep(5)
        else:
            print("\n[FAIL] Job did not complete within timeout")

if __name__ == "__main__":
    main()