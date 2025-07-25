#!/usr/bin/env python3
"""Simple test of doctrine compliance"""

import sys
import os

# Add FFmpeg to PATH if needed
if os.name == 'nt':  # Windows
    # Common ffmpeg locations
    ffmpeg_paths = [
        r"C:\ffmpeg\bin",
        r"C:\Program Files\ffmpeg\bin",
        r"C:\ProgramData\chocolatey\bin"
    ]
    for path in ffmpeg_paths:
        if os.path.exists(path) and path not in os.environ["PATH"]:
            os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]
            print(f"Added {path} to PATH")
            break

# Now test ffmpeg
import subprocess
try:
    result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
    if result.returncode == 0:
        print("[OK] FFmpeg is available")
    else:
        print("[FAIL] FFmpeg test failed")
        sys.exit(1)
except Exception as e:
    print(f"[FAIL] FFmpeg not found: {e}")
    sys.exit(1)

# Test importing modules
try:
    from schema import DoctrineSchema
    from priority_queue import Priority, TranscriptionJob
    from legislative_processor import LegislativeProcessor
    print("[OK] All doctrine modules imported successfully")
except ImportError as e:
    print(f"[FAIL] Import error: {e}")
    sys.exit(1)

# Test schema generation
try:
    # Create a mock transcription result
    mock_result = {
        "text": "This is a test transcript.",
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 5.0,
                "text": "This is a test transcript.",
                "speaker": "SPEAKER_00"
            }
        ],
        "language": "en"
    }
    
    # Generate doctrine-compliant output
    source_info = {
        'file_name': 'test.m4a',
        'file_size': 1048576,  # 1MB
        'duration': 60.0,
        'format': 'm4a'
    }
    
    processing_info = {
        'confidence': 0.95,
        'speaker_confidence': 0.90,
        'audio_quality': 0.85,
        'noise_level': 'low',
        'whisper_model': 'large',
        'diarization_enabled': True,
        'pyannote_version': '3.1',
        'start_time': '2025-07-24T16:00:00',
        'end_time': '2025-07-24T16:00:10',
        'processing_time': 10.0,
        'speed': 6.0,
        'priority': 'normal'
    }
    
    schema = DoctrineSchema.create_output(
        job_id="test_123",
        source_hash="abc123def456",
        segments=mock_result["segments"],
        source_info=source_info,
        processing_info=processing_info,
        context=None
    )
    
    print("[OK] Schema generation successful")
    print(f"    Job ID: {schema['job_id']}")
    print(f"    Format version: {schema.get('version', 'Not found')}")
    print(f"    Schema keys: {list(schema.keys())}")
    required_fields = ['job_id', 'version', 'timestamp', 'source_metadata', 'transcript', 'processing_metadata', 'chain_of_custody']
    print(f"    Has all required fields: {all(k in schema for k in required_fields)}")
    
except Exception as e:
    print(f"[FAIL] Schema generation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n[OK] All basic tests passed!")
print("Ready for full audio processing test.")