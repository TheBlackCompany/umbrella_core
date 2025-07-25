#!/usr/bin/env python3
"""
Umbrella Transcriber Engine
Core transcription functionality - doctrine compliant
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Optional, Union, List
from datetime import datetime

from transcribe import UmbrellaTranscriber
from process_folder import FolderProcessor
from context_system import ConversationContext
from schema import DoctrineSchema
from legislative_processor import LegislativeProcessor
from priority_queue import Priority, TranscriptionJob


class TranscriptionEngine:
    """
    Core engine for transcription
    Doctrine: Forensic-grade, deterministic, schema-compliant output
    """
    
    def __init__(self, model_size: str = "large", use_gpu: bool = True,
                 enable_diarization: bool = True, verbose: bool = False):
        """Initialize engine with configuration"""
        self.transcriber = UmbrellaTranscriber(
            model_size=model_size,
            use_gpu=use_gpu,
            enable_diarization=enable_diarization,
            verbose=verbose
        )
        self.transcriber.setup()
        
    def process(self, input_path: Path, output_dir: Optional[Path] = None,
                context: Optional[ConversationContext] = None,
                expected_speakers: Optional[int] = None) -> Dict:
        """
        Process audio file or folder
        Returns: Structured output conforming to doctrine
        """
        
        # Update transcriber context
        if context:
            self.transcriber.context = context
        if expected_speakers:
            self.transcriber.expected_speakers = expected_speakers
            
        # Process based on input type
        if input_path.is_file():
            return self._process_file(input_path, output_dir)
        elif input_path.is_dir():
            return self._process_folder(input_path, output_dir)
        else:
            raise ValueError(f"Invalid input path: {input_path}")
            
    def _process_file(self, file_path: Path, output_dir: Optional[Path]) -> Dict:
        """Process single audio file with doctrine compliance"""
        # Track processing start
        start_time = datetime.now()
        
        # Calculate source hash
        with open(file_path, 'rb') as f:
            source_hash = hashlib.sha256(f.read()).hexdigest()
            
        # Get file info
        file_stat = file_path.stat()
        
        # Process with transcriber
        output_path, manifest = self.transcriber.transcribe_file(file_path, output_dir)
        
        # Load outputs
        with open(output_path / "manifest.json") as f:
            manifest_data = json.load(f)
            
        with open(output_path / "transcript" / "segments.json") as f:
            segments_data = json.load(f)
            
        # Extract segments
        segments = segments_data.get('segments', [])
        
        # Apply legislative processing if needed
        source_type = self.transcriber.context.type.value if self.transcriber.context else "unknown"
        if source_type == "legislative":
            processor = LegislativeProcessor()
            markers = processor.extract_legislative_markers(segments)
            bills = processor.extract_bills_mentioned(segments)
            legislators = processor.identify_legislators(segments)
        else:
            markers = []
            bills = []
            legislators = []
            
        # Create job ID
        job_id = DoctrineSchema.create_job_id(source_hash, start_time.isoformat())
        
        # Build processing info
        processing_info = {
            'whisper_model': self.transcriber.model_size,
            'diarization_enabled': self.transcriber.enable_diarization,
            'start_time': start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'processing_time': manifest_data.get('transcription_time', 0) + manifest_data.get('diarization_time', 0),
            'speed': manifest_data.get('total_speed', 0),
            'priority': 'normal'
        }
        
        # Build source info
        source_info = {
            'file_name': file_path.name,
            'file_size': file_stat.st_size,
            'duration': manifest_data.get('duration_seconds', 0),
            'format': file_path.suffix[1:]  # Remove dot
        }
        
        # Create doctrine-compliant output
        doctrine_output = DoctrineSchema.create_output(
            job_id=job_id,
            source_hash=source_hash,
            segments=segments,
            source_info=source_info,
            processing_info=processing_info,
            context=self.transcriber.context
        )
        
        # Add legislative entities if found
        if bills or legislators or markers:
            doctrine_output['extracted_entities']['bills_mentioned'] = bills
            doctrine_output['extracted_entities']['legislators_identified'] = legislators
            doctrine_output['extracted_entities']['timestamps_extracted'] = {
                'markers': [{'type': m.type, 'time': m.timestamp} for m in markers]
            }
            
        # Save doctrine output
        doctrine_path = output_path / "doctrine_output.json"
        with open(doctrine_path, 'w') as f:
            json.dump(doctrine_output, f, indent=2)
            
        return doctrine_output
        
    def _process_folder(self, folder_path: Path, output_dir: Optional[Path]) -> Dict:
        """Process folder of audio files"""
        processor = FolderProcessor(
            folder_path=folder_path,
            output_name=output_dir.name if output_dir else None,
            context=self.transcriber.context
        )
        
        output_path = processor.process_folder(self.transcriber, interactive_select=False)
        
        # Load combined output
        with open(output_path / "manifest.json") as f:
            manifest_data = json.load(f)
            
        with open(output_path / "combined_data.json") as f:
            combined_data = json.load(f)
            
        return {
            "type": "folder",
            "input": str(folder_path),
            "output_dir": str(output_path),
            "manifest": manifest_data,
            "combined_data": combined_data,
            "status": "complete"
        }
        
    def process_batch(self, file_list: List[Path], output_dir: Path) -> Dict:
        """
        Process multiple files as separate transcriptions
        For machine/workflow usage
        """
        results = []
        
        for file_path in file_list:
            try:
                result = self.process(file_path, output_dir / file_path.stem)
                results.append(result)
            except Exception as e:
                results.append({
                    "type": "error",
                    "input": str(file_path),
                    "error": str(e),
                    "status": "failed"
                })
                
        return {
            "type": "batch",
            "processed": len([r for r in results if r["status"] == "complete"]),
            "failed": len([r for r in results if r["status"] == "failed"]),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }