#!/usr/bin/env python3
"""
Doctrine-compliant output schema
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import hashlib
import json


@dataclass
class ProcessingMetadata:
    """Processing details per doctrine"""
    pipeline_version: str = "1.0.0"
    models_used: Dict[str, str] = None
    processing_start: str = None
    processing_end: str = None
    total_duration_seconds: float = 0
    processing_speed: float = 0
    cost_estimate: float = 0
    priority_level: str = "normal"
    retry_count: int = 0
    
    def __post_init__(self):
        if self.models_used is None:
            self.models_used = {}
        if self.processing_start is None:
            self.processing_start = datetime.now().isoformat()


@dataclass
class SourceMetadata:
    """Source file metadata per doctrine"""
    file_name: str
    file_size_bytes: int
    duration_seconds: float
    format: str
    channels: int = 1
    sample_rate: int = 16000
    classification: str = "unclassified"
    project_code: Optional[str] = None
    session_date: Optional[str] = None
    location: Optional[str] = None


@dataclass
class QualityMetrics:
    """Quality assessment per doctrine"""
    overall_confidence: float
    speaker_confidence: float
    audio_quality_score: float
    background_noise_level: str
    cross_talk_detected: bool = False
    silence_percentage: float = 0.0


@dataclass
class ExtractedEntities:
    """Legislative entities per doctrine"""
    bills_mentioned: List[str] = None
    legislators_identified: List[Dict[str, str]] = None
    timestamps_referenced: List[str] = None
    organizations_mentioned: List[str] = None
    
    def __post_init__(self):
        if self.bills_mentioned is None:
            self.bills_mentioned = []
        if self.legislators_identified is None:
            self.legislators_identified = []
        if self.timestamps_referenced is None:
            self.timestamps_referenced = []
        if self.organizations_mentioned is None:
            self.organizations_mentioned = []


@dataclass
class ChainOfCustody:
    """Forensic chain of custody per doctrine"""
    created_at: str
    created_by: str = "system"
    last_accessed: str = None
    access_log: List[Dict[str, str]] = None
    retention_until: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.last_accessed is None:
            self.last_accessed = self.created_at
        if self.access_log is None:
            self.access_log = []


class DoctrineSchema:
    """Generate doctrine-compliant output"""
    
    @staticmethod
    def create_job_id(file_hash: str, timestamp: str) -> str:
        """Generate doctrine-compliant job ID"""
        # Format: YYYYMMDD_HHMMSS_<hash8>
        dt = datetime.fromisoformat(timestamp)
        hash_prefix = file_hash[:8]
        return f"{dt.strftime('%Y%m%d_%H%M%S')}_{hash_prefix}"
    
    @staticmethod
    def calculate_transcript_hash(segments: List[Dict]) -> str:
        """Calculate hash of transcript content"""
        transcript_text = " ".join([s.get('text', '') for s in segments])
        return hashlib.sha256(transcript_text.encode()).hexdigest()
    
    @staticmethod
    def estimate_cost(duration_seconds: float, priority: str = "normal") -> float:
        """Estimate processing cost"""
        base_rate = 0.001  # $0.001 per second
        priority_multipliers = {
            "emergency": 5.0,
            "urgent": 2.0,
            "normal": 1.0,
            "batch": 0.5,
            "citizen": 0.3
        }
        multiplier = priority_multipliers.get(priority, 1.0)
        return round(duration_seconds * base_rate * multiplier, 4)
    
    @classmethod
    def create_output(cls, 
                     job_id: str,
                     source_hash: str,
                     segments: List[Dict],
                     source_info: Dict,
                     processing_info: Dict,
                     context: Optional[Any] = None) -> Dict:
        """Create doctrine-compliant output"""
        
        # Extract quality metrics from processing
        quality = QualityMetrics(
            overall_confidence=processing_info.get('confidence', 0.85),
            speaker_confidence=processing_info.get('speaker_confidence', 0.80),
            audio_quality_score=processing_info.get('audio_quality', 0.75),
            background_noise_level=processing_info.get('noise_level', 'low')
        )
        
        # Create metadata structures
        source_meta = SourceMetadata(
            file_name=source_info['file_name'],
            file_size_bytes=source_info.get('file_size', 0),
            duration_seconds=source_info['duration'],
            format=source_info.get('format', 'unknown'),
            classification=context.classification if context and hasattr(context, 'classification') else "unclassified"
        )
        
        processing_meta = ProcessingMetadata(
            models_used={
                "whisper": processing_info.get('whisper_model', 'large'),
                "pyannote": processing_info.get('pyannote_version', '3.1') if processing_info.get('diarization_enabled') else None
            },
            processing_start=processing_info.get('start_time'),
            processing_end=processing_info.get('end_time', datetime.now().isoformat()),
            total_duration_seconds=processing_info.get('processing_time', 0),
            processing_speed=processing_info.get('speed', 0),
            cost_estimate=cls.estimate_cost(source_info['duration'], processing_info.get('priority', 'normal')),
            priority_level=processing_info.get('priority', 'normal')
        )
        
        # Extract entities (placeholder - would need NLP)
        entities = ExtractedEntities()
        
        # Chain of custody
        custody = ChainOfCustody(
            created_at=processing_info.get('start_time', datetime.now().isoformat()),
            retention_until=(datetime.now().replace(year=datetime.now().year + 7)).isoformat()
        )
        
        # Build complete output
        return {
            "job_id": job_id,
            "source_hash": source_hash,
            "transcript_hash": cls.calculate_transcript_hash(segments),
            "processing_metadata": asdict(processing_meta),
            "source_metadata": asdict(source_meta),
            "chain_of_custody": asdict(custody),
            "transcript": {
                "segments": segments,
                "full_text": " ".join([s.get('text', '') for s in segments]),
                "speaker_count": len(set(s.get('speaker', 'Unknown') for s in segments))
            },
            "extracted_entities": asdict(entities),
            "quality_metrics": asdict(quality),
            "status": "completed",
            "created_at": custody.created_at,
            "updated_at": datetime.now().isoformat()
        }