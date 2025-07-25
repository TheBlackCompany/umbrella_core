"""
Base schemas for Umbrella Core processing tools
Shared data structures across all services
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


class ProcessingPriority(Enum):
    """Standard priority levels across all services"""
    EMERGENCY = 0  # Immediate processing required
    URGENT = 1     # High priority
    NORMAL = 2     # Standard processing
    BATCH = 3      # Low priority batch processing
    CITIZEN = 4    # Citizen submissions


@dataclass
class JobMetadata:
    """Standard job metadata for all processing tasks"""
    job_id: str
    service: str
    priority: ProcessingPriority
    submitted_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"
    progress: float = 0.0
    error: Optional[str] = None


@dataclass
class ChainOfCustody:
    """Forensic chain of custody for legal compliance"""
    created_at: str
    created_by: str
    last_accessed: str
    access_log: List[Dict[str, str]]
    retention_until: str
    hash: str


@dataclass
class ProcessingResult:
    """Standard result format for all services"""
    job_id: str
    service: str
    version: str
    input_hash: str
    output_hash: str
    metadata: Dict[str, Any]
    result: Dict[str, Any]
    chain_of_custody: ChainOfCustody
    processing_time_seconds: float
    cost_estimate: Optional[float] = None