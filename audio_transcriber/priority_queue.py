#!/usr/bin/env python3
"""
Priority queue management per doctrine
"""

import heapq
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional
from enum import IntEnum


class Priority(IntEnum):
    """Priority levels per doctrine"""
    EMERGENCY = 0  # Live legislative session
    URGENT = 1     # Time-sensitive opposition research
    NORMAL = 2     # Regular meeting minutes
    BATCH = 3      # Historical archive processing
    CITIZEN = 4    # Public submissions


@dataclass
class ResourceAllocation:
    """Resource allocation based on priority"""
    gpu_memory_gb: int
    cpu_cores: int
    timeout_minutes: Optional[int]
    

@dataclass
class TranscriptionJob:
    """Job with priority and metadata"""
    id: str
    audio_url: str
    priority: Priority
    source_metadata: Dict
    processing_options: Dict
    submitted_at: datetime
    webhook_url: Optional[str] = None
    
    def __lt__(self, other):
        """For heap comparison - lower priority value = higher priority"""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.submitted_at < other.submitted_at


class PriorityJobQueue:
    """
    Priority-based job scheduling per doctrine
    """
    
    def __init__(self):
        self.queue = []
        self.current_job = None
        self.suspended_jobs = []
        
    def enqueue(self, job: TranscriptionJob):
        """Add job to priority queue"""
        # Emergency jobs preempt current processing
        if job.priority == Priority.EMERGENCY and self.current_job and self.current_job.priority > Priority.EMERGENCY:
            self.suspend_current()
            
        heapq.heappush(self.queue, job)
        
    def dequeue(self) -> Optional[TranscriptionJob]:
        """Get highest priority job"""
        if self.queue:
            return heapq.heappop(self.queue)
        return None
        
    def suspend_current(self):
        """Suspend current job for emergency"""
        if self.current_job:
            self.suspended_jobs.append(self.current_job)
            self.current_job = None
            
    def get_resource_allocation(self, job: TranscriptionJob) -> ResourceAllocation:
        """Allocate resources based on priority"""
        allocations = {
            Priority.EMERGENCY: ResourceAllocation(
                gpu_memory_gb=24,
                cpu_cores=16,
                timeout_minutes=None
            ),
            Priority.URGENT: ResourceAllocation(
                gpu_memory_gb=16,
                cpu_cores=8,
                timeout_minutes=120
            ),
            Priority.NORMAL: ResourceAllocation(
                gpu_memory_gb=12,
                cpu_cores=4,
                timeout_minutes=240
            ),
            Priority.BATCH: ResourceAllocation(
                gpu_memory_gb=8,
                cpu_cores=2,
                timeout_minutes=480
            ),
            Priority.CITIZEN: ResourceAllocation(
                gpu_memory_gb=8,
                cpu_cores=2,
                timeout_minutes=360
            )
        }
        
        return allocations.get(job.priority, allocations[Priority.NORMAL])
        
    def get_queue_depth(self) -> Dict[str, int]:
        """Get queue depth by priority"""
        depths = {p.name: 0 for p in Priority}
        for job in self.queue:
            depths[job.priority.name] += 1
        return depths