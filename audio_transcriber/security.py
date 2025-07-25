#!/usr/bin/env python3
"""
Security and compliance per doctrine
"""

from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import hashlib
import json


class Classification(Enum):
    """Data classification levels"""
    PUBLIC = "public"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"


class ClearanceLevel(Enum):
    """User clearance levels"""
    CITIZEN = 0
    STAFF = 1
    ANALYST = 2
    EXECUTIVE = 3


@dataclass
class User:
    """User with clearance"""
    id: str
    name: str
    clearance_level: ClearanceLevel
    roles: List[str]
    

@dataclass
class AccessLog:
    """Access audit log entry"""
    user_id: str
    job_id: str
    action: str
    timestamp: datetime
    ip_address: Optional[str] = None
    

class SecurityManager:
    """Handle access control and compliance"""
    
    # Access matrix
    ACCESS_LEVELS = {
        Classification.PUBLIC: [ClearanceLevel.CITIZEN, ClearanceLevel.STAFF, 
                               ClearanceLevel.ANALYST, ClearanceLevel.EXECUTIVE],
        Classification.CONFIDENTIAL: [ClearanceLevel.ANALYST, ClearanceLevel.EXECUTIVE],
        Classification.SECRET: [ClearanceLevel.EXECUTIVE]
    }
    
    # Retention policies per doctrine
    RETENTION_POLICIES = {
        "legislative_hearing": "permanent",
        "citizen_submission": timedelta(days=7*365),  # 7 years
        "internal_comm": timedelta(days=90),
        "phone_call": timedelta(days=365),
        "default": timedelta(days=365)
    }
    
    def __init__(self):
        self.access_logs = []
        
    def validate_access(self, user: User, job_metadata: Dict) -> bool:
        """Ensure user can access transcript based on classification"""
        classification = Classification(job_metadata.get('classification', 'public'))
        required_clearance = self.ACCESS_LEVELS.get(classification, [])
        
        return user.clearance_level in required_clearance
        
    def log_access(self, user_id: str, job_id: str, action: str, 
                   ip_address: Optional[str] = None):
        """Log access for audit trail"""
        log_entry = AccessLog(
            user_id=user_id,
            job_id=job_id,
            action=action,
            timestamp=datetime.now(),
            ip_address=ip_address
        )
        
        self.access_logs.append(log_entry)
        
        # In production, persist to database
        return log_entry
        
    def get_retention_date(self, source_type: str) -> Optional[datetime]:
        """Calculate retention date based on source type"""
        policy = self.RETENTION_POLICIES.get(source_type, self.RETENTION_POLICIES["default"])
        
        if policy == "permanent":
            return None
            
        return datetime.now() + policy
        
    def sanitize_output(self, transcript: Dict, user: User) -> Dict:
        """Remove sensitive data based on user clearance"""
        # Create a copy
        sanitized = transcript.copy()
        
        # Remove sensitive fields for lower clearance
        if user.clearance_level < ClearanceLevel.ANALYST:
            # Remove internal metadata
            if 'processing_metadata' in sanitized:
                sanitized['processing_metadata'].pop('cost_estimate_usd', None)
                sanitized['processing_metadata'].pop('processor_id', None)
                
        if user.clearance_level < ClearanceLevel.EXECUTIVE:
            # Remove chain of custody details
            if 'chain_of_custody' in sanitized:
                sanitized['chain_of_custody']['access_log'] = []
                
        return sanitized
        
    def generate_access_token(self, user: User, job_id: str, 
                            expiry_hours: int = 24) -> str:
        """Generate time-limited access token"""
        # In production, use JWT
        token_data = {
            "user_id": user.id,
            "job_id": job_id,
            "expires": (datetime.now() + timedelta(hours=expiry_hours)).isoformat(),
            "clearance": user.clearance_level.name
        }
        
        # Simple hash for demo
        token_string = json.dumps(token_data, sort_keys=True)
        return hashlib.sha256(token_string.encode()).hexdigest()
        
    def validate_token(self, token: str) -> Optional[Dict]:
        """Validate access token"""
        # In production, validate JWT
        # For now, just check if token exists
        return {"valid": True} if token else None
        
    def get_audit_trail(self, job_id: str) -> List[Dict]:
        """Get access history for a job"""
        trail = []
        
        for log in self.access_logs:
            if log.job_id == job_id:
                trail.append({
                    "user_id": log.user_id,
                    "action": log.action,
                    "timestamp": log.timestamp.isoformat(),
                    "ip_address": log.ip_address
                })
                
        return trail