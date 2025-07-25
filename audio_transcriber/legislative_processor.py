#!/usr/bin/env python3
"""
Legislative audio optimization per doctrine
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class LegislativeMarker:
    """Legislative event marker"""
    type: str  # roll_call, gavel, recess, etc.
    timestamp: float
    confidence: float
    text: Optional[str] = None


class LegislativeProcessor:
    """
    Special handling for government audio per doctrine
    """
    
    # Legislative keywords and patterns
    ROLL_CALL_PATTERNS = [
        r"roll\s*call",
        r"calling\s*the\s*roll",
        r"record\s*the\s*vote",
        r"all\s*in\s*favor"
    ]
    
    GAVEL_PATTERNS = [
        r"order\s*order",
        r"come\s*to\s*order",
        r"meeting\s*adjourned",
        r"recess"
    ]
    
    BILL_PATTERN = r"\b(SB|HB|SR|HR)\d{2,4}-\d{2,4}\b"
    
    def __init__(self):
        self.known_legislators = self._load_known_legislators()
        
    def _load_known_legislators(self) -> Dict[str, List[str]]:
        """Load known legislator names by state"""
        # In production, load from database
        return {
            "CO": [
                "Sen. Bridges", "Sen. Rodriguez", "Sen. Zenzinger",
                "Rep. Sirota", "Rep. McCluskie", "Rep. Weissman"
            ]
        }
        
    def process_legislative_audio(self, audio_path: str, state: str = "CO") -> Dict:
        """Process with legislative optimizations"""
        # This would integrate with the main transcription pipeline
        return {
            "strategy": "legislative_optimized",
            "voice_prints": self._get_voice_prints(state),
            "expected_markers": ["roll_call", "gavel", "recess"],
            "bill_detection": True
        }
        
    def extract_legislative_markers(self, segments: List[Dict]) -> List[LegislativeMarker]:
        """Extract legislative events from transcript"""
        markers = []
        
        for segment in segments:
            text = segment.get('text', '').lower()
            
            # Check for roll calls
            for pattern in self.ROLL_CALL_PATTERNS:
                if re.search(pattern, text):
                    markers.append(LegislativeMarker(
                        type="roll_call",
                        timestamp=segment['start'],
                        confidence=0.9,
                        text=segment['text']
                    ))
                    
            # Check for gavels/order
            for pattern in self.GAVEL_PATTERNS:
                if re.search(pattern, text):
                    markers.append(LegislativeMarker(
                        type="gavel",
                        timestamp=segment['start'],
                        confidence=0.85,
                        text=segment['text']
                    ))
                    
        return markers
        
    def extract_bills_mentioned(self, segments: List[Dict]) -> List[str]:
        """Extract bill numbers from transcript"""
        bills = set()
        
        for segment in segments:
            text = segment.get('text', '')
            matches = re.findall(self.BILL_PATTERN, text, re.IGNORECASE)
            bills.update(matches)
            
        return sorted(list(bills))
        
    def identify_legislators(self, segments: List[Dict], state: str = "CO") -> List[Dict[str, str]]:
        """Identify legislators mentioned in transcript"""
        legislators = []
        known_names = self.known_legislators.get(state, [])
        
        for segment in segments:
            text = segment.get('text', '')
            speaker = segment.get('speaker', 'Unknown')
            
            # Check if speaker matches known legislator
            for name in known_names:
                if name.lower() in text.lower():
                    legislators.append({
                        "name": name,
                        "speaker_id": speaker,
                        "timestamp": segment['start'],
                        "context": text[:100]
                    })
                    
        return legislators
        
    def segment_by_legislative_markers(self, audio_path: str) -> List[Tuple[float, float]]:
        """Segment audio by legislative events"""
        # In production, this would use audio analysis to detect gavels, etc.
        # For now, return standard 30-minute chunks
        segments = []
        chunk_duration = 1800  # 30 minutes
        
        # Placeholder - would analyze audio for natural breaks
        for start in range(0, 18000, chunk_duration):  # up to 5 hours
            segments.append((start, start + chunk_duration))
            
        return segments
        
    def _get_voice_prints(self, state: str) -> Dict[str, np.ndarray]:
        """Get voice prints for known legislators"""
        # In production, load from voice database
        # Returns placeholder embeddings
        voice_prints = {}
        
        for legislator in self.known_legislators.get(state, []):
            # Placeholder 256-dim embedding
            voice_prints[legislator] = np.random.randn(256)
            
        return voice_prints
        
    def match_speakers_to_legislators(self, 
                                    speaker_embeddings: Dict[str, np.ndarray],
                                    voice_prints: Dict[str, np.ndarray],
                                    threshold: float = 0.8) -> Dict[str, str]:
        """Match detected speakers to known legislators"""
        speaker_mapping = {}
        
        for speaker_id, embedding in speaker_embeddings.items():
            best_match = None
            best_score = 0.0
            
            for legislator, voice_print in voice_prints.items():
                # Cosine similarity
                score = np.dot(embedding, voice_print) / (
                    np.linalg.norm(embedding) * np.linalg.norm(voice_print)
                )
                
                if score > best_score and score > threshold:
                    best_match = legislator
                    best_score = score
                    
            if best_match:
                speaker_mapping[speaker_id] = best_match
            else:
                speaker_mapping[speaker_id] = f"Unknown_{speaker_id}"
                
        return speaker_mapping