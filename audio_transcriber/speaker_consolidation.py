#!/usr/bin/env python3
"""
Speaker consolidation post-processing
Merges over-segmented speakers based on context and voice similarity
"""

from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
import numpy as np
from dataclasses import dataclass
import json

from context_system import ConversationType, ConversationContext


@dataclass
class SpeakerStats:
    """Statistics for a speaker"""
    speaker_id: str
    segment_count: int
    total_duration: float
    avg_segment_duration: float
    first_appearance: float
    last_appearance: float
    

class SpeakerConsolidator:
    """Consolidate over-segmented speakers based on context"""
    
    def __init__(self, context: Optional[ConversationContext] = None):
        self.context = context
        
    def consolidate_speakers(self, segments: List[Dict], 
                           expected_speakers: Optional[int] = None) -> Tuple[List[Dict], Dict[str, str]]:
        """
        Consolidate speakers in segments based on context
        Returns: (consolidated_segments, speaker_mapping)
        """
        # Get expected speaker count
        if expected_speakers is None and self.context:
            expected_speakers = self.context.expected_speakers
            
        # Analyze current speakers
        speaker_stats = self._analyze_speakers(segments)
        
        # Determine consolidation strategy
        if self.context and self.context.type == ConversationType.PHONE_CALL:
            # Phone calls should have exactly 2 speakers
            return self._consolidate_phone_call(segments, speaker_stats)
        elif expected_speakers and len(speaker_stats) > expected_speakers:
            # Too many speakers detected
            return self._consolidate_to_expected(segments, speaker_stats, expected_speakers)
        else:
            # General consolidation - merge minor speakers
            return self._consolidate_minor_speakers(segments, speaker_stats)
            
    def _analyze_speakers(self, segments: List[Dict]) -> Dict[str, SpeakerStats]:
        """Analyze speaker statistics"""
        stats = defaultdict(lambda: {
            'count': 0, 
            'duration': 0.0, 
            'first': float('inf'), 
            'last': 0.0
        })
        
        for seg in segments:
            speaker = seg.get('speaker', 'Unknown')
            duration = seg['end'] - seg['start']
            
            stats[speaker]['count'] += 1
            stats[speaker]['duration'] += duration
            stats[speaker]['first'] = min(stats[speaker]['first'], seg['start'])
            stats[speaker]['last'] = max(stats[speaker]['last'], seg['end'])
            
        # Convert to SpeakerStats objects
        speaker_stats = {}
        for speaker, data in stats.items():
            speaker_stats[speaker] = SpeakerStats(
                speaker_id=speaker,
                segment_count=data['count'],
                total_duration=data['duration'],
                avg_segment_duration=data['duration'] / data['count'] if data['count'] > 0 else 0,
                first_appearance=data['first'],
                last_appearance=data['last']
            )
            
        return speaker_stats
        
    def _consolidate_phone_call(self, segments: List[Dict], 
                               speaker_stats: Dict[str, SpeakerStats]) -> Tuple[List[Dict], Dict[str, str]]:
        """Consolidate speakers for phone call (exactly 2 speakers)"""
        # Sort speakers by total speaking time
        sorted_speakers = sorted(
            speaker_stats.items(), 
            key=lambda x: x[1].total_duration, 
            reverse=True
        )
        
        # Map to Speaker_A and Speaker_B
        mapping = {}
        
        if len(sorted_speakers) >= 1:
            mapping[sorted_speakers[0][0]] = "Speaker_A"
        if len(sorted_speakers) >= 2:
            mapping[sorted_speakers[1][0]] = "Speaker_B"
            
        # Map all others to the nearest main speaker
        main_speakers = [sorted_speakers[0][0], sorted_speakers[1][0]] if len(sorted_speakers) >= 2 else [sorted_speakers[0][0]]
        
        for speaker, stats in sorted_speakers[2:]:
            # Map minor speakers to nearest main speaker by time proximity
            mapping[speaker] = self._find_nearest_speaker(segments, speaker, main_speakers, mapping)
            
        # Handle Unknown speaker
        if 'Unknown' in speaker_stats and 'Unknown' not in mapping:
            mapping['Unknown'] = self._find_nearest_speaker(segments, 'Unknown', main_speakers, mapping)
            
        # Apply mapping
        consolidated_segments = self._apply_speaker_mapping(segments, mapping)
        
        return consolidated_segments, mapping
        
    def _consolidate_to_expected(self, segments: List[Dict], 
                                speaker_stats: Dict[str, SpeakerStats],
                                expected_speakers: int) -> Tuple[List[Dict], Dict[str, str]]:
        """Consolidate to expected number of speakers"""
        # Sort speakers by total speaking time
        sorted_speakers = sorted(
            speaker_stats.items(), 
            key=lambda x: x[1].total_duration, 
            reverse=True
        )
        
        # Keep top N speakers
        mapping = {}
        main_speakers = []
        
        for i, (speaker, stats) in enumerate(sorted_speakers[:expected_speakers]):
            new_id = f"Speaker_{i+1:02d}"
            mapping[speaker] = new_id
            main_speakers.append(speaker)
            
        # Map remaining speakers to nearest main speaker
        for speaker, stats in sorted_speakers[expected_speakers:]:
            mapping[speaker] = self._find_nearest_speaker(segments, speaker, main_speakers, mapping)
            
        # Handle Unknown
        if 'Unknown' in speaker_stats and 'Unknown' not in mapping:
            mapping['Unknown'] = self._find_nearest_speaker(segments, 'Unknown', main_speakers, mapping)
            
        # Apply mapping
        consolidated_segments = self._apply_speaker_mapping(segments, mapping)
        
        return consolidated_segments, mapping
        
    def _consolidate_minor_speakers(self, segments: List[Dict], 
                                   speaker_stats: Dict[str, SpeakerStats]) -> Tuple[List[Dict], Dict[str, str]]:
        """Consolidate minor speakers (less than 5% of total speech)"""
        total_duration = sum(s.total_duration for s in speaker_stats.values())
        
        # Identify minor speakers
        minor_threshold = 0.05  # 5% of total speech
        major_speakers = []
        minor_speakers = []
        
        for speaker, stats in speaker_stats.items():
            if stats.total_duration / total_duration >= minor_threshold:
                major_speakers.append(speaker)
            else:
                minor_speakers.append(speaker)
                
        # Create mapping
        mapping = {speaker: speaker for speaker in major_speakers}
        
        # Map minor speakers to nearest major speaker
        for speaker in minor_speakers:
            mapping[speaker] = self._find_nearest_speaker(segments, speaker, major_speakers, {})
            
        # Apply mapping
        consolidated_segments = self._apply_speaker_mapping(segments, mapping)
        
        return consolidated_segments, mapping
        
    def _find_nearest_speaker(self, segments: List[Dict], target_speaker: str, 
                             candidate_speakers: List[str], existing_mapping: Dict[str, str]) -> str:
        """Find the nearest speaker based on temporal proximity"""
        if not candidate_speakers:
            return target_speaker
            
        # Get all segments for target speaker
        target_segments = [s for s in segments if s.get('speaker') == target_speaker]
        if not target_segments:
            return candidate_speakers[0]
            
        # Calculate proximity scores for each candidate
        proximity_scores = defaultdict(float)
        
        for target_seg in target_segments:
            target_mid = (target_seg['start'] + target_seg['end']) / 2
            
            # Find nearest segment from each candidate
            for candidate in candidate_speakers:
                min_distance = float('inf')
                
                for seg in segments:
                    if seg.get('speaker') == candidate:
                        seg_mid = (seg['start'] + seg['end']) / 2
                        distance = abs(seg_mid - target_mid)
                        min_distance = min(min_distance, distance)
                        
                # Weight by inverse distance
                if min_distance < float('inf'):
                    proximity_scores[candidate] += 1.0 / (1.0 + min_distance)
                    
        # Return candidate with highest proximity score
        if proximity_scores:
            best_candidate = max(proximity_scores.items(), key=lambda x: x[1])[0]
            # Map to final speaker ID if already mapped
            return existing_mapping.get(best_candidate, best_candidate)
        else:
            return candidate_speakers[0]
            
    def _apply_speaker_mapping(self, segments: List[Dict], mapping: Dict[str, str]) -> List[Dict]:
        """Apply speaker mapping to segments"""
        consolidated = []
        
        for seg in segments:
            new_seg = seg.copy()
            old_speaker = seg.get('speaker', 'Unknown')
            new_seg['speaker'] = mapping.get(old_speaker, old_speaker)
            new_seg['original_speaker'] = old_speaker  # Keep original for reference
            consolidated.append(new_seg)
            
        return consolidated


def consolidate_transcript_file(transcript_path: str, context: Optional[ConversationContext] = None,
                               expected_speakers: Optional[int] = None) -> Dict[str, any]:
    """Consolidate speakers in a transcript file"""
    # Load segments
    with open(transcript_path, 'r') as f:
        data = json.load(f)
        
    segments = data.get('segments', [])
    
    # Consolidate
    consolidator = SpeakerConsolidator(context)
    consolidated_segments, mapping = consolidator.consolidate_speakers(segments, expected_speakers)
    
    # Update data
    data['segments'] = consolidated_segments
    data['speaker_consolidation'] = {
        'performed': True,
        'mapping': mapping,
        'original_speaker_count': len(set(s.get('speaker', 'Unknown') for s in segments)),
        'consolidated_speaker_count': len(set(s.get('speaker', 'Unknown') for s in consolidated_segments))
    }
    
    return data


def update_transcript_text(segments: List[Dict], output_path: str, metadata: Dict):
    """Regenerate full transcript text with consolidated speakers"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("UMBRELLA PROJECT TRANSCRIPT\n")
        f.write("=" * 80 + "\n")
        f.write(f"File: {metadata.get('file_name', 'Unknown')}\n")
        f.write(f"Date: {metadata.get('timestamp', 'Unknown')}\n")
        f.write(f"Duration: {metadata.get('duration_seconds', 0)} seconds\n")
        f.write(f"Language: {metadata.get('language', 'Unknown').upper()}\n")
        
        # Show consolidation info
        consolidation = metadata.get('speaker_consolidation', {})
        if consolidation.get('performed'):
            f.write(f"Speakers: {consolidation['consolidated_speaker_count']} ")
            f.write(f"(consolidated from {consolidation['original_speaker_count']})\n")
        else:
            f.write(f"Speakers: {metadata.get('speaker_count', 'Unknown')}\n")
            
        f.write(f"Processing Speed: {metadata.get('total_speed', 0):.1f}x realtime\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("TRANSCRIPT WITH SPEAKERS\n")
        f.write("-" * 40 + "\n\n")
        
        current_speaker = None
        for seg in segments:
            speaker = seg.get('speaker', 'Unknown')
            
            # Add speaker label if changed
            if speaker != current_speaker:
                f.write(f"\n[{speaker}]\n")
                current_speaker = speaker
                
            # Format timestamp
            start_min = int(seg['start'] // 60)
            start_sec = int(seg['start'] % 60)
            timestamp = f"[{start_min:02d}:{start_sec:02d}]"
            
            # Write text
            text = seg.get('text', '').strip()
            if text:
                f.write(f"{timestamp} {text}\n")