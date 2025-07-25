"""
Chunked processing strategy for long audio files
Processes in overlapping chunks with speaker tracking
"""

import os
import json
import time
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import whisper
import soundfile as sf
from collections import defaultdict

from .base import ProcessingStrategy

class SpeakerTracker:
    """Track speaker identity across chunks using embeddings"""
    
    def __init__(self):
        self.speaker_embeddings = {}  # chunk_speaker -> embedding
        self.global_speakers = {}  # global_speaker -> [chunk_speakers]
        self.next_global_id = 0
        
    def add_chunk_speakers(self, chunk_id: int, speakers: Dict[str, np.ndarray]):
        """Add speakers from a chunk with their embeddings"""
        for local_speaker, embedding in speakers.items():
            chunk_speaker = f"chunk{chunk_id}_{local_speaker}"
            self.speaker_embeddings[chunk_speaker] = embedding
            
            # Find best matching global speaker
            best_match = None
            best_similarity = 0.7  # Threshold for matching
            
            for global_speaker, chunk_speakers in self.global_speakers.items():
                # Compare with embeddings from this global speaker
                for cs in chunk_speakers:
                    if cs in self.speaker_embeddings:
                        similarity = self._cosine_similarity(
                            embedding, 
                            self.speaker_embeddings[cs]
                        )
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = global_speaker
                            
            if best_match:
                self.global_speakers[best_match].append(chunk_speaker)
            else:
                # Create new global speaker
                global_id = f"SPEAKER_{self.next_global_id:02d}"
                self.next_global_id += 1
                self.global_speakers[global_id] = [chunk_speaker]
                
    def get_global_speaker(self, chunk_id: int, local_speaker: str) -> str:
        """Get global speaker ID for a chunk-local speaker"""
        chunk_speaker = f"chunk{chunk_id}_{local_speaker}"
        
        for global_speaker, chunk_speakers in self.global_speakers.items():
            if chunk_speaker in chunk_speakers:
                return global_speaker
                
        return "Unknown"
        
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class ChunkedStrategy(ProcessingStrategy):
    """Process long files in overlapping chunks"""
    
    def __init__(self, transcriber, verbose: bool = True, 
                 chunk_duration: int = 1200,  # 20 minutes
                 overlap_duration: int = 60):  # 1 minute overlap
        super().__init__(transcriber, verbose)
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        
    def can_handle(self, audio_path: Path, duration: float) -> bool:
        """Chunked strategy handles files over 30 minutes"""
        return duration > 30 * 60
        
    def process(self, audio_path: Path, output_dir: Path) -> Tuple[Path, Dict]:
        """Process audio in chunks with speaker tracking"""
        self.start_time = time.time()
        
        # Create output directories
        output_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ["audio", "transcript", "chunks", "status"]:
            (output_dir / subdir).mkdir(exist_ok=True)
            
        # Copy audio file
        audio_dest = output_dir / "audio" / audio_path.name
        shutil.copy2(str(audio_path), str(audio_dest))
        
        # Get audio info
        info = sf.info(audio_path)
        duration = info.duration
        sample_rate = info.samplerate
        
        self.log(f"Processing {duration/60:.1f} minute file in chunks", "INFO")
        self.log(f"Chunk size: {self.chunk_duration/60:.0f} minutes with {self.overlap_duration/60:.0f} minute overlap", "INFO")
        
        # Calculate chunks
        chunks = self._calculate_chunks(duration)
        self.log(f"Will process {len(chunks)} chunks", "INFO")
        
        # Process chunks
        all_segments = []
        speaker_tracker = SpeakerTracker()
        chunk_results = []
        
        for i, (start, end) in enumerate(chunks):
            self.log(f"\n[Chunk {i+1}/{len(chunks)}] {start/60:.1f}-{end/60:.1f} minutes", "PROCESS")
            
            # Extract chunk
            chunk_audio = self._extract_chunk(audio_path, start, end, sample_rate)
            
            # Save chunk for debugging
            chunk_path = output_dir / "chunks" / f"chunk_{i:03d}.wav"
            sf.write(chunk_path, chunk_audio, sample_rate)
            
            # Transcribe chunk
            chunk_segments = self._process_chunk(chunk_audio, i)
            
            # Diarize chunk if enabled
            if self.transcriber.enable_diarization and self.transcriber.diarization_pipeline:
                self.log(f"  Performing diarization for chunk {i+1}...", "PROCESS")
                chunk_speakers = self._diarize_chunk(chunk_path, chunk_segments, i)
                speaker_tracker.add_chunk_speakers(i, chunk_speakers)
                
                # Map segments to global speakers
                for seg in chunk_segments:
                    if 'speaker' in seg:
                        seg['global_speaker'] = speaker_tracker.get_global_speaker(i, seg['speaker'])
                        seg['speaker'] = seg['global_speaker']
            else:
                self.log(f"  Skipping diarization: enabled={self.transcriber.enable_diarization}, pipeline={self.transcriber.diarization_pipeline is not None}", "WARNING")
            
            # Adjust timestamps to global time
            for seg in chunk_segments:
                seg['start'] += start
                seg['end'] += start
                seg['chunk_id'] = i
                
            chunk_results.append({
                'chunk_id': i,
                'start': start,
                'end': end,
                'segments': len(chunk_segments)
            })
            
            all_segments.extend(chunk_segments)
            
            # Clear GPU memory
            if self.transcriber.device == "cuda":
                torch.cuda.empty_cache()
                
        # Remove duplicate segments from overlaps
        final_segments = self._merge_overlapping_segments(all_segments)
        
        # Generate transcript files
        self._save_transcripts(output_dir, final_segments, audio_path.name)
        
        # Create manifest
        speaker_count = len(speaker_tracker.global_speakers)
        manifest = self.create_manifest(
            audio_path, 
            final_segments,
            speaker_count,
            {
                "chunk_count": len(chunks),
                "chunk_duration": self.chunk_duration,
                "overlap_duration": self.overlap_duration,
                "chunk_results": chunk_results,
                "total_segments": len(final_segments)
            }
        )
        
        # Save manifest
        with open(output_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
            
        # Update status
        (output_dir / "status.txt").write_text("COMPLETE")
        
        total_time = time.time() - self.start_time
        self.log(f"Total processing time: {total_time/60:.1f} minutes ({duration/total_time:.1f}x realtime)", "SUCCESS")
        
        return output_dir, manifest
        
    def _calculate_chunks(self, duration: float) -> List[Tuple[float, float]]:
        """Calculate chunk boundaries with overlap"""
        chunks = []
        current = 0
        
        while current < duration:
            end = min(current + self.chunk_duration, duration)
            chunks.append((current, end))
            
            if end >= duration:
                break
                
            # Next chunk starts before current ends (overlap)
            current = end - self.overlap_duration
            
        return chunks
        
    def _extract_chunk(self, audio_path: Path, start: float, end: float, 
                      sample_rate: int) -> np.ndarray:
        """Extract audio chunk"""
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        
        chunk, _ = sf.read(
            audio_path,
            start=start_sample,
            stop=end_sample,
            dtype='float32',
            always_2d=True
        )
        
        # Convert to mono
        if chunk.shape[1] > 1:
            chunk = chunk.mean(axis=1)
        else:
            chunk = chunk[:, 0]
            
        return chunk
        
    def _process_chunk(self, audio_chunk: np.ndarray, chunk_id: int) -> List[Dict]:
        """Transcribe a single chunk"""
        # Use the transcriber's whisper model
        result = self.transcriber.model.transcribe(
            audio_chunk,
            language='en',
            task='transcribe',
            verbose=False,
            temperature=0.0,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=False  # Important for chunks
        )
        
        segments = []
        for seg in result['segments']:
            segments.append({
                'start': seg['start'],
                'end': seg['end'],
                'text': seg['text'],
                'language': result.get('language', 'en')
            })
            
        self.log(f"  Transcribed {len(segments)} segments", "SUCCESS")
        return segments
        
    def _diarize_chunk(self, chunk_path: Path, segments: List[Dict], 
                      chunk_id: int) -> Dict[str, np.ndarray]:
        """Diarize a chunk and return speaker embeddings"""
        self.log(f"  Starting diarization on chunk {chunk_id+1}", "INFO")
        
        # Apply speaker hints if available
        params = {}
        if self.transcriber.expected_speakers:
            params['min_speakers'] = max(2, self.transcriber.expected_speakers - 1)
            params['max_speakers'] = self.transcriber.expected_speakers + 1
            self.log(f"  Using speaker hints: {params['min_speakers']}-{params['max_speakers']} speakers", "INFO")
            
        # Run diarization
        self.log(f"  Running pyannote pipeline on {chunk_path.name}...", "INFO")
        diarization = self.transcriber.diarization_pipeline(str(chunk_path), **params)
        
        # Map segments to speakers
        for segment in segments:
            segment_mid = (segment['start'] + segment['end']) / 2
            segment['speaker'] = 'Unknown'
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if turn.start <= segment_mid <= turn.end:
                    segment['speaker'] = speaker
                    break
                    
        # Extract embeddings (placeholder - would need actual embedding extraction)
        embeddings = {}
        speakers = set(s['speaker'] for s in segments if s['speaker'] != 'Unknown')
        
        for speaker in speakers:
            # In production, extract actual embeddings from audio
            # For now, use random embeddings as placeholder
            embeddings[speaker] = np.random.randn(256)
            
        self.log(f"  Found {len(speakers)} speakers in chunk", "SUCCESS")
        return embeddings
        
    def _merge_overlapping_segments(self, segments: List[Dict]) -> List[Dict]:
        """Remove duplicate segments from overlapping regions"""
        # Sort by start time
        segments.sort(key=lambda x: x['start'])
        
        merged = []
        for seg in segments:
            # Check if this segment overlaps with previous
            if merged and seg['start'] < merged[-1]['end']:
                # Skip if it's essentially the same segment
                if abs(seg['start'] - merged[-1]['start']) < 1.0:
                    continue
                    
            merged.append(seg)
            
        return merged
        
    def _save_transcripts(self, output_dir: Path, segments: List[Dict], 
                         filename: str):
        """Save transcript files"""
        # Full transcript
        transcript_path = output_dir / "transcript" / "full_transcript.txt"
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(f"UMBRELLA PROJECT TRANSCRIPT - CHUNKED PROCESSING\n")
            f.write("=" * 80 + "\n")
            f.write(f"File: {filename}\n")
            f.write(f"Total segments: {len(segments)}\n")
            f.write("=" * 80 + "\n\n")
            
            current_speaker = None
            for seg in segments:
                speaker = seg.get('speaker', 'Unknown')
                if speaker != current_speaker:
                    f.write(f"\n[{speaker}]\n")
                    current_speaker = speaker
                    
                time_str = f"[{seg['start']/60:.1f}:{seg['start']%60:02.0f}]"
                f.write(f"{time_str} {seg['text'].strip()}\n")
                
        # Segments JSON
        segments_path = output_dir / "transcript" / "segments.json"
        with open(segments_path, 'w', encoding='utf-8') as f:
            json.dump({
                'segments': segments,
                'processing': 'chunked',
                'chunk_duration': self.chunk_duration,
                'overlap_duration': self.overlap_duration
            }, f, indent=2)