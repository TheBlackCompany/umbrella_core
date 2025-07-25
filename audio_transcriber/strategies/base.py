"""
Base strategy interface for audio processing
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime

class ProcessingStrategy(ABC):
    """Abstract base class for processing strategies"""
    
    def __init__(self, transcriber, verbose: bool = True):
        self.transcriber = transcriber
        self.verbose = verbose
        self.start_time = None
        
    @abstractmethod
    def can_handle(self, audio_path: Path, duration: float) -> bool:
        """Determine if this strategy can handle the given audio file"""
        pass
        
    @abstractmethod
    def process(self, audio_path: Path, output_dir: Path) -> Tuple[Path, Dict]:
        """Process the audio file and return output directory and manifest"""
        pass
        
    def get_duration(self, audio_path: Path) -> float:
        """Get audio duration in seconds"""
        import librosa
        duration = librosa.get_duration(path=str(audio_path))
        return duration
        
    def log(self, message: str, level: str = "INFO"):
        """Log message if verbose"""
        if self.verbose:
            from transcribe import print_status
            print_status(message, level)
            
    def create_manifest(self, audio_path: Path, segments: List[Dict], 
                       speaker_count: int = 0, processing_info: Dict = None) -> Dict:
        """Create standard manifest"""
        from transcribe import calculate_file_hash, generate_uid
        
        manifest = {
            "uid": generate_uid(audio_path.name + datetime.now().isoformat()),
            "file_name": audio_path.name,
            "file_hash": calculate_file_hash(str(audio_path)),
            "duration_seconds": self.get_duration(audio_path),
            "language": segments[0].get("language", "en") if segments else "en",
            "speaker_count": speaker_count,
            "processing_time": time.time() - self.start_time if self.start_time else 0,
            "strategy": self.__class__.__name__,
            "timestamp": datetime.now().isoformat()
        }
        
        if processing_info:
            manifest.update(processing_info)
            
        return manifest