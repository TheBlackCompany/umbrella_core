#!/usr/bin/env python3
"""
Umbrella Transcriber - Unified Audio Transcription Tool
GPU-accelerated transcription with optional speaker diarization
"""

import os
import sys
import time
import json
import hashlib
import shutil
import argparse
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List, Optional
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set UTF-8 encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Fix for Git Bash/MINGW subprocess issues
if 'MINGW' in os.environ.get('MSYSTEM', ''):
    # Force Windows-style subprocess handling
    import subprocess
    subprocess._USE_POSIX_SPAWN = False

# Load .env file if it exists
env_file = Path(__file__).parent / '.env'
if env_file.exists():
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                # Remove quotes if present
                value = value.strip('"').strip("'")
                os.environ[key] = value

import torch
import whisper

# Check for optional dependencies
try:
    from pyannote.audio import Pipeline
    DIARIZATION_AVAILABLE = True
except ImportError:
    DIARIZATION_AVAILABLE = False

# Try to enable colors on Windows
if sys.platform == "win32":
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        COLORS_ENABLED = True
    except:
        COLORS_ENABLED = False
else:
    COLORS_ENABLED = True

# Color codes
class Colors:
    CYAN = "\033[36m" if COLORS_ENABLED else ""
    GREEN = "\033[32m" if COLORS_ENABLED else ""
    YELLOW = "\033[33m" if COLORS_ENABLED else ""
    RED = "\033[31m" if COLORS_ENABLED else ""
    MAGENTA = "\033[35m" if COLORS_ENABLED else ""
    BLUE = "\033[34m" if COLORS_ENABLED else ""
    GRAY = "\033[90m" if COLORS_ENABLED else ""
    RESET = "\033[0m" if COLORS_ENABLED else ""
    BOLD = "\033[1m" if COLORS_ENABLED else ""

def print_status(message: str, level: str = "INFO", color: bool = True):
    """Print formatted status messages with optional color"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    symbols = {
        "INFO": "[*]",
        "SUCCESS": "[+]",
        "WARNING": "[!]",
        "ERROR": "[X]",
        "GPU": "[GPU]",
        "PROCESS": "[>]"
    }
    
    # Use color class
    colors = {
        "INFO": Colors.CYAN,
        "SUCCESS": Colors.GREEN,
        "WARNING": Colors.YELLOW,
        "ERROR": Colors.RED,
        "GPU": Colors.MAGENTA,
        "PROCESS": Colors.BLUE
    }
    
    reset = Colors.RESET
    
    if color and sys.platform != "win32":  # Use colors on non-Windows
        color_code = colors.get(level, "")
        print(f"{color_code}{timestamp} {symbols.get(level, '[*]')} {message}{reset}")
    elif color and sys.platform == "win32":
        # Try to enable ANSI on Windows
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            color_code = colors.get(level, "")
            print(f"{color_code}{timestamp} {symbols.get(level, '[*]')} {message}{reset}")
        except:
            # Fallback to no color
            print(f"{timestamp} {symbols.get(level, '[*]')} {message}")
    else:
        print(f"{timestamp} {symbols.get(level, '[*]')} {message}")

def generate_uid(seed: str) -> str:
    """Generate consistent 8-character UID"""
    return hashlib.sha256(seed.encode()).hexdigest()[:8]

def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA256 hash of file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

class UmbrellaTranscriber:
    """Unified transcription system"""
    
    def __init__(self, model_size: str = "large", use_gpu: bool = True, 
                 enable_diarization: bool = True, verbose: bool = True,
                 expected_speakers: Optional[int] = None, force_chunked: bool = False,
                 context: Optional['ConversationContext'] = None):
        self.model_size = model_size
        self.device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        self.enable_diarization = enable_diarization and DIARIZATION_AVAILABLE
        self.verbose = verbose
        self.expected_speakers = expected_speakers
        self.force_chunked = force_chunked
        self.context = context
        self.model = None
        self.diarization_pipeline = None
        
    def setup(self):
        """Initialize models and check system"""
        if self.verbose:
            # Enable colors on Windows
            if sys.platform == "win32":
                try:
                    import ctypes
                    kernel32 = ctypes.windll.kernel32
                    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
                except:
                    pass
            
            print(Colors.CYAN + "=" * 70 + Colors.RESET)
            print(Colors.CYAN + Colors.BOLD + "UMBRELLA TRANSCRIBER".center(70) + Colors.RESET)
            print(Colors.CYAN + "=" * 70 + Colors.RESET)
            
            # System info
            if self.device == "cuda":
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print_status(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)", "GPU")
                print_status(f"CUDA: {torch.version.cuda}", "GPU")
            else:
                print_status("Running on CPU (slower)", "WARNING")
            
            # Diarization status
            if self.enable_diarization:
                hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
                if hf_token:
                    print_status("Speaker diarization: Available", "SUCCESS")
                    print_status(f"Token: ...{hf_token[-8:]}", "INFO")  # Show last 8 chars
                else:
                    print_status("Speaker diarization: Requested but no token found", "WARNING")
                    print_status("Create .env file with HUGGINGFACE_TOKEN=your_token", "INFO")
                    print_status("Get a free token at: https://huggingface.co/settings/tokens", "INFO")
                    self.enable_diarization = False
            elif DIARIZATION_AVAILABLE:
                print_status("Speaker diarization: Disabled by user", "INFO")
            else:
                print_status("Speaker diarization: Not installed", "INFO")
        
        # Load Whisper model
        if self.verbose:
            print_status(f"Loading Whisper {self.model_size} model...", "PROCESS")
        
        self.model = whisper.load_model(self.model_size, device=self.device)
        
        if self.verbose:
            print_status("Model loaded successfully", "SUCCESS")
    
    def transcribe_file(self, audio_path: Path, output_dir: Optional[Path] = None) -> Tuple[Path, Dict]:
        """Transcribe a single audio file using appropriate strategy"""
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        # Determine strategy based on duration
        try:
            # Get duration using Whisper's built-in audio loading
            duration = None
            
            try:
                # Whisper already handles all audio formats internally via ffmpeg
                # This is the same method used for transcription, so it's guaranteed to work
                if self.verbose:
                    print_status(f"Detecting audio duration...", "INFO")
                
                audio = whisper.load_audio(str(audio_path))
                duration = len(audio) / 16000  # Whisper uses 16kHz sampling
                
                if self.verbose:
                    print_status(f"Duration: {duration/60:.1f} minutes", "INFO")
                    
            except Exception as e:
                if self.verbose:
                    print_status(f"Duration detection failed: {e}", "WARNING")
            
            # If we still don't have duration, force chunked for large files
            if duration is None:
                file_size_mb = audio_path.stat().st_size / (1024 * 1024)
                if file_size_mb > 100:  # Over 100MB, likely a long file
                    duration = 3600  # Assume 1 hour to trigger chunking
                    if self.verbose:
                        print_status(f"Large file ({file_size_mb:.1f}MB) - forcing chunked mode", "WARNING")
            
            if self.force_chunked or (duration and duration > 30 * 60):  # Force chunked or over 30 minutes
                if self.verbose:
                    if self.force_chunked:
                        print_status(f"Force chunked mode enabled", "INFO")
                    print_status(f"File duration: {duration/60:.1f} minutes - using chunked strategy", "INFO")
                
                # Use chunked strategy
                from strategies.chunked import ChunkedStrategy
                strategy = ChunkedStrategy(self, self.verbose)
                
                # Debug logging
                if self.verbose:
                    print_status(f"Diarization enabled: {self.enable_diarization}", "DEBUG")
                    print_status(f"Diarization pipeline loaded: {self.diarization_pipeline is not None}", "DEBUG")
                
                # Setup output directory for chunked processing
                if output_dir is None:
                    date_str = datetime.now().strftime("%Y-%m-%d")
                    uid = generate_uid(audio_path.name + datetime.now().isoformat())
                    output_dir = Path("processed") / date_str / f"{audio_path.stem}_{uid}"
                    
                return strategy.process(audio_path, output_dir)
                
        except Exception as e:
            if self.verbose:
                print_status(f"Could not determine duration, using standard processing: {e}", "WARNING")
        
        # Use standard processing
        return self._transcribe_file_standard(audio_path, output_dir)
        
    def _transcribe_file_standard(self, audio_path: Path, output_dir: Optional[Path] = None) -> Tuple[Path, Dict]:
        """Original transcribe_file implementation"""
        
        # Setup output directory
        if output_dir is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
            uid = generate_uid(audio_path.name + datetime.now().isoformat())
            output_dir = Path("processed") / date_str / f"{audio_path.stem}_{uid}"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        for subdir in ["audio", "transcript", "status"]:
            (output_dir / subdir).mkdir(exist_ok=True)
        
        # Copy audio file (handle paths with spaces by renaming)
        # Replace spaces and problematic characters in filename
        safe_name = audio_path.name.replace(' ', '_').replace('(', '').replace(')', '')
        audio_dest = output_dir / "audio" / safe_name
        
        try:
            shutil.copy2(str(audio_path), str(audio_dest))
            if safe_name != audio_path.name and self.verbose:
                print_status(f"Renamed to: {safe_name}", "INFO")
        except Exception as e:
            # More detailed error for debugging
            print_status(f"Failed to copy file: {e}", "ERROR")
            print_status(f"Source: {audio_path}", "ERROR")
            print_status(f"Source exists: {audio_path.exists()}", "ERROR")
            print_status(f"Dest: {audio_dest}", "ERROR")
            raise
        
        # Calculate file hash
        file_hash = calculate_file_hash(str(audio_dest))
        
        # Status file
        status_file = output_dir / "status.txt"
        status_file.write_text("PROCESSING")
        
        if self.verbose:
            print(f"\n{Colors.BOLD}[Processing {audio_path.name}]{Colors.RESET}")
            print_status(f"Size: {audio_path.stat().st_size / 1024 / 1024:.1f} MB", "INFO")
        
        try:
            # Transcribe
            if self.verbose:
                print_status("Starting transcription...", "PROCESS")
            
            transcribe_start = time.time()
            
            # Ensure FFmpeg is available for Whisper
            import subprocess
            import platform
            
            # On Windows, we might need to help subprocess find ffmpeg
            if platform.system() == "Windows":
                # Add common FFmpeg locations to PATH temporarily
                ffmpeg_path = shutil.which("ffmpeg")
                if not ffmpeg_path:
                    # Check common locations
                    common_paths = [
                        r"C:\ProgramData\chocolatey\bin",
                        r"C:\ffmpeg\bin",
                        r"C:\Program Files\ffmpeg\bin",
                        str(Path(__file__).parent / "ffmpeg_bin")
                    ]
                    for path in common_paths:
                        if (Path(path) / "ffmpeg.exe").exists():
                            os.environ["PATH"] = f"{path};{os.environ['PATH']}"
                            break
            
            try:
                # Test FFmpeg availability
                result = subprocess.run(["ffmpeg", "-version"], 
                                      capture_output=True, 
                                      shell=(platform.system() == "Windows"))
                if result.returncode != 0:
                    raise RuntimeError("FFmpeg test failed")
            except Exception as e:
                print_status(f"FFmpeg issue: {e}", "ERROR")
                print_status("FFmpeg is required for audio processing", "ERROR")
                print_status("Install with: choco install ffmpeg", "INFO")
                raise RuntimeError("FFmpeg is required but not working properly")
            
            # Use absolute path for audio file
            audio_file_path = str(audio_dest.resolve())
            
            # Special handling for Git Bash/MINGW
            if 'MINGW' in os.environ.get('MSYSTEM', ''):
                if self.verbose:
                    print_status("Detected Git Bash - using compatibility mode", "INFO")
                # Try to use Windows-style path
                try:
                    import pathlib
                    audio_file_path = str(pathlib.PureWindowsPath(audio_dest.resolve()))
                except:
                    pass
            
            # Generate initial prompt from context
            initial_prompt = None
            if self.context:
                initial_prompt = self.context.to_whisper_prompt()
                if self.verbose and initial_prompt:
                    print_status(f"Using context prompt: {initial_prompt[:100]}...", "INFO")
            
            try:
                result = self.model.transcribe(
                    audio_file_path,
                    fp16=(self.device == "cuda"),
                    language=None,
                    verbose=self.verbose,
                    word_timestamps=True,
                    task="transcribe",
                    temperature=0.0,
                    compression_ratio_threshold=2.4,
                    logprob_threshold=-1.0,
                    no_speech_threshold=0.6,
                    condition_on_previous_text=False,  # This helps prevent repetition
                    initial_prompt=initial_prompt,
                    suppress_tokens="-1",
                    without_timestamps=False,
                    max_initial_timestamp=1.0,
                    hallucination_silence_threshold=2.0  # Skip sections that might cause loops
                )
            except Exception as e:
                if "[WinError 2]" in str(e) and 'MINGW' in os.environ.get('MSYSTEM', ''):
                    print_status("Git Bash compatibility issue detected", "ERROR")
                    print_status("Please use Windows Command Prompt or PowerShell instead:", "INFO")
                    print_status("  1. Open Command Prompt (cmd)", "INFO")
                    print_status("  2. cd C:\\umbrella\\umbrella_transcriber", "INFO")
                    print_status("  3. venv\\Scripts\\activate", "INFO")  
                    print_status("  4. python transcribe.py", "INFO")
                    raise RuntimeError("Git Bash subprocess compatibility issue - use Command Prompt")
                raise
            
            transcribe_time = time.time() - transcribe_start
            
            # Extract results
            text = result.get('text', '').strip()
            segments = result.get('segments', [])
            language = result.get('language', 'unknown')
            
            # Clean up repetitions in segments
            segments = self._clean_repetitions(segments)
            
            # Performance metrics
            audio_duration = segments[-1]['end'] if segments else 0
            transcribe_speed = audio_duration / transcribe_time if transcribe_time > 0 else 0
            
            if self.verbose:
                print_status(f"Transcription complete ({transcribe_speed:.1f}x realtime)", "SUCCESS")
            
            # Speaker diarization
            diarization_time = 0.0
            speaker_count = 1
            
            if self.enable_diarization and segments:
                diarization_time = self._perform_diarization(audio_dest, segments)
                speakers = set(s.get("speaker", "Unknown") for s in segments)
                speaker_count = len(speakers)
            
            # Total performance
            total_time = transcribe_time + diarization_time
            total_speed = audio_duration / total_time if total_time > 0 else 0
            
            if self.verbose:
                print_status(f"Total processing: {total_time:.1f}s ({total_speed:.1f}x realtime)", "SUCCESS")
                print_status(f"Language: {language}, Speakers: {speaker_count}", "INFO")
            
            # Save outputs
            self._save_outputs(
                output_dir, audio_path, file_hash, text, segments,
                language, audio_duration, speaker_count,
                transcribe_time, diarization_time, total_speed
            )
            
            # Create manifest
            manifest = {
                'uid': output_dir.name.split('_')[-1],
                'file_name': audio_path.name,
                'file_hash': file_hash,
                'duration_seconds': audio_duration,
                'language': language,
                'speaker_count': speaker_count,
                'transcription_time': transcribe_time,
                'diarization_time': diarization_time,
                'total_speed': total_speed,
                'model': self.model_size,
                'device': self.device,
                'timestamp': datetime.now().isoformat()
            }
            
            manifest_file = output_dir / "manifest.json"
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            # Update status
            status_file.write_text("COMPLETE")
            
            return output_dir, manifest
            
        except Exception as e:
            status_file.write_text(f"FAILED: {str(e)}")
            raise
    
    def _clean_repetitions(self, segments: List[Dict], max_repetitions: int = 3) -> List[Dict]:
        """Clean up repetitive segments caused by Whisper hallucinations"""
        cleaned_segments = []
        
        for segment in segments:
            text = segment['text'].strip()
            words = text.split()
            
            # Check for excessive repetition
            if len(words) > 10:
                # Count consecutive repeated words
                repetition_count = 1
                max_consecutive = 1
                
                for i in range(1, len(words)):
                    if words[i].lower() == words[i-1].lower():
                        repetition_count += 1
                        max_consecutive = max(max_consecutive, repetition_count)
                    else:
                        repetition_count = 1
                
                # If more than half the words are the same repeated word, it's likely a hallucination
                word_counts = {}
                for word in words:
                    word_lower = word.lower()
                    word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
                
                most_common_count = max(word_counts.values()) if word_counts else 0
                
                if max_consecutive > max_repetitions or most_common_count > len(words) * 0.5:
                    # This segment is likely a hallucination
                    if self.verbose:
                        print_status(f"Detected repetition at {segment['start']:.1f}s, filtering...", "WARNING")
                    # Replace with a marker
                    segment['text'] = "[audio unclear]"
                    segment['filtered'] = True
            
            cleaned_segments.append(segment)
        
        return cleaned_segments
    
    def _perform_diarization(self, audio_path: Path, segments: List[Dict]) -> float:
        """Perform speaker diarization on segments"""
        
        if not self.enable_diarization:
            return 0.0
        
        hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
        if not hf_token:
            return 0.0
        
        if self.verbose:
            print_status("Performing speaker diarization...", "PROCESS")
        
        start_time = time.time()
        
        # Convert to WAV if not already (PyAnnote works better with WAV)
        if audio_path.suffix.lower() != '.wav':
            if self.verbose:
                print_status(f"Converting {audio_path.suffix} to WAV for diarization...", "PROCESS")
            
            wav_path = audio_path.with_suffix('.wav')
            try:
                import subprocess
                cmd = [
                    'ffmpeg', '-i', str(audio_path),
                    '-acodec', 'pcm_s16le',
                    '-ar', '16000',
                    '-ac', '1',
                    str(wav_path),
                    '-y'  # Overwrite if exists
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                audio_path = wav_path
            except Exception as e:
                if self.verbose:
                    print_status(f"Could not convert to WAV: {e}", "WARNING")
                # Continue with original file, might still work
        
        try:
            # Load pipeline if not already loaded
            if self.diarization_pipeline is None:
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization@2.1",
                    use_auth_token=hf_token
                )
                
                # Move to GPU if available
                if self.device == "cuda":
                    self.diarization_pipeline.to(torch.device("cuda"))
                    if self.verbose:
                        print_status("Diarization using GPU", "GPU")
            
            # Run diarization with progress indicator
            if self.verbose:
                # Estimate time based on audio duration
                audio_duration = segments[-1]['end'] if segments else 0
                estimated_time = audio_duration / 30  # Roughly 30x realtime on GPU
                print_status(f"Processing {audio_duration:.1f}s of audio (~{estimated_time:.0f}s estimated)", "INFO")
                
                # Start progress indicator in background
                stop_progress = threading.Event()
                def show_progress():
                    start = time.time()
                    spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
                    idx = 0
                    while not stop_progress.is_set():
                        elapsed = time.time() - start
                        # Get GPU memory usage if available
                        gpu_info = ""
                        if self.device == "cuda":
                            try:
                                allocated = torch.cuda.memory_allocated() / 1024**3
                                reserved = torch.cuda.memory_reserved() / 1024**3
                                gpu_info = f" | GPU: {allocated:.1f}/{reserved:.1f}GB"
                            except:
                                pass
                        
                        spinner_char = spinner[idx % len(spinner)]
                        print(f"\r{Colors.YELLOW}{spinner_char} [GPU] Processing speakers... {elapsed:.1f}s{gpu_info}{Colors.RESET}", end='', flush=True)
                        idx += 1
                        time.sleep(0.5)
                    print(f"\r{' ' * 80}\r", end='', flush=True)  # Clear the line
                
                progress_thread = threading.Thread(target=show_progress)
                progress_thread.start()
            
            try:
                # Check if we need to convert to WAV for diarization
                audio_path_for_diarization = audio_path
                temp_wav = None
                
                if audio_path.suffix.lower() != '.wav':
                    if self.verbose:
                        print_status("Converting to WAV format for diarization...", "INFO")
                    
                    # Create temporary WAV file
                    temp_wav = audio_path.parent / f"{audio_path.stem}_temp.wav"
                    
                    # Use ffmpeg to convert
                    import subprocess
                    cmd = [
                        "ffmpeg", "-i", str(audio_path),
                        "-ar", "16000",  # 16kHz sample rate
                        "-ac", "1",      # Mono
                        "-y",            # Overwrite
                        str(temp_wav)
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")
                    
                    audio_path_for_diarization = temp_wav
                
                # Apply speaker hints if provided
                diarization_params = {}
                
                # Use context-based parameters if available
                if self.context:
                    diarization_params.update(self.context.to_diarization_params())
                    if self.verbose:
                        print_status(f"Using context '{self.context.type.value}' for diarization", "INFO")
                elif self.expected_speakers:
                    # Set min/max speakers based on expected count
                    # Allow some flexibility (±1 speaker)
                    diarization_params['min_speakers'] = max(2, self.expected_speakers - 1)
                    diarization_params['max_speakers'] = self.expected_speakers + 1
                    
                    if self.verbose:
                        print_status(f"Using speaker hints: expecting ~{self.expected_speakers} speakers", "INFO")
                
                diarization = self.diarization_pipeline(str(audio_path_for_diarization), **diarization_params)
                
                # Clean up temp file if created
                if temp_wav and temp_wav.exists():
                    temp_wav.unlink()
                    
            finally:
                if self.verbose:
                    stop_progress.set()
                    progress_thread.join()
            
            # Map segments to speakers
            for segment in segments:
                segment_time = (segment['start'] + segment['end']) / 2
                
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    if turn.start <= segment_time <= turn.end:
                        segment["speaker"] = f"Speaker_{speaker.split('_')[-1]}"
                        break
                else:
                    segment["speaker"] = "Unknown"
            
            diarization_time = time.time() - start_time
            
            if self.verbose:
                speakers = set(s.get("speaker", "Unknown") for s in segments)
                print_status(f"Found {len(speakers)} speakers ({diarization_time:.1f}s)", "SUCCESS")
            
            # Clean up temporary WAV if we created one
            if 'wav_path' in locals() and wav_path.exists() and wav_path != audio_path:
                try:
                    wav_path.unlink()
                except:
                    pass
            
            return diarization_time
            
        except Exception as e:
            if self.verbose:
                print_status(f"Diarization failed: {e}", "WARNING")
            return 0.0
    
    def _save_outputs(self, output_dir: Path, audio_path: Path, file_hash: str,
                     text: str, segments: List[Dict], language: str,
                     audio_duration: float, speaker_count: int,
                     transcribe_time: float, diarization_time: float,
                     total_speed: float):
        """Save all output files"""
        
        # Apply speaker consolidation if context suggests it
        consolidation_info = None
        if speaker_count > 2 and self.context:
            try:
                from speaker_consolidation import SpeakerConsolidator
                
                consolidator = SpeakerConsolidator(self.context)
                consolidated_segments, mapping = consolidator.consolidate_speakers(
                    segments, 
                    self.expected_speakers
                )
                
                # Update segments and speaker count
                segments = consolidated_segments
                consolidated_speaker_count = len(set(s.get('speaker', 'Unknown') for s in segments))
                
                consolidation_info = {
                    'performed': True,
                    'mapping': mapping,
                    'original_speaker_count': speaker_count,
                    'consolidated_speaker_count': consolidated_speaker_count
                }
                
                # Update speaker count for display
                speaker_count = consolidated_speaker_count
                
                if self.verbose:
                    print_status(f"Consolidated {consolidation_info['original_speaker_count']} speakers to {speaker_count}", "INFO")
                    
            except Exception as e:
                if self.verbose:
                    print_status(f"Speaker consolidation failed: {e}", "WARNING")
        
        # Full transcript with formatting
        transcript_file = output_dir / "transcript" / "full_transcript.txt"
        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write("UMBRELLA PROJECT TRANSCRIPT\n")
            f.write("=" * 80 + "\n")
            f.write(f"File: {audio_path.name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {audio_duration:.1f} seconds\n")
            f.write(f"Language: {language.upper()}\n")
            if consolidation_info:
                f.write(f"Speakers: {speaker_count} (consolidated from {consolidation_info['original_speaker_count']})\n")
            else:
                f.write(f"Speakers: {speaker_count}\n")
            f.write(f"Processing Speed: {total_speed:.1f}x realtime\n")
            f.write("=" * 80 + "\n\n")
            
            # Segments with timestamps and speakers
            if speaker_count > 1:
                f.write("TRANSCRIPT WITH SPEAKERS\n")
                f.write("-" * 40 + "\n\n")
                
                current_speaker = None
                for segment in segments:
                    speaker = segment.get('speaker', 'Unknown')
                    if speaker != current_speaker:
                        f.write(f"\n[{speaker}]\n")
                        current_speaker = speaker
                    
                    start = segment['start']
                    text_seg = segment['text'].strip()
                    mins = int(start // 60)
                    secs = int(start % 60)
                    f.write(f"[{mins:02d}:{secs:02d}] {text_seg}\n")
            else:
                f.write("TRANSCRIPT WITH TIMESTAMPS\n")
                f.write("-" * 40 + "\n\n")
                
                for segment in segments:
                    start = segment['start']
                    text_seg = segment['text'].strip()
                    mins = int(start // 60)
                    secs = int(start % 60)
                    f.write(f"[{mins:02d}:{secs:02d}] {text_seg}\n\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("FULL TEXT\n")
            f.write("-" * 40 + "\n\n")
            f.write(text)
        
        # Segments JSON
        segments_file = output_dir / "transcript" / "segments.json"
        segments_data = {
            'file_hash': file_hash,
            'language': language,
            'duration': audio_duration,
            'speaker_count': speaker_count,
            'segments': segments
        }
        
        # Add consolidation info if performed
        if consolidation_info:
            segments_data['speaker_consolidation'] = consolidation_info
            
        with open(segments_file, 'w') as f:
            json.dump(segments_data, f, indent=2)
        
        if self.verbose:
            print_status(f"Output saved to: {output_dir}", "SUCCESS")

def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Umbrella Transcriber - GPU-accelerated audio transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all files in raw/ directory
  python transcribe.py
  
  # Process specific files
  python transcribe.py audio1.mp3 audio2.m4a
  
  # Use large model for better accuracy
  python transcribe.py --model large
  
  # Disable speaker diarization for faster processing
  python transcribe.py --no-diarization
  
  # Force CPU processing
  python transcribe.py --cpu
  
  # Quiet mode (minimal output)
  python transcribe.py --quiet
  
  # Interactive mode (choose settings via prompts)
  python transcribe.py --interactive
"""
    )
    
    parser.add_argument('files', nargs='*', help='Audio files to process (default: all in raw/)')
    parser.add_argument('--model', '-m', default='large',
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Whisper model size (default: large)')
    parser.add_argument('--expected-speakers', type=int,
                       help='Expected number of speakers (helps prevent over-segmentation)')
    
    # Context arguments
    parser.add_argument('--context-type', 
                       choices=['phone_call', 'interview', 'meeting', 'podcast', 
                               'legislative', 'field_recording', 'broadcast'],
                       help='Type of conversation for better context')
    parser.add_argument('--speaker-names', type=str,
                       help='Comma-separated list of speaker names (e.g., "John Steel,Rob Scale")')
    parser.add_argument('--environment', type=str,
                       help='Recording environment (e.g., "noisy", "restaurant", "quiet", "studio")')
    parser.add_argument('--context', type=str,
                       help='Free-form context description')
    parser.add_argument('--technical-terms', type=str,
                       help='Comma-separated technical terms that may appear')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode - prompts for context information')
    parser.add_argument('--force-chunked', action='store_true',
                       help='Force chunked processing for any file size')
    parser.add_argument('--no-diarization', '-nd', action='store_true',
                       help='Disable speaker diarization')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU processing (no GPU)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal output')
    parser.add_argument('--output-dir', '-o', type=str,
                       help='Custom output directory')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Interactive mode - choose settings via prompts')
    
    args = parser.parse_args()
    
    # Default to interactive mode if no arguments provided
    if len(sys.argv) == 1:  # No arguments provided
        args.interactive = True
    
    # Interactive mode
    if args.interactive and not args.quiet:
        # Enable colors on Windows if possible
        if sys.platform == "win32":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            except:
                pass
        
        # Colorful header
        print(Colors.CYAN + "=" * 70 + Colors.RESET)
        print(Colors.CYAN + Colors.BOLD + "UMBRELLA TRANSCRIBER".center(70) + Colors.RESET)
        print(Colors.YELLOW + "Audio to Text Conversion with GPU Acceleration".center(70) + Colors.RESET)
        print(Colors.CYAN + "=" * 70 + Colors.RESET)
        print(f"\n{Colors.GREEN}Welcome!{Colors.RESET} This tool will convert your audio files to text.")
        print("Let's set up your preferences...")
        
        # Model selection
        print(f"\n{Colors.CYAN}Select Whisper model:{Colors.RESET}")
        print(f"{Colors.YELLOW}1.{Colors.RESET} tiny   {Colors.GRAY}(39 MB){Colors.RESET}  - Fastest, basic accuracy")
        print(f"{Colors.YELLOW}2.{Colors.RESET} base   {Colors.GRAY}(74 MB){Colors.RESET}  - Balanced speed/accuracy {Colors.GREEN}[Recommended]{Colors.RESET}")
        print(f"{Colors.YELLOW}3.{Colors.RESET} small  {Colors.GRAY}(244 MB){Colors.RESET} - Good accuracy")
        print(f"{Colors.YELLOW}4.{Colors.RESET} medium {Colors.GRAY}(769 MB){Colors.RESET} - High accuracy")
        print(f"{Colors.YELLOW}5.{Colors.RESET} large  {Colors.GRAY}(1.5 GB){Colors.RESET} - Best accuracy")
        
        model_choice = input("\nChoice [1-5] (default: 2): ").strip() or "2"
        model_map = {"1": "tiny", "2": "base", "3": "small", "4": "medium", "5": "large"}
        args.model = model_map.get(model_choice, "base")
        
        # Speaker diarization
        if DIARIZATION_AVAILABLE:
            hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
            if hf_token:
                print(f"\n{Colors.GREEN}✓ HuggingFace token found{Colors.RESET}")
                diarization_choice = input("Enable speaker identification? [y/N]: ").strip().lower()
                args.no_diarization = diarization_choice != 'y'
            else:
                print(f"\n{Colors.YELLOW}Note: Speaker identification requires a HuggingFace token{Colors.RESET}")
                print(f"{Colors.GRAY}Set environment variable HUGGINGFACE_TOKEN to enable{Colors.RESET}")
                args.no_diarization = True
        
        # GPU selection
        if torch.cuda.is_available():
            gpu_choice = input("\nUse GPU acceleration? [Y/n]: ").strip().lower()
            args.cpu = gpu_choice == 'n'
        
        print(f"\n{Colors.CYAN}Settings:{Colors.RESET}")
        print(f"  {Colors.YELLOW}Model:{Colors.RESET} {args.model}")
        print(f"  {Colors.YELLOW}GPU:{Colors.RESET} {'Yes' if not args.cpu else 'No'}")
        print(f"  {Colors.YELLOW}Speaker ID:{Colors.RESET} {'Yes' if not args.no_diarization else 'No'}")
        print()
    
    # Import context system
    from context_system import ContextParser
    
    # Parse context from CLI args or interactive mode
    context = None
    if args.interactive:
        from interactive_context import InteractiveContextBuilder
        context = InteractiveContextBuilder.build_context()
        if not context:
            print("Proceeding without context...")
    else:
        context = ContextParser.from_cli_args(args)
    
    # Initialize transcriber
    transcriber = UmbrellaTranscriber(
        model_size=args.model,
        use_gpu=not args.cpu,
        enable_diarization=not args.no_diarization,
        verbose=not args.quiet,
        expected_speakers=args.expected_speakers,
        force_chunked=args.force_chunked,
        context=context
    )
    
    # Setup models
    transcriber.setup()
    
    # Find files to process
    if args.files:
        audio_files = [Path(f) for f in args.files if Path(f).exists()]
    else:
        # Default to raw/ directory
        raw_dir = Path("raw")
        if not raw_dir.exists():
            raw_dir.mkdir()
            
        audio_files = []
        for ext in ['.m4a', '.mp3', '.wav', '.flac', '.ogg', '.webm', '.mp4', '.avi', '.mov']:
            audio_files.extend(raw_dir.glob(f"*{ext}"))
    
    if not audio_files:
        print_status("No audio files found", "WARNING")
        print_status("Place audio files in raw/ directory or specify files", "INFO")
        return
    
    # Interactive file selection
    if args.interactive and not args.quiet and len(audio_files) > 1:
        print("\nAvailable audio files:")
        for i, f in enumerate(audio_files, 1):
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"{i}. {f.name} ({size_mb:.1f} MB)")
        
        print(f"\nSelect files to process:")
        print("  - Enter numbers separated by commas (e.g., 1,3,5)")
        print("  - Enter 'all' to process all files")
        print("  - Press Enter for all files")
        
        selection = input("\nSelection: ").strip().lower()
        
        if selection and selection != "all":
            try:
                indices = [int(x.strip()) - 1 for x in selection.split(",")]
                selected_files = [audio_files[i] for i in indices if 0 <= i < len(audio_files)]
                if selected_files:
                    audio_files = selected_files
                else:
                    print_status("Invalid selection, processing all files", "WARNING")
            except ValueError:
                print_status("Invalid selection, processing all files", "WARNING")
    
    if not args.quiet:
        print(f"\nFound {len(audio_files)} file(s) to process")
    
    # Process files
    success_count = 0
    for i, audio_file in enumerate(audio_files, 1):
        if not args.quiet:
            print(f"\n[{i}/{len(audio_files)}]")
        
        try:
            output_dir = None
            if args.output_dir:
                output_dir = Path(args.output_dir) / audio_file.stem
            
            output_path, manifest = transcriber.transcribe_file(audio_file, output_dir)
            success_count += 1
            
            # Move processed file to completed/ directory (preserve original)
            if audio_file.parent.name == "raw":
                completed_dir = Path("completed")
                completed_dir.mkdir(exist_ok=True)
                
                # Move to completed with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dest_name = f"{timestamp}_{audio_file.name}"
                dest_path = completed_dir / dest_name
                
                try:
                    shutil.move(str(audio_file), str(dest_path))
                    if not args.quiet:
                        print_status(f"Moved original to: completed/{dest_name}", "INFO")
                except Exception as e:
                    print_status(f"Could not move file: {e}", "WARNING")
                    # Leave file in place if move fails
                
        except Exception as e:
            print_status(f"Error processing {audio_file.name}: {e}", "ERROR")
    
    # Summary
    if not args.quiet:
        print(f"\n{Colors.CYAN}{'=' * 70}{Colors.RESET}")
        if success_count == len(audio_files):
            print(f"{Colors.GREEN}{Colors.BOLD}✓ ALL FILES PROCESSED SUCCESSFULLY{Colors.RESET}".center(70 + len(Colors.GREEN) + len(Colors.BOLD) + len(Colors.RESET)))
        else:
            print(f"{Colors.YELLOW}PROCESSING COMPLETE{Colors.RESET}".center(70 + len(Colors.YELLOW) + len(Colors.RESET)))
        print(f"{Colors.GREEN}Success: {success_count}{Colors.RESET} | {Colors.RED}Failed: {len(audio_files) - success_count}{Colors.RESET}")
        print(f"{Colors.CYAN}{'=' * 70}{Colors.RESET}")

if __name__ == "__main__":
    main()