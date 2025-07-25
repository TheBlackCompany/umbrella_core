#!/usr/bin/env python3
"""
Process an entire folder of audio files as a single conversation
Maintains file boundaries while creating unified output
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import argparse
import json

from transcribe import UmbrellaTranscriber, print_status, Colors
from context_system import ContextParser, ConversationContext
from interactive_context import InteractiveContextBuilder

class FolderProcessor:
    """Process multiple audio files as a single conversation"""
    
    def __init__(self, folder_path: Path, output_name: str = None, 
                 context: Optional[ConversationContext] = None):
        self.folder_path = Path(folder_path)
        self.output_name = output_name or f"{self.folder_path.name}_combined"
        self.context = context
        self.processed_dirs = []
        
    def get_audio_files(self, interactive_select: bool = False) -> List[Path]:
        """Find all audio files in folder"""
        audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.webm']
        audio_files = []
        
        # Use case-insensitive matching on Windows
        seen_files = set()  # Track files we've already added
        
        for ext in audio_extensions:
            # On Windows, glob is case-insensitive by default
            # But we'll still check both cases and deduplicate
            for pattern in [f"*{ext}", f"*{ext.upper()}"]:
                for file in self.folder_path.glob(pattern):
                    # Use lowercase path for deduplication
                    file_key = str(file).lower()
                    if file_key not in seen_files:
                        audio_files.append(file)
                        seen_files.add(file_key)
            
        # Sort by name to maintain order
        audio_files = sorted(audio_files)
        
        # Interactive selection mode
        if interactive_select and len(audio_files) > 1:
            print(f"\n{Colors.BOLD}Found {len(audio_files)} audio files:{Colors.RESET}")
            for i, file in enumerate(audio_files, 1):
                print(f"  {i}. {file.name} ({file.stat().st_size / 1024 / 1024:.1f} MB)")
            
            print(f"\n{Colors.YELLOW}Select files to process:{Colors.RESET}")
            print("  A. All files")
            print("  S. Select specific files (comma-separated numbers)")
            print("  R. Range (e.g., 1-3)")
            
            choice = input("\nYour choice: ").strip().upper()
            
            if choice == 'S':
                selections = input("Enter file numbers (e.g., 1,3,5): ").strip()
                indices = [int(x.strip())-1 for x in selections.split(',') 
                          if x.strip().isdigit()]
                audio_files = [audio_files[i] for i in indices 
                              if 0 <= i < len(audio_files)]
            elif choice == 'R':
                range_input = input("Enter range (e.g., 1-3): ").strip()
                if '-' in range_input:
                    start, end = range_input.split('-')
                    start = max(1, int(start.strip()))
                    end = min(len(audio_files), int(end.strip()))
                    audio_files = audio_files[start-1:end]
            # else 'A' or default - use all files
            
        return audio_files
    
    def process_folder(self, transcriber: UmbrellaTranscriber, 
                      interactive_select: bool = False) -> Path:
        """Process all files in folder and combine results"""
        
        audio_files = self.get_audio_files(interactive_select)
        if not audio_files:
            raise ValueError(f"No audio files found in {self.folder_path}")
            
        print(f"\n{Colors.BOLD}Found {len(audio_files)} audio files:{Colors.RESET}")
        for i, file in enumerate(audio_files, 1):
            print(f"  {i}. {file.name} ({file.stat().st_size / 1024 / 1024:.1f} MB)")
        print()
        
        # Process each file
        print(f"{Colors.BOLD}Processing files...{Colors.RESET}")
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}] Processing: {audio_file.name}")
            
            try:
                # Process with same context
                output_dir, manifest = transcriber.transcribe_file(audio_file)
                self.processed_dirs.append(output_dir)
                print_status(f"Completed: {audio_file.name}", "SUCCESS")
                
            except Exception as e:
                print_status(f"Failed to process {audio_file.name}: {e}", "ERROR")
                continue
        
        if not self.processed_dirs:
            raise RuntimeError("No files were successfully processed")
            
        # Combine all transcripts
        print(f"\n{Colors.BOLD}Combining transcripts...{Colors.RESET}")
        
        # Create output directory
        date_str = datetime.now().strftime("%Y-%m-%d")
        combined_output = Path("processed") / date_str / self.output_name
        
        # Build metadata
        metadata = {
            'source_folder': str(self.folder_path),
            'file_count': len(audio_files),
            'processed_count': len(self.processed_dirs),
            'files': [f.name for f in audio_files]
        }
        
        if self.context:
            metadata['context'] = {
                'type': self.context.type.value,
                'description': self.context.description,
                'environment': self.context.environment,
                'expected_speakers': self.context.expected_speakers
            }
        
        # Combine with enhanced formatting
        combine_manifest = self._combine_with_boundaries(
            self.processed_dirs, 
            combined_output,
            metadata,
            audio_files
        )
        
        print_status(f"Combined transcript saved to: {combined_output}", "SUCCESS")
        return combined_output
    
    def _combine_with_boundaries(self, session_dirs: List[Path], 
                                output_path: Path, metadata: Dict,
                                original_files: List[Path]) -> Dict:
        """Combine transcripts with clear file boundaries"""
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load all transcripts
        all_segments = []
        combined_text_parts = []
        file_boundaries = []
        current_time_offset = 0
        
        for i, (session_dir, audio_file) in enumerate(zip(session_dirs, original_files)):
            # Load transcript data
            manifest_path = session_dir / "manifest.json"
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            transcript_path = session_dir / "transcript" / "full_transcript.txt"
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript_text = f.read()
            
            segments_path = session_dir / "transcript" / "segments.json" 
            with open(segments_path, 'r') as f:
                segments_data = json.load(f)
            
            # Add file boundary marker
            boundary_text = f"\n{'='*80}\n"
            boundary_text += f"FILE {i+1}/{len(session_dirs)}: {audio_file.name}\n"
            boundary_text += f"Duration: {manifest['duration_seconds']/60:.1f} minutes | "
            boundary_text += f"Speakers: {manifest.get('speaker_count', 'Unknown')}\n"
            boundary_text += f"{'='*80}\n"
            
            combined_text_parts.append(boundary_text)
            combined_text_parts.append(transcript_text)
            
            # Track boundaries for navigation
            file_boundaries.append({
                'file': audio_file.name,
                'start_time': current_time_offset,
                'end_time': current_time_offset + manifest['duration_seconds'],
                'segment_range': [len(all_segments), None]
            })
            
            # Adjust segment timestamps
            for segment in segments_data.get('segments', []):
                segment['original_file'] = audio_file.name
                segment['global_start'] = segment['start'] + current_time_offset
                segment['global_end'] = segment['end'] + current_time_offset
                all_segments.append(segment)
            
            file_boundaries[-1]['segment_range'][1] = len(all_segments)
            current_time_offset += manifest['duration_seconds']
        
        # Add summary at the end
        summary_text = f"\n{'='*80}\n"
        summary_text += "SUMMARY\n"
        summary_text += f"{'='*80}\n"
        summary_text += f"Total Files: {len(session_dirs)}\n"
        summary_text += f"Total Duration: {current_time_offset/3600:.1f} hours\n"
        summary_text += f"Files Processed:\n"
        for i, boundary in enumerate(file_boundaries, 1):
            summary_text += f"  {i}. {boundary['file']} "
            summary_text += f"({boundary['end_time']-boundary['start_time']:.0f}s)\n"
        
        combined_text_parts.append(summary_text)
        
        # Save combined transcript
        combined_text = '\n'.join(combined_text_parts)
        transcript_path = output_path / "combined_transcript.txt"
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(combined_text)
        
        # Save structured data
        combined_data = {
            'metadata': metadata,
            'file_boundaries': file_boundaries,
            'total_duration_seconds': current_time_offset,
            'segments': all_segments
        }
        
        data_path = output_path / "combined_data.json"
        with open(data_path, 'w') as f:
            json.dump(combined_data, f, indent=2)
        
        # Create manifest
        manifest = {
            'type': 'folder_processing',
            'created_at': datetime.now().isoformat(),
            'source_folder': str(self.folder_path),
            'file_count': len(original_files),
            'processed_count': len(session_dirs),
            'total_duration_seconds': current_time_offset,
            'output_path': str(output_path),
            'file_boundaries': file_boundaries
        }
        
        manifest_path = output_path / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
            
        return manifest

def main():
    parser = argparse.ArgumentParser(
        description='Process entire folder of audio files as single conversation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process folder with interactive context
  python process_folder.py /path/to/audio/folder --interactive
  
  # Process with specific context
  python process_folder.py /path/to/audio/folder --context-type meeting --expected-speakers 5
  
  # Process with custom output name
  python process_folder.py /path/to/audio/folder --output "city_council_meeting"
"""
    )
    
    parser.add_argument('folder', help='Folder containing audio files')
    parser.add_argument('--output', help='Output name (default: foldername_combined)')
    parser.add_argument('--model', default='large', 
                       choices=['tiny', 'base', 'small', 'medium', 'large'])
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode for context')
    parser.add_argument('--select-files', action='store_true',
                       help='Interactively select which files to process')
    
    # Context arguments
    parser.add_argument('--context-type', 
                       choices=['phone_call', 'interview', 'meeting', 'podcast', 
                               'legislative', 'field_recording', 'broadcast'])
    parser.add_argument('--expected-speakers', type=int)
    parser.add_argument('--speaker-names', type=str)
    parser.add_argument('--environment', type=str)
    parser.add_argument('--context', type=str)
    
    args = parser.parse_args()
    
    # Print header
    print(f"\n{Colors.CYAN}{'='*60}")
    print(f"UMBRELLA FOLDER PROCESSOR")
    print(f"{'='*60}{Colors.RESET}\n")
    
    # Get context
    context = None
    if args.interactive:
        context = InteractiveContextBuilder.build_context()
    else:
        context = ContextParser.from_cli_args(args)
    
    # Initialize transcriber
    transcriber = UmbrellaTranscriber(
        model_size=args.model,
        use_gpu=True,
        enable_diarization=True,
        verbose=True,
        expected_speakers=args.expected_speakers if not context else context.expected_speakers,
        context=context
    )
    transcriber.setup()
    
    # Process folder
    processor = FolderProcessor(
        folder_path=args.folder,
        output_name=args.output,
        context=context
    )
    
    try:
        output_path = processor.process_folder(transcriber, 
                                             interactive_select=args.select_files)
        print(f"\n{Colors.GREEN}✓ Processing complete!{Colors.RESET}")
        print(f"Output: {output_path}")
        
    except Exception as e:
        print(f"\n{Colors.RED}✗ Processing failed: {e}{Colors.RESET}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())