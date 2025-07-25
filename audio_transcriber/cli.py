#!/usr/bin/env python3
"""
CLI wrapper for Umbrella Transcriber
User-facing interface only
"""

import sys
import argparse
from pathlib import Path
from engine import TranscriptionEngine
from interactive_context import InteractiveContextBuilder

def main():
    parser = argparse.ArgumentParser(
        description='Umbrella Transcriber - User CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('path', help='Audio file or folder to process')
    parser.add_argument('--output-dir', help='Output directory')
    parser.add_argument('--model', default='large', 
                       choices=['tiny', 'base', 'small', 'medium', 'large'])
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU')
    parser.add_argument('--no-diarization', action='store_true', help='Disable speakers')
    parser.add_argument('--interactive', action='store_true', help='Interactive context')
    parser.add_argument('--expected-speakers', type=int, help='Number of speakers')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    
    args = parser.parse_args()
    
    # Get context if interactive
    context = None
    if args.interactive:
        context = InteractiveContextBuilder.build_context()
    
    # Initialize engine
    engine = TranscriptionEngine(
        model_size=args.model,
        use_gpu=not args.no_gpu,
        enable_diarization=not args.no_diarization,
        verbose=not args.quiet
    )
    
    # Process
    try:
        result = engine.process(
            Path(args.path),
            output_dir=Path(args.output_dir) if args.output_dir else None,
            context=context,
            expected_speakers=args.expected_speakers
        )
        
        # Result might be doctrine output or have output_dir
        if 'output_dir' in result:
            print(f"\nOutput: {result['output_dir']}")
        elif 'job_id' in result:
            print(f"\nJob completed: {result['job_id']}")
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())