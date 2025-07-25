"""
Context system for providing conversation metadata to improve transcription
"""

from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum

class ConversationType(Enum):
    """Standard conversation types with diarization hints"""
    PHONE_CALL = "phone_call"
    INTERVIEW = "interview"
    MEETING = "meeting"
    PODCAST = "podcast"
    LEGISLATIVE = "legislative"
    FIELD_RECORDING = "field_recording"
    BROADCAST = "broadcast"
    CUSTOM = "custom"

@dataclass
class ConversationContext:
    """Context information for transcription"""
    type: ConversationType
    expected_speakers: Optional[int] = None
    speaker_names: Optional[List[str]] = None
    environment: Optional[str] = None
    description: Optional[str] = None
    language_hints: Optional[List[str]] = None
    technical_terms: Optional[List[str]] = None
    
    def to_whisper_prompt(self) -> str:
        """Generate initial prompt for Whisper based on context"""
        prompts = {
            ConversationType.PHONE_CALL: "This is a phone conversation between two people.",
            ConversationType.INTERVIEW: "This is an interview with questions and answers.",
            ConversationType.MEETING: "This is a business meeting with multiple participants.",
            ConversationType.PODCAST: "This is a podcast recording.",
            ConversationType.LEGISLATIVE: "This is a government hearing or legislative session.",
            ConversationType.FIELD_RECORDING: "This is a field recording that may contain background noise.",
            ConversationType.BROADCAST: "This is a broadcast or media recording."
        }
        
        base_prompt = prompts.get(self.type, "")
        
        # Add speaker names if provided
        if self.speaker_names:
            base_prompt += f" Speakers include: {', '.join(self.speaker_names)}."
            
        # Add environment context
        if self.environment:
            base_prompt += f" Recording environment: {self.environment}."
            
        # Add custom description
        if self.description:
            base_prompt += f" {self.description}"
            
        # Add technical terms hint
        if self.technical_terms:
            base_prompt += f" Technical terms that may appear: {', '.join(self.technical_terms[:5])}."
            
        return base_prompt.strip()
    
    def to_diarization_params(self) -> Dict:
        """Generate diarization parameters based on context"""
        # Base parameters
        params = {}
        
        # Set speaker constraints based on type
        if self.expected_speakers:
            params['min_speakers'] = max(1, self.expected_speakers - 1)
            params['max_speakers'] = self.expected_speakers + 1
        else:
            # Type-based defaults
            defaults = {
                ConversationType.PHONE_CALL: (2, 2),
                ConversationType.INTERVIEW: (2, 3),
                ConversationType.MEETING: (2, 8),
                ConversationType.PODCAST: (1, 4),
                ConversationType.LEGISLATIVE: (5, 20),
                ConversationType.FIELD_RECORDING: (1, 5),
                ConversationType.BROADCAST: (1, 3)
            }
            min_spk, max_spk = defaults.get(self.type, (1, 10))
            params['min_speakers'] = min_spk
            params['max_speakers'] = max_spk
        
        # Note: pyannote 3.x doesn't support 'segmentation' parameter
        # These parameters would need to be set differently
        # For now, just use min/max speakers
            
        return params

class ContextParser:
    """Parse context from various sources"""
    
    @staticmethod
    def from_cli_args(args) -> Optional[ConversationContext]:
        """Parse context from command line arguments"""
        if not any([args.context_type, args.speaker_names, args.environment, args.context]):
            return None
            
        context = ConversationContext(
            type=ConversationType(args.context_type) if args.context_type else ConversationType.CUSTOM,
            expected_speakers=args.expected_speakers,
            speaker_names=args.speaker_names.split(',') if args.speaker_names else None,
            environment=args.environment,
            description=args.context,
            technical_terms=args.technical_terms.split(',') if hasattr(args, 'technical_terms') and args.technical_terms else None
        )
        
        return context
    
    @staticmethod
    def from_api_request(request_data: Dict) -> Optional[ConversationContext]:
        """Parse context from API request"""
        context_data = request_data.get('context', {})
        if not context_data:
            return None
            
        return ConversationContext(
            type=ConversationType(context_data.get('type', 'custom')),
            expected_speakers=context_data.get('expected_speakers'),
            speaker_names=context_data.get('speaker_names'),
            environment=context_data.get('environment'),
            description=context_data.get('description'),
            language_hints=context_data.get('language_hints'),
            technical_terms=context_data.get('technical_terms')
        )
    
    @staticmethod
    def from_mcp_params(params: Dict) -> Optional[ConversationContext]:
        """Parse context from MCP (Model Context Protocol) parameters"""
        # MCP might send context in a specific format
        return ConversationContext(
            type=ConversationType(params.get('conversation_type', 'custom')),
            expected_speakers=params.get('speakers_count'),
            speaker_names=params.get('speakers'),
            environment=params.get('recording_environment'),
            description=params.get('context_description')
        )