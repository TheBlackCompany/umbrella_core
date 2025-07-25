"""
Interactive context gathering for better transcription
"""

from typing import Optional
from context_system import ConversationContext, ConversationType

class InteractiveContextBuilder:
    """Interactively build context through user prompts"""
    
    @staticmethod
    def build_context() -> Optional[ConversationContext]:
        """Interactive prompt system for context building"""
        
        print("\n" + "="*60)
        print("TRANSCRIPTION CONTEXT HELPER")
        print("="*60)
        print("\nProviding context helps improve speaker detection and accuracy.")
        print("Press Enter to skip any question.\n")
        
        # Conversation type
        print("What type of recording is this?")
        print("1. Phone call")
        print("2. In-person interview") 
        print("3. Meeting/Conference")
        print("4. Podcast")
        print("5. Legislative/Government")
        print("6. Field recording")
        print("7. Broadcast/Media")
        print("8. Other/Unknown")
        
        type_choice = input("\nSelect (1-8): ").strip()
        
        type_map = {
            '1': ConversationType.PHONE_CALL,
            '2': ConversationType.INTERVIEW,
            '3': ConversationType.MEETING,
            '4': ConversationType.PODCAST,
            '5': ConversationType.LEGISLATIVE,
            '6': ConversationType.FIELD_RECORDING,
            '7': ConversationType.BROADCAST,
            '8': ConversationType.CUSTOM
        }
        
        conv_type = type_map.get(type_choice, ConversationType.CUSTOM)
        
        # Expected speakers
        expected_speakers = None
        speaker_input = input("\nHow many speakers are in this recording? (press Enter if unsure): ").strip()
        if speaker_input.isdigit():
            expected_speakers = int(speaker_input)
        
        # Speaker names
        speaker_names = None
        if expected_speakers and expected_speakers <= 5:
            print(f"\nWould you like to provide names for the {expected_speakers} speakers?")
            names_input = input("Enter names separated by commas (e.g., John Doe, Jane Smith): ").strip()
            if names_input:
                speaker_names = [name.strip() for name in names_input.split(',')]
        
        # Environment
        print("\nWhere was this recorded?")
        print("1. Quiet room/Studio")
        print("2. Office/Indoor")
        print("3. Restaurant/Cafe (noisy)")
        print("4. Outdoor")
        print("5. Vehicle")
        print("6. Unknown")
        
        env_choice = input("\nSelect (1-6): ").strip()
        
        env_map = {
            '1': 'quiet',
            '2': 'office', 
            '3': 'noisy',
            '4': 'outdoor',
            '5': 'vehicle',
            '6': None
        }
        
        environment = env_map.get(env_choice)
        
        # Description
        print("\nDescribe the recording in your own words:")
        print("(What's being discussed, any important context, etc.)")
        description = input("> ").strip() or None
        
        # Technical terms
        print("\nAre there any technical terms, acronyms, or special vocabulary?")
        print("(Enter comma-separated, e.g., API, machine learning, cryptocurrency)")
        tech_terms_input = input("> ").strip()
        technical_terms = [term.strip() for term in tech_terms_input.split(',')] if tech_terms_input else None
        
        # Location/Date info
        print("\nAdditional details (optional):")
        location = input("Location (city, venue, etc.): ").strip() or None
        date = input("Date of recording: ").strip() or None
        
        # Build full description
        full_description = description or ""
        if location:
            full_description += f" Location: {location}."
        if date:
            full_description += f" Recorded on: {date}."
        
        # Summary
        print("\n" + "="*60)
        print("CONTEXT SUMMARY")
        print("="*60)
        print(f"Type: {conv_type.value}")
        print(f"Speakers: {expected_speakers or 'Auto-detect'}")
        if speaker_names:
            print(f"Names: {', '.join(speaker_names)}")
        print(f"Environment: {environment or 'Unknown'}")
        if description:
            print(f"Description: {description}")
        if technical_terms:
            print(f"Technical terms: {', '.join(technical_terms[:5])}")
        print("="*60)
        
        confirm = input("\nUse this context? (Y/n): ").strip().lower()
        if confirm == 'n':
            return None
            
        # Create context object
        context = ConversationContext(
            type=conv_type,
            expected_speakers=expected_speakers,
            speaker_names=speaker_names,
            environment=environment,
            description=full_description.strip() if full_description else None,
            technical_terms=technical_terms
        )
        
        return context