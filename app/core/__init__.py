"""
Core functionality for the Clippy application.

This package contains the core modules for audio processing, voice separation,
and other fundamental functionality of the application.
"""

from app.core.audio_processor import AudioProcessor, AudioValidationError
from app.core.voice_separator import VoiceSeparator, VoiceSeparationError
from app.core.diarization import Diarizer, DiarizationError, SpeakerSegment
from app.core.embedding import EmbeddingProcessor, SpeakerEmbedding, EmbeddingError

__all__ = [
    'AudioProcessor', 'AudioValidationError',
    'VoiceSeparator', 'VoiceSeparationError',
    'Diarizer', 'DiarizationError', 'SpeakerSegment',
    'EmbeddingProcessor', 'SpeakerEmbedding', 'EmbeddingError'
]
