"""
Core functionality for the Clippy application.

This package contains the core modules for audio processing, voice separation,
and other fundamental functionality of the application.
"""

from app.core.audio_processor import AudioProcessor, AudioValidationError

__all__ = ['AudioProcessor', 'AudioValidationError']
