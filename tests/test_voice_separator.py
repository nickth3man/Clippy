"""
Tests for the VoiceSeparator class in the core module.
"""
import os
import unittest
from pathlib import Path
import tempfile
import numpy as np
import soundfile as sf
from unittest.mock import patch, MagicMock

from app.core import VoiceSeparator, VoiceSeparationError, AudioProcessor


class TestVoiceSeparator(unittest.TestCase):
    """Test cases for the VoiceSeparator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the model loading to avoid actual model downloads
        with patch('app.core.voice_separator.VoiceSeparator._load_model'):
            self.separator = VoiceSeparator(device="cpu")
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        
        # Create a test audio file (simulated speech)
        self.sample_rate = 16000
        duration = 2.0  # seconds
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        
        # Create a more complex signal to simulate speech (multiple frequencies)
        self.test_audio = (
            0.5 * np.sin(2 * np.pi * 200 * t) +  # Fundamental frequency
            0.3 * np.sin(2 * np.pi * 400 * t) +  # First harmonic
            0.2 * np.sin(2 * np.pi * 600 * t)    # Second harmonic
        ).reshape(1, -1).astype(np.float32)
        
        # Create test file
        self.audio_path = self.test_dir / "test_speech.wav"
        sf.write(self.audio_path, self.test_audio[0], self.sample_rate)
        
        # Create an audio processor for loading audio
        self.audio_processor = AudioProcessor()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test that the separator initializes correctly."""
        with patch('app.core.voice_separator.VoiceSeparator._load_model') as mock_load:
            separator = VoiceSeparator(device="cpu")
            mock_load.assert_called_once()
            self.assertEqual(separator.device, "cpu")
    
    def test_separate_voices_mock(self):
        """Test voice separation with mocked model."""
        # Load the test audio
        audio, sr = self.audio_processor.load_audio(self.audio_path)
        
        # Mock the actual separation function
        with patch.object(self.separator, '_separate_voices') as mock_separate:
            # Create mock separated voices (2 speakers)
            mock_voices = [
                audio * 0.7,  # First speaker
                audio * 0.3   # Second speaker
            ]
            mock_separate.return_value = mock_voices
            
            # Call the separate_voices method
            result = self.separator.separate_voices(audio, sr)
            
            # Verify the result
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0].shape, audio.shape)
            self.assertEqual(result[1].shape, audio.shape)
            
            # Verify the mock was called with the right arguments
            mock_separate.assert_called_once()
    
    def test_separate_voices_with_segments(self):
        """Test voice separation with speaker segments."""
        # Load the test audio
        audio, sr = self.audio_processor.load_audio(self.audio_path)
        
        # Create mock speaker segments
        segments = [
            {"start": 0.0, "end": 1.0, "speaker": "speaker1"},
            {"start": 1.0, "end": 2.0, "speaker": "speaker2"}
        ]
        
        # Mock the segment-based separation function
        with patch.object(self.separator, '_separate_voices_with_segments') as mock_separate:
            # Create mock separated voices
            mock_voices = {
                "speaker1": audio[:, :sr] * 0.7,  # First speaker (first half)
                "speaker2": audio[:, sr:] * 0.3   # Second speaker (second half)
            }
            mock_separate.return_value = mock_voices
            
            # Call the separate_voices_with_segments method
            result = self.separator.separate_voices_with_segments(audio, sr, segments)
            
            # Verify the result
            self.assertEqual(len(result), 2)
            self.assertIn("speaker1", result)
            self.assertIn("speaker2", result)
            
            # Verify the mock was called with the right arguments
            mock_separate.assert_called_once()
    
    def test_error_handling(self):
        """Test error handling during voice separation."""
        # Load the test audio
        audio, sr = self.audio_processor.load_audio(self.audio_path)
        
        # Mock the separation function to raise an exception
        with patch.object(self.separator, '_separate_voices', side_effect=Exception("Test error")):
            # The method should raise a VoiceSeparationError
            with self.assertRaises(VoiceSeparationError):
                self.separator.separate_voices(audio, sr)
    
    def test_invalid_input(self):
        """Test handling of invalid input."""
        # Test with empty audio
        empty_audio = np.zeros((1, 0), dtype=np.float32)
        with self.assertRaises(VoiceSeparationError):
            self.separator.separate_voices(empty_audio, self.sample_rate)
        
        # Test with invalid sample rate
        with self.assertRaises(VoiceSeparationError):
            self.separator.separate_voices(self.test_audio, -1)


if __name__ == '__main__':
    unittest.main() 