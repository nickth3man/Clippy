"""
Tests for the Diarizer class in the core module.
"""
import os
import unittest
from pathlib import Path
import tempfile
import numpy as np
import soundfile as sf
from unittest.mock import patch, MagicMock

from app.core import Diarizer, DiarizationError, SpeakerSegment, AudioProcessor


class TestDiarizer(unittest.TestCase):
    """Test cases for the Diarizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the model loading to avoid actual model downloads
        with patch('app.core.diarization.Diarizer._load_model'):
            self.diarizer = Diarizer(device="cpu")
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        
        # Create a test audio file (simulated speech)
        self.sample_rate = 16000
        duration = 5.0  # seconds
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        
        # Create a signal to simulate speech
        self.test_audio = (
            0.5 * np.sin(2 * np.pi * 200 * t) +  # Fundamental frequency
            0.3 * np.sin(2 * np.pi * 400 * t)    # First harmonic
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
        """Test that the diarizer initializes correctly."""
        with patch('app.core.diarization.Diarizer._load_model') as mock_load:
            diarizer = Diarizer(device="cpu")
            mock_load.assert_called_once()
            self.assertEqual(diarizer.device, "cpu")
    
    def test_diarize_mock(self):
        """Test diarization with mocked model."""
        # Load the test audio
        audio, sr = self.audio_processor.load_audio(self.audio_path)
        
        # Mock the actual diarization function
        with patch.object(self.diarizer, '_diarize') as mock_diarize:
            # Create mock diarization segments (2 speakers)
            mock_segments = [
                SpeakerSegment(start=0.0, end=2.5, speaker="speaker1"),
                SpeakerSegment(start=2.5, end=5.0, speaker="speaker2")
            ]
            mock_diarize.return_value = mock_segments
            
            # Call the diarize method
            result = self.diarizer.diarize(audio, sr)
            
            # Verify the result
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0].start, 0.0)
            self.assertEqual(result[0].end, 2.5)
            self.assertEqual(result[0].speaker, "speaker1")
            self.assertEqual(result[1].start, 2.5)
            self.assertEqual(result[1].end, 5.0)
            self.assertEqual(result[1].speaker, "speaker2")
            
            # Verify the mock was called with the right arguments
            mock_diarize.assert_called_once()
    
    def test_diarize_with_num_speakers(self):
        """Test diarization with specified number of speakers."""
        # Load the test audio
        audio, sr = self.audio_processor.load_audio(self.audio_path)
        
        # Mock the diarization function
        with patch.object(self.diarizer, '_diarize') as mock_diarize:
            mock_segments = [
                SpeakerSegment(start=0.0, end=2.5, speaker="speaker1"),
                SpeakerSegment(start=2.5, end=5.0, speaker="speaker2")
            ]
            mock_diarize.return_value = mock_segments
            
            # Call the diarize method with num_speakers=2
            result = self.diarizer.diarize(audio, sr, num_speakers=2)
            
            # Verify the mock was called with num_speakers=2
            mock_diarize.assert_called_once()
            # Extract the arguments the mock was called with
            args, kwargs = mock_diarize.call_args
            self.assertEqual(kwargs.get('num_speakers'), 2)
    
    def test_diarize_with_min_max_speakers(self):
        """Test diarization with min and max speakers."""
        # Load the test audio
        audio, sr = self.audio_processor.load_audio(self.audio_path)
        
        # Mock the diarization function
        with patch.object(self.diarizer, '_diarize') as mock_diarize:
            mock_segments = [
                SpeakerSegment(start=0.0, end=2.5, speaker="speaker1"),
                SpeakerSegment(start=2.5, end=5.0, speaker="speaker2")
            ]
            mock_diarize.return_value = mock_segments
            
            # Call the diarize method with min_speakers=1 and max_speakers=3
            result = self.diarizer.diarize(audio, sr, min_speakers=1, max_speakers=3)
            
            # Verify the mock was called with the right arguments
            mock_diarize.assert_called_once()
            args, kwargs = mock_diarize.call_args
            self.assertEqual(kwargs.get('min_speakers'), 1)
            self.assertEqual(kwargs.get('max_speakers'), 3)
    
    def test_error_handling(self):
        """Test error handling during diarization."""
        # Load the test audio
        audio, sr = self.audio_processor.load_audio(self.audio_path)
        
        # Mock the diarization function to raise an exception
        with patch.object(self.diarizer, '_diarize', side_effect=Exception("Test error")):
            # The method should raise a DiarizationError
            with self.assertRaises(DiarizationError):
                self.diarizer.diarize(audio, sr)
    
    def test_invalid_input(self):
        """Test handling of invalid input."""
        # Test with empty audio
        empty_audio = np.zeros((1, 0), dtype=np.float32)
        with self.assertRaises(DiarizationError):
            self.diarizer.diarize(empty_audio, self.sample_rate)
        
        # Test with invalid sample rate
        with self.assertRaises(DiarizationError):
            self.diarizer.diarize(self.test_audio, -1)
        
        # Test with invalid speaker count
        with self.assertRaises(DiarizationError):
            self.diarizer.diarize(self.test_audio, self.sample_rate, min_speakers=5, max_speakers=2)


if __name__ == '__main__':
    unittest.main() 