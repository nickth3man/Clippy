"""
Tests for the AudioProcessor class in the core module.
"""
import os
import unittest
from pathlib import Path
import tempfile
import numpy as np
import soundfile as sf

from app.core import AudioProcessor, AudioValidationError

class TestAudioProcessor(unittest.TestCase):
    """Test cases for the AudioProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = AudioProcessor()
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        
        # Create a test sine wave (1kHz tone at 16kHz sample rate)
        self.sample_rate = 16000
        duration = 1.0  # seconds
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        self.test_audio = np.sin(2 * np.pi * 1000 * t).reshape(1, -1).astype(np.float32)
        
        # Create test files in different formats
        self.wav_path = self.test_dir / "test.wav"
        sf.write(self.wav_path, self.test_audio[0], self.sample_rate)
        
        # Create a stereo file for testing multi-channel handling
        stereo_audio = np.vstack([self.test_audio, self.test_audio * 0.8])
        self.stereo_path = self.test_dir / "stereo.wav"
        sf.write(self.stereo_path, stereo_audio.T, self.sample_rate)
        
        # Create a silent file for validation testing
        self.silent_path = self.test_dir / "silent.wav"
        silent_audio = np.zeros_like(self.test_audio[0])
        sf.write(self.silent_path, silent_audio, self.sample_rate)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    def test_load_audio(self):
        """Test basic audio loading functionality."""
        audio, sr = self.processor.load_audio(self.wav_path, normalize=False)
        
        self.assertEqual(sr, self.sample_rate)
        self.assertEqual(audio.shape[0], 1)  # Should be mono
        self.assertEqual(audio.shape[1], self.test_audio.shape[1])
        self.assertTrue(np.allclose(audio, self.test_audio, rtol=1e-4, atol=1e-4))
    
    def test_stereo_to_mono_conversion(self):
        """Test that stereo files are properly converted to mono."""
        audio, sr = self.processor.load_audio(self.stereo_path, normalize=False)
        
        self.assertEqual(sr, self.sample_rate)
        self.assertEqual(audio.shape[0], 1)  # Should be converted to mono
        
        # The resulting audio should be the average of the two channels
        expected = np.mean(np.vstack([self.test_audio, self.test_audio * 0.8]), axis=0).reshape(1, -1)
        self.assertTrue(np.allclose(audio, expected, rtol=1e-4, atol=1e-4))
    
    def test_audio_validation(self):
        """Test that validation correctly identifies problematic audio."""
        # Silent audio should fail validation
        with self.assertRaises(AudioValidationError):
            self.processor.load_audio(self.silent_path, validate=True)
        
        # But should load if validation is disabled
        audio, sr = self.processor.load_audio(self.silent_path, validate=False)
        self.assertEqual(audio.shape[1], self.test_audio.shape[1])
    
    def test_normalize_audio(self):
        """Test that normalization adjusts audio levels correctly."""
        # Create quiet audio
        quiet_path = self.test_dir / "quiet.wav"
        quiet_audio = self.test_audio * 0.1
        sf.write(quiet_path, quiet_audio[0], self.sample_rate)
        
        # Load with normalization
        audio_norm, _ = self.processor.load_audio(quiet_path, normalize=True)
        audio_no_norm, _ = self.processor.load_audio(quiet_path, normalize=False)
        
        # Normalized audio should be louder
        self.assertTrue(np.max(np.abs(audio_norm)) > np.max(np.abs(audio_no_norm)))
    
    def test_get_metadata(self):
        """Test metadata extraction functionality."""
        metadata = self.processor.get_audio_metadata(self.wav_path)
        
        self.assertEqual(metadata['sample_rate'], self.sample_rate)
        self.assertEqual(metadata['channels'], 1)
        self.assertAlmostEqual(metadata['duration'], 1.0, places=2)
    
    def test_check_audio_quality(self):
        """Test audio quality assessment functionality."""
        audio, sr = self.processor.load_audio(self.wav_path, normalize=False)
        quality = self.processor.check_audio_quality(audio, sr)
        
        # Basic checks that the metrics are present and reasonable
        self.assertIn('loudness_lufs', quality)
        self.assertIn('peak_amplitude', quality)
        self.assertIn('dynamic_range_db', quality)
        self.assertIn('clipping_percentage', quality)
        self.assertIn('signal_to_noise_ratio_db', quality)
        self.assertIn('duration_seconds', quality)
        
        # Peak amplitude should be close to 1.0 for a sine wave
        self.assertAlmostEqual(quality['peak_amplitude'], 1.0, places=1)
    
    def test_audio_streaming(self):
        """Test that audio streaming works correctly."""
        # Use the create_audio_stream method to get chunks
        chunks = list(self.processor.create_audio_stream(self.wav_path, chunk_duration=0.25))
        
        # We should get 4 chunks of 0.25 seconds each
        self.assertEqual(len(chunks), 4)
        
        # Each chunk should be the correct size
        expected_chunk_size = int(self.sample_rate * 0.25)
        for chunk in chunks:
            self.assertEqual(chunk.shape[1], expected_chunk_size)
        
        # Concatenating all chunks should be approximately equal to the original audio
        concatenated = np.hstack(chunks)
        self.assertTrue(np.allclose(concatenated, self.test_audio, rtol=1e-4, atol=1e-4))


if __name__ == '__main__':
    unittest.main() 