"""
Tests for the EmbeddingProcessor class in the core module.
"""
import os
import unittest
from pathlib import Path
import tempfile
import numpy as np
import soundfile as sf
from unittest.mock import patch, MagicMock

from app.core import EmbeddingProcessor, SpeakerEmbedding, EmbeddingError, AudioProcessor


class TestEmbeddingProcessor(unittest.TestCase):
    """Test cases for the EmbeddingProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the model loading to avoid actual model downloads
        with patch('app.core.embedding.EmbeddingProcessor._load_model'):
            self.embedding_processor = EmbeddingProcessor(device="cpu")
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        
        # Create a test audio file (simulated speech)
        self.sample_rate = 16000
        duration = 3.0  # seconds
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
        """Test that the embedding processor initializes correctly."""
        with patch('app.core.embedding.EmbeddingProcessor._load_model') as mock_load:
            processor = EmbeddingProcessor(device="cpu")
            mock_load.assert_called_once()
            self.assertEqual(processor.device, "cpu")
    
    def test_extract_embedding_mock(self):
        """Test embedding extraction with mocked model."""
        # Load the test audio
        audio, sr = self.audio_processor.load_audio(self.audio_path)
        
        # Mock the actual embedding extraction function
        with patch.object(self.embedding_processor, '_extract_embedding') as mock_extract:
            # Create a mock embedding (512-dimensional vector)
            mock_embedding = np.random.rand(512).astype(np.float32)
            mock_extract.return_value = mock_embedding
            
            # Call the extract_embedding method
            result = self.embedding_processor.extract_embedding(audio, sr)
            
            # Verify the result
            self.assertIsInstance(result, SpeakerEmbedding)
            self.assertEqual(result.vector.shape, (512,))
            
            # Verify the mock was called with the right arguments
            mock_extract.assert_called_once()
    
    def test_extract_embedding_from_segments(self):
        """Test embedding extraction from multiple segments."""
        # Load the test audio
        audio, sr = self.audio_processor.load_audio(self.audio_path)
        
        # Create segments (first half and second half of the audio)
        segments = [
            {"start": 0.0, "end": 1.5, "audio": audio[:, :int(sr * 1.5)]},
            {"start": 1.5, "end": 3.0, "audio": audio[:, int(sr * 1.5):]}
        ]
        
        # Mock the extract_embedding method
        with patch.object(self.embedding_processor, 'extract_embedding') as mock_extract:
            # Create mock embeddings
            mock_embeddings = [
                SpeakerEmbedding(np.random.rand(512).astype(np.float32)),
                SpeakerEmbedding(np.random.rand(512).astype(np.float32))
            ]
            mock_extract.side_effect = mock_embeddings
            
            # Call the extract_embeddings_from_segments method
            result = self.embedding_processor.extract_embeddings_from_segments(segments, sr)
            
            # Verify the result
            self.assertEqual(len(result), 2)
            self.assertIsInstance(result[0], SpeakerEmbedding)
            self.assertIsInstance(result[1], SpeakerEmbedding)
            
            # Verify the mock was called twice
            self.assertEqual(mock_extract.call_count, 2)
    
    def test_compare_embeddings(self):
        """Test comparing two embeddings."""
        # Create two random embeddings
        embedding1 = SpeakerEmbedding(np.random.rand(512).astype(np.float32))
        embedding2 = SpeakerEmbedding(np.random.rand(512).astype(np.float32))
        
        # Mock the _compute_similarity method
        with patch.object(self.embedding_processor, '_compute_similarity', return_value=0.85):
            # Call the compare_embeddings method
            similarity = self.embedding_processor.compare_embeddings(embedding1, embedding2)
            
            # Verify the result
            self.assertEqual(similarity, 0.85)
    
    def test_error_handling(self):
        """Test error handling during embedding extraction."""
        # Load the test audio
        audio, sr = self.audio_processor.load_audio(self.audio_path)
        
        # Mock the embedding extraction function to raise an exception
        with patch.object(self.embedding_processor, '_extract_embedding', side_effect=Exception("Test error")):
            # The method should raise an EmbeddingError
            with self.assertRaises(EmbeddingError):
                self.embedding_processor.extract_embedding(audio, sr)
    
    def test_invalid_input(self):
        """Test handling of invalid input."""
        # Test with empty audio
        empty_audio = np.zeros((1, 0), dtype=np.float32)
        with self.assertRaises(EmbeddingError):
            self.embedding_processor.extract_embedding(empty_audio, self.sample_rate)
        
        # Test with audio that's too short
        short_audio = np.zeros((1, 100), dtype=np.float32)  # Very short audio
        with self.assertRaises(EmbeddingError):
            self.embedding_processor.extract_embedding(short_audio, self.sample_rate)
        
        # Test with invalid sample rate
        with self.assertRaises(EmbeddingError):
            self.embedding_processor.extract_embedding(self.test_audio, -1)
    
    def test_embedding_serialization(self):
        """Test serialization and deserialization of embeddings."""
        # Create a random embedding
        original_vector = np.random.rand(512).astype(np.float32)
        embedding = SpeakerEmbedding(original_vector)
        
        # Serialize the embedding
        serialized = embedding.to_bytes()
        
        # Deserialize the embedding
        deserialized = SpeakerEmbedding.from_bytes(serialized)
        
        # Verify the deserialized embedding matches the original
        self.assertTrue(np.array_equal(deserialized.vector, original_vector))


if __name__ == '__main__':
    unittest.main() 