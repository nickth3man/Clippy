"""
Tests for the Pipeline class in the core module.
"""
import os
import unittest
from pathlib import Path
import tempfile
import numpy as np
import soundfile as sf
from unittest.mock import patch, MagicMock

from app.core import (
    Pipeline, PipelineError, ProcessingResult, PipelineEvent,
    AudioProcessor, VoiceSeparator, Diarizer, EmbeddingProcessor,
    ClusteringProcessor, ProfileDatabase, SpeakerSegment, SpeakerEmbedding
)


class TestPipeline(unittest.TestCase):
    """Test cases for the Pipeline class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        
        # Create a test audio file
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
        
        # Create a test database path
        self.db_path = self.test_dir / "test_profiles.db"
        
        # Mock all the component classes
        self.audio_processor_mock = MagicMock(spec=AudioProcessor)
        self.voice_separator_mock = MagicMock(spec=VoiceSeparator)
        self.diarizer_mock = MagicMock(spec=Diarizer)
        self.embedding_processor_mock = MagicMock(spec=EmbeddingProcessor)
        self.clustering_processor_mock = MagicMock(spec=ClusteringProcessor)
        self.profile_db_mock = MagicMock(spec=ProfileDatabase)
        
        # Set up the mocks with appropriate return values
        self.audio_processor_mock.load_audio.return_value = (self.test_audio, self.sample_rate)
        
        # Set up diarizer mock to return speaker segments
        self.diarizer_mock.diarize.return_value = [
            SpeakerSegment(start=0.0, end=1.5, speaker="speaker1"),
            SpeakerSegment(start=1.5, end=3.0, speaker="speaker2")
        ]
        
        # Set up voice separator mock to return separated voices
        self.voice_separator_mock.separate_voices_with_segments.return_value = {
            "speaker1": self.test_audio[:, :int(self.sample_rate * 1.5)],
            "speaker2": self.test_audio[:, int(self.sample_rate * 1.5):]
        }
        
        # Set up embedding processor mock to return embeddings
        self.embedding_processor_mock.extract_embedding.return_value = SpeakerEmbedding(
            np.random.rand(512).astype(np.float32)
        )
        
        # Initialize the pipeline with mocked components
        with patch('app.core.pipeline.AudioProcessor', return_value=self.audio_processor_mock), \
             patch('app.core.pipeline.VoiceSeparator', return_value=self.voice_separator_mock), \
             patch('app.core.pipeline.Diarizer', return_value=self.diarizer_mock), \
             patch('app.core.pipeline.EmbeddingProcessor', return_value=self.embedding_processor_mock), \
             patch('app.core.pipeline.ClusteringProcessor', return_value=self.clustering_processor_mock), \
             patch('app.core.pipeline.ProfileDatabase', return_value=self.profile_db_mock):
            
            self.pipeline = Pipeline(
                models_dir=self.test_dir,
                db_path=self.db_path,
                device="cpu"
            )
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test that the pipeline initializes correctly."""
        # Verify the components were initialized
        self.assertIsNotNone(self.pipeline.audio_processor)
        self.assertIsNotNone(self.pipeline.voice_separator)
        self.assertIsNotNone(self.pipeline.diarizer)
        self.assertIsNotNone(self.pipeline.embedding_processor)
        self.assertIsNotNone(self.pipeline.clustering_processor)
        self.assertIsNotNone(self.pipeline.profile_db)
        
        # Verify the event handlers dictionary was initialized
        self.assertIsInstance(self.pipeline.event_handlers, dict)
    
    def test_process_recording(self):
        """Test the process_recording method."""
        # Process a recording
        result = self.pipeline.process_recording(self.audio_path)
        
        # Verify the result
        self.assertIsInstance(result, ProcessingResult)
        self.assertEqual(result.recording_path, str(self.audio_path))
        
        # Verify the components were called
        self.audio_processor_mock.load_audio.assert_called_once()
        self.diarizer_mock.diarize.assert_called_once()
        self.voice_separator_mock.separate_voices_with_segments.assert_called_once()
        
        # The embedding processor should be called for each speaker
        self.assertEqual(self.embedding_processor_mock.extract_embedding.call_count, 2)
    
    def test_process_recording_with_min_max_speakers(self):
        """Test process_recording with min and max speakers."""
        # Process a recording with min_speakers=1 and max_speakers=3
        result = self.pipeline.process_recording(
            self.audio_path, min_speakers=1, max_speakers=3
        )
        
        # Verify the diarizer was called with the right arguments
        args, kwargs = self.diarizer_mock.diarize.call_args
        self.assertEqual(kwargs.get('min_speakers'), 1)
        self.assertEqual(kwargs.get('max_speakers'), 3)
    
    def test_event_system_integration(self):
        """Test that events are triggered during processing."""
        # Create a mock event handler
        event_handler = MagicMock()
        
        # Register the handler for multiple events
        self.pipeline.register_event_handler(PipelineEvent.PROCESSING_START, event_handler)
        self.pipeline.register_event_handler(PipelineEvent.AUDIO_LOADED, event_handler)
        self.pipeline.register_event_handler(PipelineEvent.DIARIZATION_COMPLETE, event_handler)
        self.pipeline.register_event_handler(PipelineEvent.PROCESSING_COMPLETE, event_handler)
        
        # Process a recording
        result = self.pipeline.process_recording(self.audio_path)
        
        # Verify the event handler was called for each event
        self.assertEqual(event_handler.call_count, 4)
        
        # Verify the event data contains the expected fields
        for call in event_handler.call_args_list:
            event_data = call[0][0]
            self.assertIn('timestamp', event_data)
            self.assertIn('event_type', event_data)
    
    def test_error_handling(self):
        """Test error handling during processing."""
        # Make the audio processor raise an exception
        self.audio_processor_mock.load_audio.side_effect = Exception("Test error")
        
        # Create a mock event handler for the error event
        error_handler = MagicMock()
        self.pipeline.register_event_handler(PipelineEvent.PROCESSING_ERROR, error_handler)
        
        # Process a recording (should handle the error)
        result = self.pipeline.process_recording(self.audio_path)
        
        # Verify the result contains an error
        self.assertTrue(result.has_errors())
        
        # Verify the error event was triggered
        error_handler.assert_called_once()
        event_data = error_handler.call_args[0][0]
        self.assertEqual(event_data['event_type'], PipelineEvent.PROCESSING_ERROR)
        self.assertIn('error', event_data)
    
    def test_profile_operations(self):
        """Test profile operations in the pipeline."""
        # Mock the profile_db to return a profile
        mock_profile = MagicMock()
        mock_profile.id = "test_profile_id"
        mock_profile.name = "Test Profile"
        self.profile_db_mock.create_profile.return_value = mock_profile
        self.profile_db_mock.get_profile.return_value = mock_profile
        
        # Create a profile
        profile = self.pipeline.create_profile(name="Test Profile")
        
        # Verify the profile was created
        self.assertEqual(profile.id, "test_profile_id")
        self.assertEqual(profile.name, "Test Profile")
        
        # Get the profile
        retrieved = self.pipeline.get_profile("test_profile_id")
        
        # Verify the profile was retrieved
        self.assertEqual(retrieved.id, "test_profile_id")
        
        # Delete the profile
        self.profile_db_mock.delete_profile.return_value = True
        result = self.pipeline.delete_profile("test_profile_id")
        
        # Verify the deletion was successful
        self.assertTrue(result)
    
    def test_close(self):
        """Test closing the pipeline."""
        # Close the pipeline
        self.pipeline.close()
        
        # Verify the profile_db was closed
        self.profile_db_mock.close.assert_called_once()


if __name__ == '__main__':
    unittest.main() 