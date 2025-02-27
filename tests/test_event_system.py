"""
Tests for the Pipeline events system.
"""
import unittest
from unittest.mock import MagicMock, patch
import time

from app.core import Pipeline, PipelineEvent


class TestEventSystem(unittest.TestCase):
    """Tests for the event system in the Pipeline class."""
    
    def setUp(self):
        """Set up test fixtures."""
        with patch('app.core.AudioProcessor'), \
             patch('app.core.VoiceSeparator'), \
             patch('app.core.Diarizer'), \
             patch('app.core.EmbeddingProcessor'), \
             patch('app.core.ClusteringProcessor'), \
             patch('app.core.ProfileDatabase'):
            self.pipeline = Pipeline(
                models_dir="models",
                db_path="test.db",
                device="cpu"
            )
    
    def test_register_event_handler(self):
        """Test registering an event handler."""
        handler = MagicMock()
        
        # Register the handler
        self.pipeline.register_event_handler(PipelineEvent.PROCESSING_START, handler)
        
        # Check that the handler was registered
        self.assertIn(PipelineEvent.PROCESSING_START, self.pipeline.event_handlers)
        self.assertIn(handler, self.pipeline.event_handlers[PipelineEvent.PROCESSING_START])
    
    def test_unregister_event_handler(self):
        """Test unregistering an event handler."""
        handler = MagicMock()
        
        # Register the handler
        self.pipeline.register_event_handler(PipelineEvent.PROCESSING_START, handler)
        
        # Unregister the handler
        result = self.pipeline.unregister_event_handler(PipelineEvent.PROCESSING_START, handler)
        
        # Check that the handler was unregistered and the result is True
        self.assertTrue(result)
        self.assertNotIn(handler, self.pipeline.event_handlers[PipelineEvent.PROCESSING_START])
    
    def test_unregister_nonexistent_handler(self):
        """Test unregistering a handler that wasn't registered."""
        handler = MagicMock()
        
        # Try to unregister a handler that wasn't registered
        result = self.pipeline.unregister_event_handler(PipelineEvent.PROCESSING_START, handler)
        
        # Check that the result is False
        self.assertFalse(result)
    
    def test_trigger_event(self):
        """Test triggering an event."""
        # Create a mock handler
        handler = MagicMock()
        
        # Register the handler
        self.pipeline.register_event_handler(PipelineEvent.PROCESSING_START, handler)
        
        # Create event data
        event_data = {'test': True}
        
        # Trigger the event
        self.pipeline.trigger_event(PipelineEvent.PROCESSING_START, event_data)
        
        # Check that the handler was called with the event data
        handler.assert_called_once()
        called_data = handler.call_args[0][0]
        self.assertEqual(called_data['test'], True)
        self.assertIn('timestamp', called_data)
        self.assertIn('event_type', called_data)
        self.assertEqual(called_data['event_type'], PipelineEvent.PROCESSING_START)
    
    def test_trigger_event_no_handlers(self):
        """Test triggering an event with no handlers."""
        # This should not raise an exception
        self.pipeline.trigger_event(PipelineEvent.PROCESSING_START, {'test': True})
    
    def test_trigger_event_handler_exception(self):
        """Test triggering an event where a handler raises an exception."""
        # Create a handler that raises an exception
        handler = MagicMock(side_effect=Exception("Test exception"))
        
        # Register the handler
        self.pipeline.register_event_handler(PipelineEvent.PROCESSING_START, handler)
        
        # This should not raise an exception
        self.pipeline.trigger_event(PipelineEvent.PROCESSING_START, {'test': True})
        
        # The handler should still have been called
        handler.assert_called_once()
    
    def test_multiple_handlers(self):
        """Test triggering an event with multiple handlers."""
        # Create mock handlers
        handler1 = MagicMock()
        handler2 = MagicMock()
        
        # Register the handlers
        self.pipeline.register_event_handler(PipelineEvent.PROCESSING_START, handler1)
        self.pipeline.register_event_handler(PipelineEvent.PROCESSING_START, handler2)
        
        # Trigger the event
        self.pipeline.trigger_event(PipelineEvent.PROCESSING_START, {'test': True})
        
        # Check that both handlers were called
        handler1.assert_called_once()
        handler2.assert_called_once()
    
    def test_event_data_isolation(self):
        """Test that event data is properly isolated between handlers."""
        # Define handlers that modify the event data
        def handler1(data):
            data['handler1'] = True
        
        def handler2(data):
            data['handler2'] = True
            # This should see handler1's modification
            self.assertTrue(data.get('handler1', False))
        
        # Register the handlers
        self.pipeline.register_event_handler(PipelineEvent.PROCESSING_START, handler1)
        self.pipeline.register_event_handler(PipelineEvent.PROCESSING_START, handler2)
        
        # Trigger the event
        original_data = {'test': True}
        self.pipeline.trigger_event(PipelineEvent.PROCESSING_START, original_data)
        
        # The original data should not be modified
        self.assertNotIn('handler1', original_data)
        self.assertNotIn('handler2', original_data)


if __name__ == '__main__':
    unittest.main() 