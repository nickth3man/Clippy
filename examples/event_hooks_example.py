#!/usr/bin/env python3
"""
Event Hooks Example

This example demonstrates how to use the event hooks system in the Pipeline class
to monitor and extend the processing pipeline.
"""

import os
import sys
import logging
from pathlib import Path
import time

# Add the parent directory to the Python path so we can import the app module
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core import Pipeline, PipelineEvent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def on_processing_start(event_data):
    """
    Handler for the processing start event.
    
    Args:
        event_data: Event data dictionary
    """
    file_path = event_data.get('file_path', 'Unknown')
    logger.info(f"Processing started for file: {os.path.basename(file_path)}")
    logger.info(f"Min speakers: {event_data.get('min_speakers')}")
    logger.info(f"Max speakers: {event_data.get('max_speakers')}")
    
    # Store the start time for later performance calculation
    event_data['custom_start_time'] = time.time()


def on_processing_complete(event_data):
    """
    Handler for the processing complete event.
    
    Args:
        event_data: Event data dictionary
    """
    result_dict = event_data.get('result', {})
    recording_id = result_dict.get('recording_id', 'Unknown')
    processing_time = event_data.get('processing_time', 0)
    
    logger.info(f"Processing completed for recording: {recording_id}")
    logger.info(f"Processing time: {processing_time:.2f} seconds")
    
    # Calculate custom processing metrics
    start_time = event_data.get('custom_start_time', 0)
    if start_time:
        total_time = time.time() - start_time
        overhead = total_time - processing_time
        logger.info(f"Total time including overhead: {total_time:.2f} seconds")
        logger.info(f"Pipeline overhead: {overhead:.2f} seconds ({(overhead/total_time)*100:.1f}%)")


def on_voice_separation_complete(event_data):
    """
    Handler for the voice separation complete event.
    
    Args:
        event_data: Event data dictionary
    """
    result_dict = event_data.get('result', {})
    recording_id = result_dict.get('recording_id', 'Unknown')
    
    # Count the number of separated voices
    separated_voices = event_data.get('separated_voices', [])
    voice_count = len(separated_voices)
    
    logger.info(f"Voice separation completed for recording: {recording_id}")
    logger.info(f"Number of separated voices: {voice_count}")
    
    # Log some statistics about the separated voices
    for i, voice in enumerate(separated_voices):
        if voice:  # Check if the voice data exists
            duration = len(voice) / 16000  # Assuming 16kHz sample rate
            logger.info(f"Voice {i+1}: {duration:.2f} seconds")


def on_error(event_data):
    """
    Handler for the processing error event.
    
    Args:
        event_data: Event data dictionary
    """
    error = event_data.get('error', 'Unknown error')
    exception = event_data.get('exception', 'No exception details')
    
    logger.error(f"Processing error: {error}")
    logger.error(f"Exception: {exception}")
    
    # Here you could implement custom error handling, such as:
    # - Sending notifications
    # - Logging to a database
    # - Attempting recovery
    # - etc.


def progress_callback(message, progress):
    """
    Progress callback for the pipeline.
    
    Args:
        message: Progress message
        progress: Progress value (0.0 to 1.0)
    """
    bar_length = 30
    filled_length = int(bar_length * progress)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    percent = int(progress * 100)
    
    print(f"\r{message}: [{bar}] {percent}%", end='', flush=True)
    
    if progress >= 1.0:
        print()


def main():
    """Main function demonstrating event hooks usage."""
    
    # Create a Pipeline instance
    pipeline = Pipeline(
        models_dir="models",
        db_path="clippy.db",
        device="cpu"
    )
    
    # Register event handlers
    pipeline.register_event_handler(PipelineEvent.PROCESSING_START, on_processing_start)
    pipeline.register_event_handler(PipelineEvent.PROCESSING_COMPLETE, on_processing_complete)
    pipeline.register_event_handler(PipelineEvent.VOICE_SEPARATION_COMPLETE, on_voice_separation_complete)
    pipeline.register_event_handler(PipelineEvent.PROCESSING_ERROR, on_error)
    
    # Check if a file path was provided
    if len(sys.argv) < 2:
        print("Usage: python event_hooks_example.py <audio_file>")
        sys.exit(1)
    
    # Get the audio file path
    audio_file = sys.argv[1]
    
    # Process the audio file
    print(f"Processing audio file: {audio_file}")
    result = pipeline.process_recording(
        file_path=audio_file,
        min_speakers=None,
        max_speakers=None,
        progress_callback=progress_callback
    )
    
    # Print the result
    print("\nProcessing result:")
    print(f"Recording ID: {result.recording_id}")
    print(f"Duration: {result.duration:.2f} seconds")
    print(f"Processing time: {result.processing_time:.2f} seconds")
    print(f"Speakers: {len(result.speakers)}")
    
    # Close the pipeline
    pipeline.close()


if __name__ == "__main__":
    main() 