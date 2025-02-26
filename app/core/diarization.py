"""
Diarization Module

This module implements speaker diarization functionality using WhisperX.
It provides capabilities to identify speaker segments in audio recordings.

Key features:
1. Speaker diarization using WhisperX
2. Speaker embedding extraction
3. Integration with voice separation
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import tempfile

import numpy as np
import torch
import librosa

# Import WhisperX specific modules
try:
    import whisperx
    from pyannote.audio import Pipeline
    WHISPERX_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import WhisperX dependencies: {e}")
    WHISPERX_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)

# Define constants
DEFAULT_MODEL_PATH = Path("models/whisperx")
DEFAULT_SAMPLE_RATE = 16000


class DiarizationError(Exception):
    """Exception raised for diarization errors."""
    pass


class SpeakerSegment:
    """
    Represents a segment of speech from a single speaker.
    """
    
    def __init__(self, 
                 start: float, 
                 end: float, 
                 speaker_id: str,
                 confidence: float = 1.0,
                 text: str = None):
        """
        Initialize a speaker segment.
        
        Args:
            start: Start time in seconds
            end: End time in seconds
            speaker_id: Speaker identifier
            confidence: Confidence score (0-1)
            text: Transcribed text (if available)
        """
        self.start = start
        self.end = end
        self.speaker_id = speaker_id
        self.confidence = confidence
        self.text = text
        
    def duration(self) -> float:
        """Get segment duration in seconds."""
        return self.end - self.start
        
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'start': self.start,
            'end': self.end,
            'speaker_id': self.speaker_id,
            'confidence': self.confidence,
            'text': self.text,
            'duration': self.duration()
        }
        
    def __repr__(self) -> str:
        """String representation."""
        text_preview = f'"{self.text[:20]}..."' if self.text and len(self.text) > 20 else f'"{self.text}"'
        return (f"SpeakerSegment(start={self.start:.2f}s, end={self.end:.2f}s, "
                f"duration={self.duration():.2f}s, speaker={self.speaker_id}, "
                f"confidence={self.confidence:.2f}, text={text_preview})")


class Diarizer:
    """
    Handles speaker diarization using WhisperX.
    
    Provides functionality to identify speaker segments in audio recordings.
    """
    
    def __init__(self, 
                 model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
                 device: str = None,
                 sample_rate: int = DEFAULT_SAMPLE_RATE):
        """
        Initialize the Diarizer.
        
        Args:
            model_path: Path to the directory containing the WhisperX models
            device: Device to run the model on ('cuda' or 'cpu')
            sample_rate: Sample rate for processing
        """
        self.model_path = Path(model_path)
        self.sample_rate = sample_rate
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self.asr_model = None
        self.diarization_model = None
        self.available = WHISPERX_AVAILABLE
        
        if self.available:
            self._load_models()
        else:
            logger.warning("WhisperX is not available. Diarization functionality will be limited.")
    
    def _load_models(self):
        """
        Load WhisperX models for ASR and diarization.
        """
        try:
            # Create model directory if it doesn't exist
            self.model_path.mkdir(parents=True, exist_ok=True)
            
            # Load ASR model
            logger.info("Loading WhisperX ASR model")
            self.asr_model = whisperx.load_model(
                "medium", 
                self.device, 
                compute_type="float16" if self.device == "cuda" else "float32",
                download_root=str(self.model_path)
            )
            
            # Load diarization model
            logger.info("Loading diarization model")
            self.diarization_model = whisperx.DiarizationPipeline(
                use_auth_token=None,  # Use local model
                device=self.device,
                download_root=str(self.model_path)
            )
            
            logger.info("WhisperX models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load WhisperX models: {e}")
            self.available = False
    
    def diarize(self, 
                audio_path: Union[str, Path],
                min_speakers: int = None,
                max_speakers: int = None) -> List[SpeakerSegment]:
        """
        Perform diarization on an audio file.
        
        Args:
            audio_path: Path to the audio file
            min_speakers: Minimum number of speakers (optional)
            max_speakers: Maximum number of speakers (optional)
            
        Returns:
            List of speaker segments
            
        Raises:
            DiarizationError: If diarization fails
        """
        if not self.available:
            raise DiarizationError("WhisperX is not available")
            
        if self.asr_model is None or self.diarization_model is None:
            self._load_models()
            
        if self.asr_model is None or self.diarization_model is None:
            raise DiarizationError("Failed to load WhisperX models")
            
        try:
            # Transcribe audio
            logger.info(f"Transcribing audio: {audio_path}")
            audio_path = str(audio_path)  # Convert to string for WhisperX
            result = self.asr_model.transcribe(audio_path, batch_size=16)
            
            # Align whisper output
            logger.info("Aligning transcription")
            result = whisperx.align(
                result["segments"],
                self.diarization_model,
                audio_path,
                self.device,
                return_char_alignments=False
            )
            
            # Diarize audio
            logger.info("Performing diarization")
            diarize_segments = self.diarization_model(
                audio_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
            
            # Assign speaker labels
            logger.info("Assigning speaker labels")
            result = whisperx.assign_word_speakers(diarize_segments, result)
            
            # Convert to SpeakerSegment objects
            segments = []
            for segment in result["segments"]:
                speaker = segment.get("speaker", "UNKNOWN")
                start = segment["start"]
                end = segment["end"]
                text = segment["text"]
                confidence = segment.get("confidence", 1.0)
                
                segments.append(SpeakerSegment(
                    start=start,
                    end=end,
                    speaker_id=speaker,
                    confidence=confidence,
                    text=text
                ))
                
            return segments
            
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            raise DiarizationError(f"Diarization failed: {e}")
    
    def diarize_from_array(self, 
                          audio: np.ndarray,
                          sr: int,
                          min_speakers: int = None,
                          max_speakers: int = None) -> List[SpeakerSegment]:
        """
        Perform diarization on an audio array.
        
        Args:
            audio: Audio data as numpy array
            sr: Sample rate
            min_speakers: Minimum number of speakers (optional)
            max_speakers: Maximum number of speakers (optional)
            
        Returns:
            List of speaker segments
            
        Raises:
            DiarizationError: If diarization fails
        """
        # WhisperX requires a file, so we need to save the array to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            
        try:
            # Ensure audio is in the correct format
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
                
            # Ensure mono audio
            if len(audio.shape) > 1 and audio.shape[0] > 1:
                audio = np.mean(audio, axis=0)
                
            # Flatten if needed
            if len(audio.shape) > 1:
                audio = audio.flatten()
                
            # Save to temporary file
            import soundfile as sf
            sf.write(temp_path, audio, sr)
            
            # Perform diarization
            return self.diarize(temp_path, min_speakers, max_speakers)
            
        except Exception as e:
            logger.error(f"Diarization from array failed: {e}")
            raise DiarizationError(f"Diarization from array failed: {e}")
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_path}: {e}")
    
    def get_speaker_timeline(self, segments: List[SpeakerSegment]) -> Dict[str, List[Tuple[float, float]]]:
        """
        Get a timeline of when each speaker is active.
        
        Args:
            segments: List of speaker segments
            
        Returns:
            Dictionary mapping speaker IDs to lists of (start, end) tuples
        """
        timeline = {}
        
        for segment in segments:
            if segment.speaker_id not in timeline:
                timeline[segment.speaker_id] = []
                
            timeline[segment.speaker_id].append((segment.start, segment.end))
            
        return timeline
    
    def get_speaker_stats(self, segments: List[SpeakerSegment]) -> Dict[str, Dict]:
        """
        Get statistics for each speaker.
        
        Args:
            segments: List of speaker segments
            
        Returns:
            Dictionary mapping speaker IDs to statistics
        """
        stats = {}
        
        for segment in segments:
            if segment.speaker_id not in stats:
                stats[segment.speaker_id] = {
                    'total_duration': 0.0,
                    'segment_count': 0,
                    'word_count': 0,
                    'avg_segment_duration': 0.0
                }
                
            speaker_stats = stats[segment.speaker_id]
            speaker_stats['total_duration'] += segment.duration()
            speaker_stats['segment_count'] += 1
            
            if segment.text:
                speaker_stats['word_count'] += len(segment.text.split())
                
        # Calculate averages
        for speaker_id, speaker_stats in stats.items():
            if speaker_stats['segment_count'] > 0:
                speaker_stats['avg_segment_duration'] = (
                    speaker_stats['total_duration'] / speaker_stats['segment_count']
                )
                
        return stats 