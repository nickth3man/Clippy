"""
Core Audio Processing Module

This module provides platform-independent audio loading, validation, quality checks,
and streaming capabilities for processing audio files.

Key features:
1. Platform-independent audio loading with multiple library fallbacks
2. Audio validation and quality checks
3. Streaming capabilities for efficient processing of large files
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Generator, BinaryIO
from pathlib import Path
import tempfile

import numpy as np
import torch
import torchaudio
import soundfile as sf
import librosa
import pyloudnorm as pyln

# Set up logging
logger = logging.getLogger(__name__)

# Define constants
SUPPORTED_FORMATS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']
DEFAULT_SAMPLE_RATE = 16000  # 16kHz is standard for most speech processing
MIN_DURATION_SEC = 0.1  # Minimum valid audio duration in seconds
MAX_FILE_SIZE_GB = 2  # Maximum file size in GB before switching to streaming mode
STREAMING_CHUNK_SIZE = 4 * 16000  # Chunk size for streaming (4 seconds at 16kHz)
SILENCE_THRESHOLD_DB = -60  # Threshold for silence detection in dB


class AudioValidationError(Exception):
    """Exception raised for audio validation errors."""
    pass


class AudioProcessor:
    """
    Handles platform-independent audio loading, validation, and streaming.
    
    Provides a unified interface for audio processing regardless of platform,
    with multiple library fallbacks to ensure compatibility.
    """
    
    def __init__(self, target_sr: int = DEFAULT_SAMPLE_RATE):
        """
        Initialize the AudioProcessor.
        
        Args:
            target_sr: Target sample rate for all audio processing
        """
        self.target_sr = target_sr
        self.meter = pyln.Meter(target_sr)  # Loudness meter
    
    def load_audio(self, 
                   file_path: Union[str, Path], 
                   normalize: bool = True,
                   validate: bool = True) -> Tuple[np.ndarray, int]:
        """
        Load audio file with platform-independent fallbacks.
        
        Args:
            file_path: Path to the audio file
            normalize: Whether to normalize audio to -23 LUFS
            validate: Whether to perform validation checks
            
        Returns:
            Tuple of (audio_data, sample_rate)
            
        Raises:
            AudioValidationError: If validation fails and validate=True
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is unsupported
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
            
        if file_path.suffix.lower() not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported audio format: {file_path.suffix}. "
                             f"Supported formats: {', '.join(SUPPORTED_FORMATS)}")
        
        # Check file size to determine if streaming is needed
        file_size_gb = file_path.stat().st_size / (1024 ** 3)
        if file_size_gb > MAX_FILE_SIZE_GB:
            logger.info(f"Large file detected ({file_size_gb:.2f} GB). Using streaming loader.")
            audio, sr = self._load_audio_streaming(file_path)
        else:
            # Try multiple loading methods with fallbacks
            try:
                # First try torchaudio (fastest and most consistent across platforms)
                try:
                    audio, sr = self._load_with_torchaudio(file_path)
                except Exception as e:
                    logger.debug(f"torchaudio loading failed: {e}, trying soundfile...")
                    
                    # Try soundfile next
                    try:
                        audio, sr = self._load_with_soundfile(file_path)
                    except Exception as e:
                        logger.debug(f"soundfile loading failed: {e}, trying librosa...")
                        
                        # Finally try librosa (slowest but most robust)
                        audio, sr = self._load_with_librosa(file_path)
            
            except Exception as e:
                logger.error(f"All audio loading methods failed for {file_path}: {e}")
                raise RuntimeError(f"Failed to load audio file: {file_path}") from e
        
        # Resample if needed
        if sr != self.target_sr:
            logger.debug(f"Resampling audio from {sr}Hz to {self.target_sr}Hz")
            if isinstance(audio, torch.Tensor):
                audio = audio.numpy()
            audio = librosa.resample(
                y=audio, 
                orig_sr=sr, 
                target_sr=self.target_sr
            )
            sr = self.target_sr
        
        # Ensure mono audio
        if len(audio.shape) > 1 and audio.shape[0] > 1:
            logger.debug(f"Converting audio from {audio.shape[0]} channels to mono")
            if isinstance(audio, torch.Tensor):
                audio = torch.mean(audio, dim=0).numpy()
            else:
                audio = np.mean(audio, axis=0)
        
        # Ensure correct shape
        if len(audio.shape) == 1:
            audio = audio.reshape(1, -1)  # Add channel dimension if needed
        
        # Validate audio if requested
        if validate:
            self._validate_audio(audio, sr, file_path)
        
        # Normalize audio if requested
        if normalize:
            audio = self._normalize_audio(audio)
            
        return audio, sr
    
    def create_audio_stream(self, 
                           file_path: Union[str, Path],
                           chunk_duration: float = 4.0) -> Generator[np.ndarray, None, None]:
        """
        Create a generator that streams audio in chunks for memory-efficient processing.
        
        Args:
            file_path: Path to the audio file
            chunk_duration: Duration of each chunk in seconds
            
        Yields:
            Chunks of audio data as numpy arrays
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is unsupported
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
            
        if file_path.suffix.lower() not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported audio format: {file_path.suffix}. "
                             f"Supported formats: {', '.join(SUPPORTED_FORMATS)}")
            
        # Calculate chunk size in samples
        chunk_size = int(self.target_sr * chunk_duration)
        
        # Open the file
        with sf.SoundFile(file_path) as sound_file:
            # Calculate resampling ratio if needed
            resample_ratio = self.target_sr / sound_file.samplerate
            actual_chunk_size = int(chunk_size / resample_ratio)
            
            # Stream chunks
            while sound_file.tell() < sound_file.frames:
                chunk = sound_file.read(actual_chunk_size)
                
                # Convert to mono if needed
                if sound_file.channels > 1:
                    chunk = np.mean(chunk, axis=1)
                
                # Resample if needed
                if sound_file.samplerate != self.target_sr:
                    chunk = librosa.resample(
                        y=chunk, 
                        orig_sr=sound_file.samplerate, 
                        target_sr=self.target_sr
                    )
                
                # Reshape for consistent output
                chunk = chunk.reshape(1, -1)
                
                yield chunk
    
    def get_audio_metadata(self, file_path: Union[str, Path]) -> Dict:
        """
        Get metadata about an audio file without loading the full content.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary with metadata (sample_rate, channels, duration, etc.)
            
        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        try:
            with sf.SoundFile(file_path) as sound_file:
                return {
                    'sample_rate': sound_file.samplerate,
                    'channels': sound_file.channels,
                    'duration': sound_file.frames / sound_file.samplerate,
                    'format': sound_file.format,
                    'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                }
        except Exception as e:
            logger.debug(f"Error getting metadata with soundfile: {e}, trying librosa...")
            
            try:
                info = librosa.get_duration(path=str(file_path)), librosa.get_samplerate(path=str(file_path))
                return {
                    'sample_rate': info[1],
                    'channels': None,  # Librosa doesn't provide this easily
                    'duration': info[0],
                    'format': file_path.suffix.replace('.', ''),
                    'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                }
            except Exception as e2:
                logger.error(f"Failed to get audio metadata: {e2}")
                raise RuntimeError(f"Could not read audio metadata from {file_path}") from e2
    
    def check_audio_quality(self, audio: np.ndarray, sr: int) -> Dict:
        """
        Perform quality checks on audio data.
        
        Args:
            audio: Audio data as numpy array
            sr: Sample rate
            
        Returns:
            Dictionary with quality metrics
        """
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
            
        # Ensure proper shape for analysis
        if len(audio.shape) > 1:
            audio_for_analysis = audio[0]
        else:
            audio_for_analysis = audio
            
        # Calculate loudness using pyln
        try:
            loudness = self.meter.integrated_loudness(audio_for_analysis)
        except Exception as e:
            logger.warning(f"Could not calculate loudness: {e}")
            loudness = None
            
        # Calculate peak amplitude
        peak = np.max(np.abs(audio_for_analysis))
        
        # Calculate dynamic range (difference between RMS and peak in dB)
        rms = np.sqrt(np.mean(audio_for_analysis**2))
        if rms > 0 and peak > 0:
            dynamic_range_db = 20 * np.log10(peak / rms)
        else:
            dynamic_range_db = 0
            
        # Detect clipping
        clipping_samples = np.sum(np.abs(audio_for_analysis) > 0.99)
        clipping_percentage = clipping_samples / len(audio_for_analysis) * 100
        
        # Calculate signal-to-noise ratio (approximation)
        # This is a simple estimation and might not be accurate for all cases
        try:
            signal_segments = librosa.effects.split(
                audio_for_analysis, 
                top_db=-SILENCE_THRESHOLD_DB,
                frame_length=1024,
                hop_length=256
            )
            
            if len(signal_segments) > 0:
                noise_mask = np.ones_like(audio_for_analysis, dtype=bool)
                for start, end in signal_segments:
                    noise_mask[start:end] = False
                    
                signal_energy = np.mean(audio_for_analysis[~noise_mask]**2) if np.any(~noise_mask) else 0
                noise_energy = np.mean(audio_for_analysis[noise_mask]**2) if np.any(noise_mask) else 0
                
                if noise_energy > 0 and signal_energy > 0:
                    snr = 10 * np.log10(signal_energy / noise_energy)
                else:
                    snr = float('inf')
            else:
                snr = 0
        except Exception as e:
            logger.warning(f"Could not calculate SNR: {e}")
            snr = None
            
        return {
            'loudness_lufs': loudness,
            'peak_amplitude': float(peak),
            'dynamic_range_db': float(dynamic_range_db),
            'clipping_percentage': float(clipping_percentage),
            'signal_to_noise_ratio_db': snr,
            'duration_seconds': len(audio_for_analysis) / sr,
        }
    
    def _load_with_torchaudio(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """Load audio using torchaudio."""
        waveform, sample_rate = torchaudio.load(str(file_path))
        return waveform.numpy(), sample_rate
    
    def _load_with_soundfile(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """Load audio using soundfile."""
        audio, sample_rate = sf.read(str(file_path))
        # Reshape to [channels, samples] to match torchaudio format
        if len(audio.shape) == 1:
            audio = audio.reshape(1, -1)
        else:
            audio = audio.T
        return audio, sample_rate
    
    def _load_with_librosa(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """Load audio using librosa."""
        audio, sample_rate = librosa.load(str(file_path), sr=None, mono=False)
        if len(audio.shape) == 1:
            audio = audio.reshape(1, -1)
        return audio, sample_rate
    
    def _load_audio_streaming(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """
        Load large audio files in a streaming fashion to avoid memory issues.
        This is used for files exceeding MAX_FILE_SIZE_GB.
        """
        # Get file info without loading data
        with sf.SoundFile(file_path) as sound_file:
            sr = sound_file.samplerate
            channels = sound_file.channels
            frames = sound_file.frames
            
        # Determine chunk size based on available memory and file size
        chunk_frames = STREAMING_CHUNK_SIZE
        
        # Preallocate buffer for resampled audio
        target_frames = int(frames * self.target_sr / sr)
        full_audio = np.zeros((1, target_frames), dtype=np.float32)
        
        # Stream in chunks
        position = 0
        with sf.SoundFile(file_path) as sound_file:
            while position < target_frames:
                # Read a chunk
                chunk = sound_file.read(chunk_frames)
                
                # Convert to mono if necessary
                if channels > 1:
                    chunk = np.mean(chunk, axis=1)
                
                # Resample chunk if necessary
                if sr != self.target_sr:
                    chunk = librosa.resample(
                        y=chunk, 
                        orig_sr=sr, 
                        target_sr=self.target_sr
                    )
                
                # Calculate how much of the buffer to fill
                chunk_len = len(chunk)
                if position + chunk_len > target_frames:
                    chunk_len = target_frames - position
                
                # Add chunk to buffer
                full_audio[0, position:position+chunk_len] = chunk[:chunk_len]
                position += chunk_len
                
                # Check if we've reached the end
                if chunk_len < len(chunk) or sound_file.tell() >= sound_file.frames:
                    break
                    
        return full_audio, self.target_sr
    
    def _validate_audio(self, audio: np.ndarray, sr: int, file_path: Path) -> None:
        """
        Validate audio data for common issues.
        
        Raises:
            AudioValidationError: If validation fails
        """
        if audio.size == 0:
            raise AudioValidationError(f"Audio file {file_path} contains no data")
        
        duration = audio.shape[1] / sr
        if duration < MIN_DURATION_SEC:
            raise AudioValidationError(
                f"Audio duration ({duration:.2f}s) is too short (min: {MIN_DURATION_SEC}s)"
            )
        
        # Check for NaN or Inf values
        if np.isnan(audio).any() or np.isinf(audio).any():
            raise AudioValidationError(f"Audio file {file_path} contains NaN or Inf values")
        
        # Check if audio is completely silent
        if np.max(np.abs(audio)) < 1e-6:
            raise AudioValidationError(f"Audio file {file_path} appears to be silent")
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to target loudness (-23 LUFS is EBU R128 standard).
        """
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
            
        # Ensure proper shape for pyln
        if len(audio.shape) > 1:
            audio_for_analysis = audio[0]
        else:
            audio_for_analysis = audio
            
        try:
            # Measure current loudness
            current_loudness = self.meter.integrated_loudness(audio_for_analysis)
            
            # Calculate the gain needed to reach target loudness
            target_loudness = -23.0  # LUFS (EBU R128 standard)
            gain_db = target_loudness - current_loudness
            
            # Apply gain
            gain_linear = 10 ** (gain_db / 20.0)
            normalized_audio = audio * gain_linear
            
            return normalized_audio
        except Exception as e:
            logger.warning(f"Audio normalization failed: {e}. Returning original audio.")
            return audio 