"""
Voice Separation Module

This module implements voice separation functionality using the MulCat blocks architecture.
It provides capabilities to separate multiple speakers from a single audio recording.

Key features:
1. Model loading from bundled directory
2. Voice separation for known and unknown number of speakers
3. Quality evaluation of separated voices
4. Integration with WhisperX for diarization
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import time

import numpy as np
import torch
import torchaudio
import librosa

# Import SVoice specific modules
try:
    from asteroid_filterbanks import make_enc_dec
    from speechbrain.pretrained import EncoderClassifier
except ImportError as e:
    logging.warning(f"Could not import SVoice dependencies: {e}")

# Set up logging
logger = logging.getLogger(__name__)

# Define constants
DEFAULT_MODEL_PATH = Path("models/svoice")
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_NUM_SPEAKERS = 2
MAX_NUM_SPEAKERS = 6


class VoiceSeparationError(Exception):
    """Exception raised for voice separation errors."""
    pass


class VoiceSeparator:
    """
    Handles voice separation using the MulCat blocks architecture.
    
    Provides functionality to separate multiple speakers from a single audio recording,
    with support for both known and unknown number of speakers.
    """
    
    def __init__(self, 
                 model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
                 device: str = None,
                 sample_rate: int = DEFAULT_SAMPLE_RATE):
        """
        Initialize the VoiceSeparator.
        
        Args:
            model_path: Path to the directory containing the SVoice models
            device: Device to run the model on ('cuda' or 'cpu')
            sample_rate: Sample rate for processing
        """
        self.model_path = Path(model_path)
        self.sample_rate = sample_rate
        
        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self.models = {}
        self.encoders = {}
        self.decoders = {}
        self._load_models()
        
        # Initialize speaker embedding model for quality evaluation
        self.speaker_encoder = None
    
    def _load_models(self):
        """
        Load SVoice models for different numbers of speakers.
        
        Loads models for 2-6 speakers from the model directory.
        """
        if not self.model_path.exists():
            raise VoiceSeparationError(f"Model directory not found: {self.model_path}")
            
        # Check for model configuration
        config_path = self.model_path / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                logger.info(f"Loaded model configuration: {config}")
        else:
            logger.warning(f"No model configuration found at {config_path}")
            config = {}
            
        # Load models for different numbers of speakers
        for num_speakers in range(2, MAX_NUM_SPEAKERS + 1):
            model_file = self.model_path / f"svoice_{num_speakers}spk.pt"
            
            if model_file.exists():
                try:
                    # Load model
                    logger.info(f"Loading model for {num_speakers} speakers from {model_file}")
                    checkpoint = torch.load(model_file, map_location=self.device)
                    
                    # Get model configuration
                    model_config = checkpoint.get('config', {})
                    
                    # Create encoder/decoder
                    enc_dec_class = model_config.get('filterbank', 'free')
                    n_filters = model_config.get('n_filters', 256)
                    kernel_size = model_config.get('kernel_size', 16)
                    stride = model_config.get('stride', 8)
                    
                    encoder, decoder = make_enc_dec(
                        enc_dec_class, 
                        n_filters, 
                        kernel_size, 
                        stride=stride
                    )
                    
                    # Load model weights
                    model = checkpoint['model']
                    
                    # Store models
                    self.models[num_speakers] = model.to(self.device)
                    self.encoders[num_speakers] = encoder.to(self.device)
                    self.decoders[num_speakers] = decoder.to(self.device)
                    
                    logger.info(f"Successfully loaded model for {num_speakers} speakers")
                except Exception as e:
                    logger.error(f"Failed to load model for {num_speakers} speakers: {e}")
            else:
                logger.warning(f"No model found for {num_speakers} speakers at {model_file}")
                
        if not self.models:
            raise VoiceSeparationError("No models could be loaded")
            
        logger.info(f"Loaded models for {list(self.models.keys())} speakers")
    
    def separate_voices(self, 
                        audio: np.ndarray, 
                        sr: int, 
                        num_speakers: Optional[int] = None) -> List[np.ndarray]:
        """
        Separate voices from an audio recording.
        
        Args:
            audio: Audio data as numpy array (shape: [channels, samples])
            sr: Sample rate of the audio
            num_speakers: Number of speakers to separate (if None, will be estimated)
            
        Returns:
            List of separated voice audio arrays
            
        Raises:
            VoiceSeparationError: If separation fails
        """
        # Ensure audio is in the correct format
        if isinstance(audio, torch.Tensor):
            audio_tensor = audio
        else:
            audio_tensor = torch.from_numpy(audio)
            
        # Ensure mono audio
        if audio_tensor.shape[0] > 1:
            audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
            
        # Resample if needed
        if sr != self.sample_rate:
            logger.debug(f"Resampling from {sr}Hz to {self.sample_rate}Hz")
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, 
                orig_freq=sr, 
                new_freq=self.sample_rate
            )
            
        # Move to device
        audio_tensor = audio_tensor.to(self.device)
        
        # Estimate number of speakers if not provided
        if num_speakers is None:
            num_speakers = self._estimate_num_speakers(audio_tensor)
            logger.info(f"Estimated number of speakers: {num_speakers}")
            
        # Ensure we have a model for this number of speakers
        if num_speakers not in self.models:
            closest_num = min(self.models.keys(), key=lambda x: abs(x - num_speakers))
            logger.warning(
                f"No model available for {num_speakers} speakers. "
                f"Using model for {closest_num} speakers instead."
            )
            num_speakers = closest_num
            
        # Get the appropriate model and encoder/decoder
        model = self.models[num_speakers]
        encoder = self.encoders[num_speakers]
        decoder = self.decoders[num_speakers]
        
        # Separate voices
        try:
            # Encode audio
            with torch.no_grad():
                # Convert to batch format
                batch = audio_tensor.unsqueeze(0)
                
                # Encode
                encoded = encoder(batch)
                
                # Apply separation model
                masks = model(encoded)
                
                # Apply masks and decode
                separated_voices = []
                for i in range(num_speakers):
                    mask = masks[:, i]
                    separated = decoder(encoded * mask)
                    separated_voices.append(separated.squeeze(0).cpu().numpy())
                    
                return separated_voices
                
        except Exception as e:
            logger.error(f"Voice separation failed: {e}")
            raise VoiceSeparationError(f"Voice separation failed: {e}")
    
    def _estimate_num_speakers(self, audio_tensor: torch.Tensor) -> int:
        """
        Estimate the number of speakers in an audio recording.
        
        This is a placeholder implementation. In a real implementation, this would use
        a more sophisticated approach like diarization or clustering.
        
        Args:
            audio_tensor: Audio data as torch tensor
            
        Returns:
            Estimated number of speakers
        """
        # TODO: Implement proper speaker count estimation
        # For now, just return the default
        return DEFAULT_NUM_SPEAKERS
    
    def evaluate_separation_quality(self, 
                                   original_audio: np.ndarray,
                                   separated_voices: List[np.ndarray]) -> Dict:
        """
        Evaluate the quality of voice separation.
        
        Args:
            original_audio: Original mixed audio
            separated_voices: List of separated voice audio arrays
            
        Returns:
            Dictionary with quality metrics
        """
        # Initialize metrics
        metrics = {
            'cross_correlation': [],
            'signal_to_distortion': [],
            'voice_characteristics_retention': []
        }
        
        # Calculate cross-correlation between separated voices
        for i in range(len(separated_voices)):
            for j in range(i+1, len(separated_voices)):
                voice1 = separated_voices[i].flatten()
                voice2 = separated_voices[j].flatten()
                
                # Normalize
                voice1 = (voice1 - np.mean(voice1)) / (np.std(voice1) + 1e-8)
                voice2 = (voice2 - np.mean(voice2)) / (np.std(voice2) + 1e-8)
                
                # Calculate correlation
                correlation = np.abs(np.correlate(voice1, voice2, mode='valid').max())
                metrics['cross_correlation'].append(float(correlation))
        
        # Calculate average cross-correlation
        if metrics['cross_correlation']:
            metrics['avg_cross_correlation'] = float(np.mean(metrics['cross_correlation']))
        else:
            metrics['avg_cross_correlation'] = 0.0
            
        # TODO: Implement more sophisticated quality metrics
        
        return metrics
    
    def load_speaker_encoder(self):
        """
        Load the speaker embedding model for quality evaluation.
        """
        if self.speaker_encoder is None:
            try:
                logger.info("Loading speaker embedding model")
                self.speaker_encoder = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir=str(self.model_path / "speaker_encoder")
                )
                logger.info("Speaker embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load speaker embedding model: {e}")
                self.speaker_encoder = None
    
    def extract_speaker_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract speaker embedding from audio.
        
        Args:
            audio: Audio data as numpy array
            sr: Sample rate
            
        Returns:
            Speaker embedding as numpy array
        """
        if self.speaker_encoder is None:
            self.load_speaker_encoder()
            
        if self.speaker_encoder is None:
            logger.warning("Speaker encoder not available, returning empty embedding")
            return np.zeros(192)  # Default ECAPA-TDNN embedding size
            
        # Ensure audio is in the correct format
        if isinstance(audio, torch.Tensor):
            audio_tensor = audio.cpu().numpy()
        else:
            audio_tensor = audio
            
        # Ensure mono audio
        if len(audio_tensor.shape) > 1 and audio_tensor.shape[0] > 1:
            audio_tensor = np.mean(audio_tensor, axis=0)
            
        # Flatten if needed
        if len(audio_tensor.shape) > 1:
            audio_tensor = audio_tensor.flatten()
            
        # Resample if needed
        if sr != 16000:  # SpeechBrain models expect 16kHz
            audio_tensor = librosa.resample(
                y=audio_tensor, 
                orig_sr=sr, 
                target_sr=16000
            )
            
        try:
            # Extract embedding
            with torch.no_grad():
                embedding = self.speaker_encoder.encode_batch(torch.tensor(audio_tensor).unsqueeze(0))
                return embedding.squeeze(0).cpu().numpy()
        except Exception as e:
            logger.error(f"Failed to extract speaker embedding: {e}")
            return np.zeros(192)  # Default ECAPA-TDNN embedding size 