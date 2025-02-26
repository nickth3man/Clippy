"""
Pipeline Integration Module

This module integrates all core components into a complete processing pipeline.
It provides a unified interface for processing audio recordings and building speaker profiles.

Key features:
1. Voice separation pipeline
2. Speaker identification system
3. Profile building system
4. Cross-recording analysis
"""

import os
import logging
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import json

import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

# Import local modules
from app.core.audio_processor import AudioProcessor, AudioValidationError
from app.core.voice_separator import VoiceSeparator, VoiceSeparationError
from app.core.diarization import Diarizer, DiarizationError, SpeakerSegment
from app.core.embedding import EmbeddingProcessor, SpeakerEmbedding, EmbeddingError
from app.core.clustering import ClusteringProcessor, SpeakerCluster, ClusteringError
from app.core.profile_db import ProfileDatabase, SpeakerProfile, ProfileDBError


class PipelineError(Exception):
    """Exception raised for pipeline errors."""
    pass


class ProcessingResult:
    """
    Represents the result of processing an audio recording.
    """
    
    def __init__(self, 
                 recording_id: str = None,
                 recording_path: str = None):
        """
        Initialize a processing result.
        
        Args:
            recording_id: Recording identifier (optional)
            recording_path: Path to the recording file (optional)
        """
        self.recording_id = recording_id or str(uuid.uuid4())
        self.recording_path = recording_path
        self.created_at = time.time()
        
        # Processing results
        self.duration = 0.0
        self.num_speakers = 0
        self.segments = []  # List of SpeakerSegment objects
        self.separated_voices = []  # List of separated voice arrays
        self.embeddings = []  # List of SpeakerEmbedding objects
        self.clusters = []  # List of SpeakerCluster objects
        self.profiles = []  # List of SpeakerProfile objects
        
        # Processing metadata
        self.processing_time = 0.0
        self.quality_metrics = {}
        self.errors = []
        self.warnings = []
        
    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
        logger.error(f"Processing error: {error}")
        
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
        logger.warning(f"Processing warning: {warning}")
        
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'recording_id': self.recording_id,
            'recording_path': self.recording_path,
            'created_at': self.created_at,
            'duration': self.duration,
            'num_speakers': self.num_speakers,
            'num_segments': len(self.segments),
            'num_separated_voices': len(self.separated_voices),
            'num_embeddings': len(self.embeddings),
            'num_clusters': len(self.clusters),
            'num_profiles': len(self.profiles),
            'processing_time': self.processing_time,
            'quality_metrics': self.quality_metrics,
            'errors': self.errors,
            'warnings': self.warnings,
            'has_errors': len(self.errors) > 0
        }
        
    def __repr__(self) -> str:
        """String representation."""
        return (f"ProcessingResult(id={self.recording_id}, "
                f"speakers={self.num_speakers}, "
                f"segments={len(self.segments)}, "
                f"clusters={len(self.clusters)}, "
                f"profiles={len(self.profiles)}, "
                f"errors={len(self.errors)})")


class Pipeline:
    """
    Integrates all core components into a complete processing pipeline.
    
    Provides a unified interface for processing audio recordings and building speaker profiles.
    """
    
    def __init__(self, 
                 models_dir: Union[str, Path] = None,
                 db_path: Union[str, Path] = None,
                 device: str = None):
        """
        Initialize the Pipeline.
        
        Args:
            models_dir: Path to the directory containing models (optional)
            db_path: Path to the database file (optional)
            device: Device to run models on ('cuda' or 'cpu')
        """
        # Set default models directory if not provided
        if models_dir is None:
            models_dir = Path("models")
            
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.audio_processor = AudioProcessor()
        self.voice_separator = VoiceSeparator(
            model_path=self.models_dir / "svoice",
            device=device
        )
        self.diarizer = Diarizer(
            model_path=self.models_dir / "whisperx",
            device=device
        )
        self.embedding_processor = EmbeddingProcessor(
            model_path=self.models_dir / "embeddings"
        )
        self.clustering_processor = ClusteringProcessor()
        self.profile_db = ProfileDatabase(db_path)
        
        # Initialize state
        self.current_result = None
        
        logger.info("Pipeline initialized")
    
    def process_recording(self, 
                         file_path: Union[str, Path],
                         min_speakers: int = None,
                         max_speakers: int = None,
                         progress_callback: Callable[[str, float], None] = None) -> ProcessingResult:
        """
        Process an audio recording.
        
        Args:
            file_path: Path to the audio file
            min_speakers: Minimum number of speakers (optional)
            max_speakers: Maximum number of speakers (optional)
            progress_callback: Callback function for progress updates (optional)
            
        Returns:
            Processing result
            
        Raises:
            PipelineError: If processing fails
        """
        start_time = time.time()
        
        # Create result object
        file_path = Path(file_path)
        result = ProcessingResult(
            recording_id=str(uuid.uuid4()),
            recording_path=str(file_path)
        )
        self.current_result = result
        
        try:
            # Update progress
            if progress_callback:
                progress_callback("Loading audio", 0.0)
                
            # Load audio
            audio, sr = self.audio_processor.load_audio(file_path)
            result.duration = audio.shape[1] / sr
            
            # Check audio quality
            quality = self.audio_processor.check_audio_quality(audio, sr)
            result.quality_metrics['audio_quality'] = quality
            
            # Update progress
            if progress_callback:
                progress_callback("Performing diarization", 0.1)
                
            # Perform diarization if available
            if self.diarizer.available:
                try:
                    segments = self.diarizer.diarize(
                        file_path,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers
                    )
                    result.segments = segments
                    
                    # Get unique speakers
                    speaker_ids = set(s.speaker_id for s in segments)
                    result.num_speakers = len(speaker_ids)
                    
                    # Get speaker stats
                    speaker_stats = self.diarizer.get_speaker_stats(segments)
                    result.quality_metrics['speaker_stats'] = speaker_stats
                    
                except DiarizationError as e:
                    result.add_warning(f"Diarization failed: {e}")
                    # Continue with voice separation
            else:
                result.add_warning("Diarization not available")
                
            # Update progress
            if progress_callback:
                progress_callback("Separating voices", 0.3)
                
            # Separate voices
            try:
                # Use diarization result for number of speakers if available
                num_speakers_for_separation = result.num_speakers or None
                
                # Ensure within supported range
                if num_speakers_for_separation is not None:
                    num_speakers_for_separation = max(2, min(num_speakers_for_separation, 6))
                    
                separated_voices = self.voice_separator.separate_voices(
                    audio, sr, num_speakers=num_speakers_for_separation
                )
                result.separated_voices = separated_voices
                
                # If diarization failed, update num_speakers
                if result.num_speakers == 0:
                    result.num_speakers = len(separated_voices)
                    
                # Evaluate separation quality
                separation_quality = self.voice_separator.evaluate_separation_quality(
                    audio, separated_voices
                )
                result.quality_metrics['separation_quality'] = separation_quality
                
            except VoiceSeparationError as e:
                result.add_error(f"Voice separation failed: {e}")
                return result
                
            # Update progress
            if progress_callback:
                progress_callback("Extracting embeddings", 0.5)
                
            # Extract embeddings
            embeddings = []
            for i, voice in enumerate(result.separated_voices):
                try:
                    # Use speaker ID from diarization if available
                    speaker_id = None
                    if i < len(result.segments):
                        speaker_id = result.segments[i].speaker_id
                        
                    embedding = self.embedding_processor.extract_embedding(
                        voice, sr, speaker_id=speaker_id
                    )
                    embeddings.append(embedding)
                    
                except EmbeddingError as e:
                    result.add_warning(f"Embedding extraction failed for voice {i}: {e}")
                    
            result.embeddings = embeddings
            
            # Update progress
            if progress_callback:
                progress_callback("Clustering speakers", 0.7)
                
            # Cluster embeddings
            if len(embeddings) > 1:
                try:
                    clusters = self.clustering_processor.cluster_embeddings(
                        embeddings, adaptive=True
                    )
                    result.clusters = clusters
                    
                    # Evaluate clustering
                    if len(clusters) > 0:
                        vectors = np.vstack([e.vector for e in embeddings])
                        labels = np.array([-1] * len(embeddings))
                        
                        for i, cluster in enumerate(clusters):
                            for embedding in cluster.embeddings:
                                idx = embeddings.index(embedding)
                                labels[idx] = i
                                
                        clustering_quality = self.clustering_processor.evaluate_clustering(
                            vectors, labels
                        )
                        result.quality_metrics['clustering_quality'] = clustering_quality
                        
                except ClusteringError as e:
                    result.add_warning(f"Clustering failed: {e}")
            else:
                # Create a single cluster if only one embedding
                if embeddings:
                    cluster = SpeakerCluster(
                        cluster_id="cluster_0",
                        embeddings=embeddings
                    )
                    result.clusters = [cluster]
                    
            # Update progress
            if progress_callback:
                progress_callback("Matching with profiles", 0.9)
                
            # Match with existing profiles
            profiles = []
            for cluster in result.clusters:
                # Use representative embedding from cluster
                if not cluster.embeddings:
                    continue
                    
                representative = cluster.embeddings[0]
                
                # Search for matching profiles
                matches = self.profile_db.search_profiles(
                    representative, threshold=0.7, limit=1
                )
                
                if matches:
                    # Update existing profile
                    profile_dict, similarity = matches[0]
                    profile = self.profile_db.update_profile_from_cluster(
                        profile_dict['profile_id'], cluster, confidence=similarity
                    )
                    if profile:
                        profiles.append(profile)
                else:
                    # Create new profile
                    profile = self.profile_db.create_profile_from_cluster(
                        cluster, name=f"Speaker {len(profiles) + 1}"
                    )
                    profiles.append(profile)
                    
            result.profiles = profiles
            
            # Add recording to database
            self.profile_db.add_recording(
                result.recording_id,
                file_path.name,
                str(file_path),
                result.duration
            )
            
            # Add appearances
            for profile in profiles:
                # Calculate total duration for this speaker
                duration = 0.0
                for segment in result.segments:
                    if segment.speaker_id == profile.profile_id:
                        duration += segment.duration()
                        
                # If no segments, use a default duration
                if duration == 0.0:
                    duration = result.duration / len(profiles)
                    
                self.profile_db.add_appearance(
                    profile.profile_id,
                    result.recording_id,
                    confidence=profile.confidence,
                    duration=duration
                )
                
            # Update progress
            if progress_callback:
                progress_callback("Processing complete", 1.0)
                
            # Calculate processing time
            result.processing_time = time.time() - start_time
            
            return result
            
        except Exception as e:
            result.add_error(f"Processing failed: {e}")
            logger.exception(f"Processing failed: {e}")
            
            # Calculate processing time even for failed processing
            result.processing_time = time.time() - start_time
            
            return result
    
    def get_profile(self, profile_id: str) -> Optional[SpeakerProfile]:
        """
        Get a speaker profile.
        
        Args:
            profile_id: Profile identifier
            
        Returns:
            Speaker profile or None if not found
        """
        return self.profile_db.get_profile(profile_id)
    
    def list_profiles(self) -> List[Dict]:
        """
        List all speaker profiles.
        
        Returns:
            List of profile dictionaries
        """
        return self.profile_db.list_profiles()
    
    def delete_profile(self, profile_id: str) -> bool:
        """
        Delete a speaker profile.
        
        Args:
            profile_id: Profile identifier
            
        Returns:
            True if the profile was deleted, False otherwise
        """
        return self.profile_db.delete_profile(profile_id)
    
    def get_profile_appearances(self, profile_id: str) -> List[Dict]:
        """
        Get all appearances of a profile.
        
        Args:
            profile_id: Profile identifier
            
        Returns:
            List of appearance dictionaries
        """
        return self.profile_db.get_profile_appearances(profile_id)
    
    def get_database_stats(self) -> Dict:
        """
        Get statistics about the database.
        
        Returns:
            Dictionary with statistics
        """
        return self.profile_db.get_database_stats()
    
    def backup_database(self, backup_path: Union[str, Path] = None) -> str:
        """
        Create a backup of the database.
        
        Args:
            backup_path: Path to save the backup to (optional)
            
        Returns:
            Path to the backup file
        """
        return self.profile_db.backup_database(backup_path)
    
    def restore_database(self, backup_path: Union[str, Path]) -> None:
        """
        Restore the database from a backup.
        
        Args:
            backup_path: Path to the backup file
        """
        self.profile_db.restore_database(backup_path)
    
    def close(self) -> None:
        """Close all components."""
        self.profile_db.close() 