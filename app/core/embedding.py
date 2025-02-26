"""
Speaker Embedding Module

This module implements speaker embedding functionality for voice identification.
It provides capabilities to extract and compare speaker embeddings.

Key features:
1. Speaker embedding extraction
2. Embedding comparison and similarity calculation
3. Embedding visualization
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import json
import pickle
import time

import numpy as np
import torch
import librosa
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import umap
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)

# Define constants
DEFAULT_MODEL_PATH = Path("models/embeddings")
DEFAULT_SAMPLE_RATE = 16000
EMBEDDING_DIMENSION = 192  # Default for ECAPA-TDNN


class EmbeddingError(Exception):
    """Exception raised for embedding errors."""
    pass


class SpeakerEmbedding:
    """
    Represents a speaker embedding vector with metadata.
    """
    
    def __init__(self, 
                 vector: np.ndarray,
                 speaker_id: str = None,
                 confidence: float = 1.0,
                 metadata: Dict = None):
        """
        Initialize a speaker embedding.
        
        Args:
            vector: Embedding vector as numpy array
            speaker_id: Speaker identifier (optional)
            confidence: Confidence score (0-1)
            metadata: Additional metadata (optional)
        """
        self.vector = vector
        self.speaker_id = speaker_id
        self.confidence = confidence
        self.metadata = metadata or {}
        self.created_at = time.time()
        
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self.vector.shape[0]
        
    def to_dict(self) -> Dict:
        """Convert to dictionary representation (without vector)."""
        return {
            'speaker_id': self.speaker_id,
            'confidence': self.confidence,
            'dimension': self.dimension(),
            'created_at': self.created_at,
            'metadata': self.metadata
        }
        
    def __repr__(self) -> str:
        """String representation."""
        return (f"SpeakerEmbedding(speaker_id={self.speaker_id}, "
                f"dimension={self.dimension()}, "
                f"confidence={self.confidence:.2f})")


class EmbeddingProcessor:
    """
    Handles speaker embedding extraction and comparison.
    
    Provides functionality to extract speaker embeddings from audio
    and compare them for similarity.
    """
    
    def __init__(self, model_path: Union[str, Path] = DEFAULT_MODEL_PATH):
        """
        Initialize the EmbeddingProcessor.
        
        Args:
            model_path: Path to the directory containing embedding models
        """
        self.model_path = Path(model_path)
        
        # Initialize embedding extractor
        # Note: We'll use the speaker encoder from VoiceSeparator
        # This is just a placeholder for the interface
        self.extractor = None
        
    def extract_embedding(self, 
                         audio: np.ndarray, 
                         sr: int,
                         speaker_id: str = None,
                         metadata: Dict = None) -> SpeakerEmbedding:
        """
        Extract speaker embedding from audio.
        
        This is a placeholder that would normally use the speaker encoder.
        In the actual implementation, this would be connected to the
        VoiceSeparator's speaker encoder.
        
        Args:
            audio: Audio data as numpy array
            sr: Sample rate
            speaker_id: Speaker identifier (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            Speaker embedding
            
        Raises:
            EmbeddingError: If extraction fails
        """
        # This is a placeholder - in the real implementation, this would
        # use the actual embedding extractor from VoiceSeparator
        try:
            # Ensure audio is in the correct format
            if isinstance(audio, torch.Tensor):
                audio_array = audio.cpu().numpy()
            else:
                audio_array = audio
                
            # Ensure mono audio
            if len(audio_array.shape) > 1 and audio_array.shape[0] > 1:
                audio_array = np.mean(audio_array, axis=0)
                
            # Flatten if needed
            if len(audio_array.shape) > 1:
                audio_array = audio_array.flatten()
                
            # Resample if needed
            if sr != DEFAULT_SAMPLE_RATE:
                audio_array = librosa.resample(
                    y=audio_array, 
                    orig_sr=sr, 
                    target_sr=DEFAULT_SAMPLE_RATE
                )
                
            # In a real implementation, this would use the actual model
            # For now, we'll create a random embedding for demonstration
            logger.warning("Using placeholder embedding extractor")
            vector = np.random.randn(EMBEDDING_DIMENSION)
            vector = normalize(vector.reshape(1, -1))[0]  # Normalize to unit length
            
            return SpeakerEmbedding(
                vector=vector,
                speaker_id=speaker_id,
                confidence=0.5,  # Lower confidence since this is a placeholder
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            raise EmbeddingError(f"Embedding extraction failed: {e}")
    
    def compare_embeddings(self, 
                          embedding1: SpeakerEmbedding, 
                          embedding2: SpeakerEmbedding) -> float:
        """
        Compare two speaker embeddings for similarity.
        
        Args:
            embedding1: First speaker embedding
            embedding2: Second speaker embedding
            
        Returns:
            Similarity score (0-1)
            
        Raises:
            EmbeddingError: If comparison fails
        """
        try:
            # Ensure vectors are normalized
            vec1 = normalize(embedding1.vector.reshape(1, -1))[0]
            vec2 = normalize(embedding2.vector.reshape(1, -1))[0]
            
            # Calculate cosine similarity
            similarity = cosine_similarity([vec1], [vec2])[0, 0]
            
            # Convert to range 0-1 (cosine similarity is between -1 and 1)
            similarity = (similarity + 1) / 2
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Embedding comparison failed: {e}")
            raise EmbeddingError(f"Embedding comparison failed: {e}")
    
    def compare_embedding_to_group(self, 
                                  embedding: SpeakerEmbedding, 
                                  group: List[SpeakerEmbedding]) -> Dict[str, float]:
        """
        Compare an embedding to a group of embeddings.
        
        Args:
            embedding: Speaker embedding to compare
            group: List of speaker embeddings to compare against
            
        Returns:
            Dictionary mapping speaker IDs to similarity scores
            
        Raises:
            EmbeddingError: If comparison fails
        """
        try:
            results = {}
            
            for other in group:
                similarity = self.compare_embeddings(embedding, other)
                
                # Use speaker ID if available, otherwise use index
                key = other.speaker_id or f"unknown_{id(other)}"
                results[key] = similarity
                
            return results
            
        except Exception as e:
            logger.error(f"Embedding group comparison failed: {e}")
            raise EmbeddingError(f"Embedding group comparison failed: {e}")
    
    def find_most_similar(self, 
                         embedding: SpeakerEmbedding, 
                         group: List[SpeakerEmbedding],
                         threshold: float = 0.7) -> Optional[Tuple[SpeakerEmbedding, float]]:
        """
        Find the most similar embedding in a group.
        
        Args:
            embedding: Speaker embedding to compare
            group: List of speaker embeddings to compare against
            threshold: Minimum similarity threshold (0-1)
            
        Returns:
            Tuple of (most_similar_embedding, similarity_score) or None if no match
            
        Raises:
            EmbeddingError: If comparison fails
        """
        try:
            if not group:
                return None
                
            similarities = []
            
            for other in group:
                similarity = self.compare_embeddings(embedding, other)
                similarities.append((other, similarity))
                
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Get the most similar
            most_similar, highest_similarity = similarities[0]
            
            # Check if it meets the threshold
            if highest_similarity >= threshold:
                return most_similar, highest_similarity
            else:
                return None
                
        except Exception as e:
            logger.error(f"Finding most similar embedding failed: {e}")
            raise EmbeddingError(f"Finding most similar embedding failed: {e}")
    
    def save_embedding(self, 
                      embedding: SpeakerEmbedding, 
                      file_path: Union[str, Path]) -> None:
        """
        Save a speaker embedding to a file.
        
        Args:
            embedding: Speaker embedding to save
            file_path: Path to save the embedding to
            
        Raises:
            EmbeddingError: If saving fails
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'wb') as f:
                pickle.dump(embedding, f)
                
        except Exception as e:
            logger.error(f"Failed to save embedding: {e}")
            raise EmbeddingError(f"Failed to save embedding: {e}")
    
    def load_embedding(self, file_path: Union[str, Path]) -> SpeakerEmbedding:
        """
        Load a speaker embedding from a file.
        
        Args:
            file_path: Path to load the embedding from
            
        Returns:
            Speaker embedding
            
        Raises:
            EmbeddingError: If loading fails
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise EmbeddingError(f"Embedding file not found: {file_path}")
                
            with open(file_path, 'rb') as f:
                embedding = pickle.load(f)
                
            if not isinstance(embedding, SpeakerEmbedding):
                raise EmbeddingError(f"File does not contain a valid SpeakerEmbedding: {file_path}")
                
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to load embedding: {e}")
            raise EmbeddingError(f"Failed to load embedding: {e}")
    
    def visualize_embeddings(self, 
                            embeddings: List[SpeakerEmbedding], 
                            output_path: Union[str, Path] = None,
                            title: str = "Speaker Embeddings Visualization") -> Any:
        """
        Visualize speaker embeddings using UMAP.
        
        Args:
            embeddings: List of speaker embeddings to visualize
            output_path: Path to save the visualization to (optional)
            title: Title for the visualization
            
        Returns:
            Matplotlib figure or None if visualization is not available
            
        Raises:
            EmbeddingError: If visualization fails
        """
        if not VISUALIZATION_AVAILABLE:
            logger.warning("Visualization libraries not available")
            return None
            
        if len(embeddings) < 2:
            logger.warning("Need at least 2 embeddings for visualization")
            return None
            
        try:
            # Extract vectors and labels
            vectors = np.array([e.vector for e in embeddings])
            labels = [e.speaker_id or f"unknown_{i}" for i, e in enumerate(embeddings)]
            
            # Reduce dimensionality with UMAP
            reducer = umap.UMAP(n_neighbors=min(15, len(vectors)), min_dist=0.1, random_state=42)
            embedding_2d = reducer.fit_transform(vectors)
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create a colormap
            unique_labels = list(set(labels))
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
            color_map = {label: color for label, color in zip(unique_labels, colors)}
            
            # Plot points
            for i, (x, y) in enumerate(embedding_2d):
                label = labels[i]
                color = color_map[label]
                ax.scatter(x, y, color=color, label=label, alpha=0.7, s=100)
                
            # Remove duplicate labels
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='best')
            
            # Add title and labels
            ax.set_title(title)
            ax.set_xlabel("UMAP Dimension 1")
            ax.set_ylabel("UMAP Dimension 2")
            
            # Save if output path is provided
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                
            return fig
            
        except Exception as e:
            logger.error(f"Embedding visualization failed: {e}")
            raise EmbeddingError(f"Embedding visualization failed: {e}")
            
        finally:
            # Close the figure to prevent memory leaks
            if 'fig' in locals():
                plt.close(fig) 