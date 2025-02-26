"""
Clustering Module

This module implements speaker clustering functionality for identifying unique speakers.
It provides capabilities to cluster speaker embeddings and match them across recordings.

Key features:
1. Speaker embedding clustering
2. Cross-recording speaker matching
3. Adaptive parameter selection
4. Cluster quality validation
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import json
import pickle
import time
import uuid

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

# Try to import advanced clustering libraries
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)

# Import local modules
from app.core.embedding import SpeakerEmbedding, EmbeddingError


class ClusteringError(Exception):
    """Exception raised for clustering errors."""
    pass


class SpeakerCluster:
    """
    Represents a cluster of speaker embeddings.
    """
    
    def __init__(self, 
                 cluster_id: str = None,
                 embeddings: List[SpeakerEmbedding] = None,
                 centroid: np.ndarray = None,
                 metadata: Dict = None):
        """
        Initialize a speaker cluster.
        
        Args:
            cluster_id: Cluster identifier (optional)
            embeddings: List of speaker embeddings (optional)
            centroid: Centroid vector (optional)
            metadata: Additional metadata (optional)
        """
        self.cluster_id = cluster_id or str(uuid.uuid4())
        self.embeddings = embeddings or []
        self.centroid = centroid
        self.metadata = metadata or {}
        self.created_at = time.time()
        self.updated_at = self.created_at
        
        # Calculate centroid if not provided
        if self.centroid is None and self.embeddings:
            self._update_centroid()
    
    def add_embedding(self, embedding: SpeakerEmbedding) -> None:
        """
        Add an embedding to the cluster.
        
        Args:
            embedding: Speaker embedding to add
        """
        self.embeddings.append(embedding)
        self._update_centroid()
        self.updated_at = time.time()
    
    def remove_embedding(self, embedding: SpeakerEmbedding) -> bool:
        """
        Remove an embedding from the cluster.
        
        Args:
            embedding: Speaker embedding to remove
            
        Returns:
            True if the embedding was removed, False otherwise
        """
        try:
            self.embeddings.remove(embedding)
            self._update_centroid()
            self.updated_at = time.time()
            return True
        except ValueError:
            return False
    
    def merge_with(self, other_cluster: 'SpeakerCluster') -> None:
        """
        Merge with another cluster.
        
        Args:
            other_cluster: Cluster to merge with
        """
        self.embeddings.extend(other_cluster.embeddings)
        self._update_centroid()
        self.updated_at = time.time()
    
    def _update_centroid(self) -> None:
        """Update the centroid based on current embeddings."""
        if not self.embeddings:
            self.centroid = None
            return
            
        # Stack embedding vectors
        vectors = np.vstack([e.vector for e in self.embeddings])
        
        # Calculate mean
        self.centroid = np.mean(vectors, axis=0)
        
        # Normalize
        norm = np.linalg.norm(self.centroid)
        if norm > 0:
            self.centroid = self.centroid / norm
    
    def get_size(self) -> int:
        """Get the number of embeddings in the cluster."""
        return len(self.embeddings)
    
    def get_coherence(self) -> float:
        """
        Calculate the coherence of the cluster.
        
        Returns:
            Coherence score (higher is better)
        """
        if len(self.embeddings) < 2:
            return 1.0  # Perfect coherence for single-element clusters
            
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(self.embeddings)):
            for j in range(i+1, len(self.embeddings)):
                vec1 = normalize(self.embeddings[i].vector.reshape(1, -1))[0]
                vec2 = normalize(self.embeddings[j].vector.reshape(1, -1))[0]
                similarity = np.dot(vec1, vec2)
                similarities.append(similarity)
                
        # Return mean similarity
        return float(np.mean(similarities))
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation (without embeddings)."""
        return {
            'cluster_id': self.cluster_id,
            'size': self.get_size(),
            'coherence': self.get_coherence(),
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'metadata': self.metadata
        }
        
    def __repr__(self) -> str:
        """String representation."""
        return (f"SpeakerCluster(id={self.cluster_id}, "
                f"size={self.get_size()}, "
                f"coherence={self.get_coherence():.2f})")


class ClusteringProcessor:
    """
    Handles speaker clustering and matching.
    
    Provides functionality to cluster speaker embeddings and match
    clusters across recordings.
    """
    
    def __init__(self, 
                 distance_threshold: float = 0.5,
                 min_cluster_size: int = 2,
                 min_samples: int = 1):
        """
        Initialize the ClusteringProcessor.
        
        Args:
            distance_threshold: Distance threshold for clustering (0-1)
            min_cluster_size: Minimum cluster size for HDBSCAN
            min_samples: Minimum samples for HDBSCAN
        """
        self.distance_threshold = distance_threshold
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
    
    def cluster_embeddings(self, 
                          embeddings: List[SpeakerEmbedding],
                          method: str = 'agglomerative',
                          adaptive: bool = True) -> List[SpeakerCluster]:
        """
        Cluster speaker embeddings.
        
        Args:
            embeddings: List of speaker embeddings to cluster
            method: Clustering method ('agglomerative' or 'hdbscan')
            adaptive: Whether to adaptively select parameters
            
        Returns:
            List of speaker clusters
            
        Raises:
            ClusteringError: If clustering fails
        """
        if not embeddings:
            return []
            
        try:
            # Extract vectors
            vectors = np.vstack([e.vector for e in embeddings])
            
            # Normalize vectors
            vectors = normalize(vectors)
            
            # Determine parameters if adaptive
            if adaptive:
                self._adapt_parameters(vectors)
                
            # Perform clustering
            if method == 'hdbscan' and HDBSCAN_AVAILABLE:
                labels = self._cluster_hdbscan(vectors)
            else:
                labels = self._cluster_agglomerative(vectors)
                
            # Create clusters
            clusters = {}
            for i, label in enumerate(labels):
                # Skip noise points (label -1)
                if label == -1:
                    continue
                    
                if label not in clusters:
                    clusters[label] = SpeakerCluster(
                        cluster_id=f"cluster_{label}",
                        embeddings=[embeddings[i]]
                    )
                else:
                    clusters[label].add_embedding(embeddings[i])
                    
            return list(clusters.values())
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            raise ClusteringError(f"Clustering failed: {e}")
    
    def _cluster_agglomerative(self, vectors: np.ndarray) -> np.ndarray:
        """
        Perform agglomerative clustering.
        
        Args:
            vectors: Array of embedding vectors
            
        Returns:
            Array of cluster labels
        """
        # Convert distance threshold to affinity threshold
        # distance_threshold is in range 0-1, where 0 is identical
        # affinity_threshold is in range 0-2, where 0 is identical
        affinity_threshold = self.distance_threshold * 2
        
        # Create clustering model
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=affinity_threshold,
            affinity='cosine',
            linkage='average'
        )
        
        # Fit model
        return model.fit_predict(vectors)
    
    def _cluster_hdbscan(self, vectors: np.ndarray) -> np.ndarray:
        """
        Perform HDBSCAN clustering.
        
        Args:
            vectors: Array of embedding vectors
            
        Returns:
            Array of cluster labels
        """
        if not HDBSCAN_AVAILABLE:
            raise ClusteringError("HDBSCAN is not available")
            
        # Create clustering model
        model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='cosine',
            cluster_selection_method='eom'
        )
        
        # Fit model
        return model.fit_predict(vectors)
    
    def _adapt_parameters(self, vectors: np.ndarray) -> None:
        """
        Adaptively select clustering parameters.
        
        Args:
            vectors: Array of embedding vectors
        """
        # This is a simple adaptive parameter selection
        # In a real implementation, this would be more sophisticated
        
        # Adjust distance threshold based on number of vectors
        n_vectors = vectors.shape[0]
        
        if n_vectors < 5:
            # For very few vectors, use a stricter threshold
            self.distance_threshold = 0.3
        elif n_vectors < 10:
            # For few vectors, use a moderate threshold
            self.distance_threshold = 0.4
        else:
            # For many vectors, use a more relaxed threshold
            self.distance_threshold = 0.5
            
        # Adjust min_cluster_size based on number of vectors
        self.min_cluster_size = max(2, int(n_vectors * 0.1))
        
        logger.debug(f"Adapted parameters: distance_threshold={self.distance_threshold}, "
                    f"min_cluster_size={self.min_cluster_size}")
    
    def evaluate_clustering(self, 
                           vectors: np.ndarray, 
                           labels: np.ndarray) -> Dict:
        """
        Evaluate clustering quality.
        
        Args:
            vectors: Array of embedding vectors
            labels: Array of cluster labels
            
        Returns:
            Dictionary with quality metrics
        """
        metrics = {}
        
        # Skip if all points are noise
        if np.all(labels == -1):
            metrics['silhouette_score'] = 0.0
            metrics['num_clusters'] = 0
            metrics['noise_percentage'] = 100.0
            return metrics
            
        # Calculate number of clusters
        unique_labels = np.unique(labels)
        num_clusters = len(unique_labels[unique_labels != -1])
        metrics['num_clusters'] = num_clusters
        
        # Calculate noise percentage
        noise_percentage = np.sum(labels == -1) / len(labels) * 100
        metrics['noise_percentage'] = float(noise_percentage)
        
        # Calculate silhouette score if more than one cluster
        if num_clusters > 1:
            # Filter out noise points
            non_noise_mask = labels != -1
            non_noise_vectors = vectors[non_noise_mask]
            non_noise_labels = labels[non_noise_mask]
            
            # Calculate silhouette score
            if len(np.unique(non_noise_labels)) > 1:
                silhouette = silhouette_score(
                    non_noise_vectors, 
                    non_noise_labels, 
                    metric='cosine'
                )
                metrics['silhouette_score'] = float(silhouette)
            else:
                metrics['silhouette_score'] = 0.0
        else:
            metrics['silhouette_score'] = 0.0
            
        return metrics
    
    def match_clusters(self, 
                      source_clusters: List[SpeakerCluster],
                      target_clusters: List[SpeakerCluster],
                      threshold: float = 0.7) -> Dict[str, str]:
        """
        Match clusters across recordings.
        
        Args:
            source_clusters: List of source clusters
            target_clusters: List of target clusters
            threshold: Similarity threshold for matching
            
        Returns:
            Dictionary mapping source cluster IDs to target cluster IDs
            
        Raises:
            ClusteringError: If matching fails
        """
        try:
            matches = {}
            
            for source in source_clusters:
                best_match = None
                best_similarity = 0.0
                
                for target in target_clusters:
                    # Calculate similarity between centroids
                    similarity = self._calculate_centroid_similarity(source, target)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = target
                        
                # Check if similarity exceeds threshold
                if best_match is not None and best_similarity >= threshold:
                    matches[source.cluster_id] = best_match.cluster_id
                    
            return matches
            
        except Exception as e:
            logger.error(f"Cluster matching failed: {e}")
            raise ClusteringError(f"Cluster matching failed: {e}")
    
    def _calculate_centroid_similarity(self, 
                                     cluster1: SpeakerCluster, 
                                     cluster2: SpeakerCluster) -> float:
        """
        Calculate similarity between cluster centroids.
        
        Args:
            cluster1: First cluster
            cluster2: Second cluster
            
        Returns:
            Similarity score (0-1)
        """
        if cluster1.centroid is None or cluster2.centroid is None:
            return 0.0
            
        # Normalize centroids
        centroid1 = cluster1.centroid / np.linalg.norm(cluster1.centroid)
        centroid2 = cluster2.centroid / np.linalg.norm(cluster2.centroid)
        
        # Calculate cosine similarity
        similarity = np.dot(centroid1, centroid2)
        
        # Convert to range 0-1 (cosine similarity is between -1 and 1)
        similarity = (similarity + 1) / 2
        
        return float(similarity)
    
    def save_clusters(self, 
                     clusters: List[SpeakerCluster], 
                     file_path: Union[str, Path]) -> None:
        """
        Save clusters to a file.
        
        Args:
            clusters: List of speaker clusters
            file_path: Path to save the clusters to
            
        Raises:
            ClusteringError: If saving fails
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'wb') as f:
                pickle.dump(clusters, f)
                
        except Exception as e:
            logger.error(f"Failed to save clusters: {e}")
            raise ClusteringError(f"Failed to save clusters: {e}")
    
    def load_clusters(self, file_path: Union[str, Path]) -> List[SpeakerCluster]:
        """
        Load clusters from a file.
        
        Args:
            file_path: Path to load the clusters from
            
        Returns:
            List of speaker clusters
            
        Raises:
            ClusteringError: If loading fails
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise ClusteringError(f"Clusters file not found: {file_path}")
                
            with open(file_path, 'rb') as f:
                clusters = pickle.load(f)
                
            if not isinstance(clusters, list) or not all(isinstance(c, SpeakerCluster) for c in clusters):
                raise ClusteringError(f"File does not contain valid SpeakerClusters: {file_path}")
                
            return clusters
            
        except Exception as e:
            logger.error(f"Failed to load clusters: {e}")
            raise ClusteringError(f"Failed to load clusters: {e}")
    
    def merge_clusters(self, 
                      clusters1: List[SpeakerCluster],
                      clusters2: List[SpeakerCluster],
                      threshold: float = 0.7) -> List[SpeakerCluster]:
        """
        Merge two sets of clusters.
        
        Args:
            clusters1: First set of clusters
            clusters2: Second set of clusters
            threshold: Similarity threshold for merging
            
        Returns:
            Merged list of clusters
            
        Raises:
            ClusteringError: If merging fails
        """
        try:
            # Start with a copy of the first set
            merged_clusters = [SpeakerCluster(
                cluster_id=c.cluster_id,
                embeddings=c.embeddings.copy(),
                metadata=c.metadata.copy()
            ) for c in clusters1]
            
            # Match clusters
            matches = self.match_clusters(clusters2, merged_clusters, threshold)
            
            # Merge matched clusters
            for cluster2 in clusters2:
                if cluster2.cluster_id in matches:
                    # Find the matching cluster in merged_clusters
                    for merged_cluster in merged_clusters:
                        if merged_cluster.cluster_id == matches[cluster2.cluster_id]:
                            # Merge embeddings
                            for embedding in cluster2.embeddings:
                                merged_cluster.add_embedding(embedding)
                            break
                else:
                    # Add as a new cluster
                    merged_clusters.append(SpeakerCluster(
                        cluster_id=cluster2.cluster_id,
                        embeddings=cluster2.embeddings.copy(),
                        metadata=cluster2.metadata.copy()
                    ))
                    
            return merged_clusters
            
        except Exception as e:
            logger.error(f"Cluster merging failed: {e}")
            raise ClusteringError(f"Cluster merging failed: {e}")
    
    def get_cluster_stats(self, clusters: List[SpeakerCluster]) -> Dict:
        """
        Get statistics for a set of clusters.
        
        Args:
            clusters: List of speaker clusters
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'num_clusters': len(clusters),
            'total_embeddings': sum(c.get_size() for c in clusters),
            'avg_cluster_size': 0.0,
            'avg_coherence': 0.0,
            'cluster_sizes': [],
            'cluster_coherences': []
        }
        
        if clusters:
            stats['avg_cluster_size'] = stats['total_embeddings'] / stats['num_clusters']
            stats['avg_coherence'] = sum(c.get_coherence() for c in clusters) / stats['num_clusters']
            stats['cluster_sizes'] = [c.get_size() for c in clusters]
            stats['cluster_coherences'] = [c.get_coherence() for c in clusters]
            
        return stats 