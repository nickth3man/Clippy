"""
Tests for the ClusteringProcessor class in the core module.
"""
import unittest
import numpy as np
from unittest.mock import patch, MagicMock

from app.core import ClusteringProcessor, SpeakerCluster, ClusteringError, SpeakerEmbedding


class TestClusteringProcessor(unittest.TestCase):
    """Test cases for the ClusteringProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.clustering_processor = ClusteringProcessor()
        
        # Create some test embeddings (10 embeddings, 512-dimensional)
        self.test_embeddings = []
        for i in range(10):
            # Create embeddings where the first 5 are similar to each other
            # and the last 5 are similar to each other
            if i < 5:
                base_vector = np.ones(512) * 0.1
                noise = np.random.rand(512) * 0.05
                vector = base_vector + noise
            else:
                base_vector = np.ones(512) * 0.9
                noise = np.random.rand(512) * 0.05
                vector = base_vector + noise
                
            self.test_embeddings.append(SpeakerEmbedding(vector.astype(np.float32)))
    
    def test_cluster_embeddings_mock(self):
        """Test clustering with mocked clustering algorithm."""
        # Mock the actual clustering function
        with patch.object(self.clustering_processor, '_cluster_embeddings') as mock_cluster:
            # Create mock clusters (2 clusters)
            mock_clusters = [
                SpeakerCluster(id="cluster1", embeddings=self.test_embeddings[:5]),
                SpeakerCluster(id="cluster2", embeddings=self.test_embeddings[5:])
            ]
            mock_cluster.return_value = mock_clusters
            
            # Call the cluster_embeddings method
            result = self.clustering_processor.cluster_embeddings(self.test_embeddings)
            
            # Verify the result
            self.assertEqual(len(result), 2)
            self.assertEqual(len(result[0].embeddings), 5)
            self.assertEqual(len(result[1].embeddings), 5)
            
            # Verify the mock was called with the right arguments
            mock_cluster.assert_called_once()
    
    def test_cluster_with_num_clusters(self):
        """Test clustering with specified number of clusters."""
        # Mock the clustering function
        with patch.object(self.clustering_processor, '_cluster_embeddings') as mock_cluster:
            mock_clusters = [
                SpeakerCluster(id="cluster1", embeddings=self.test_embeddings[:5]),
                SpeakerCluster(id="cluster2", embeddings=self.test_embeddings[5:])
            ]
            mock_cluster.return_value = mock_clusters
            
            # Call the cluster_embeddings method with num_clusters=2
            result = self.clustering_processor.cluster_embeddings(self.test_embeddings, num_clusters=2)
            
            # Verify the mock was called with num_clusters=2
            mock_cluster.assert_called_once()
            args, kwargs = mock_cluster.call_args
            self.assertEqual(kwargs.get('num_clusters'), 2)
    
    def test_cluster_with_min_max_clusters(self):
        """Test clustering with min and max clusters."""
        # Mock the clustering function
        with patch.object(self.clustering_processor, '_cluster_embeddings') as mock_cluster:
            mock_clusters = [
                SpeakerCluster(id="cluster1", embeddings=self.test_embeddings[:5]),
                SpeakerCluster(id="cluster2", embeddings=self.test_embeddings[5:])
            ]
            mock_cluster.return_value = mock_clusters
            
            # Call the cluster_embeddings method with min_clusters=1 and max_clusters=3
            result = self.clustering_processor.cluster_embeddings(
                self.test_embeddings, min_clusters=1, max_clusters=3
            )
            
            # Verify the mock was called with the right arguments
            mock_cluster.assert_called_once()
            args, kwargs = mock_cluster.call_args
            self.assertEqual(kwargs.get('min_clusters'), 1)
            self.assertEqual(kwargs.get('max_clusters'), 3)
    
    def test_error_handling(self):
        """Test error handling during clustering."""
        # Mock the clustering function to raise an exception
        with patch.object(self.clustering_processor, '_cluster_embeddings', side_effect=Exception("Test error")):
            # The method should raise a ClusteringError
            with self.assertRaises(ClusteringError):
                self.clustering_processor.cluster_embeddings(self.test_embeddings)
    
    def test_invalid_input(self):
        """Test handling of invalid input."""
        # Test with empty embeddings list
        with self.assertRaises(ClusteringError):
            self.clustering_processor.cluster_embeddings([])
        
        # Test with invalid min/max clusters
        with self.assertRaises(ClusteringError):
            self.clustering_processor.cluster_embeddings(
                self.test_embeddings, min_clusters=5, max_clusters=2
            )
        
        # Test with num_clusters > number of embeddings
        with self.assertRaises(ClusteringError):
            self.clustering_processor.cluster_embeddings(
                self.test_embeddings[:3], num_clusters=5
            )
    
    def test_cluster_quality(self):
        """Test cluster quality assessment."""
        # Create two clusters
        cluster1 = SpeakerCluster(id="cluster1", embeddings=self.test_embeddings[:5])
        cluster2 = SpeakerCluster(id="cluster2", embeddings=self.test_embeddings[5:])
        
        # Mock the _compute_cluster_quality method
        with patch.object(self.clustering_processor, '_compute_cluster_quality', return_value=0.85):
            # Call the compute_cluster_quality method
            quality = self.clustering_processor.compute_cluster_quality([cluster1, cluster2])
            
            # Verify the result
            self.assertEqual(quality, 0.85)
    
    def test_merge_clusters(self):
        """Test merging clusters."""
        # Create two clusters
        cluster1 = SpeakerCluster(id="cluster1", embeddings=self.test_embeddings[:5])
        cluster2 = SpeakerCluster(id="cluster2", embeddings=self.test_embeddings[5:])
        
        # Merge the clusters
        merged = self.clustering_processor.merge_clusters([cluster1, cluster2])
        
        # Verify the result
        self.assertEqual(len(merged.embeddings), 10)
        self.assertIsInstance(merged, SpeakerCluster)


if __name__ == '__main__':
    unittest.main() 