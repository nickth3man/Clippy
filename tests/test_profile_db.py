"""
Tests for the ProfileDatabase class in the core module.
"""
import os
import unittest
import tempfile
from pathlib import Path
import numpy as np
from unittest.mock import patch, MagicMock

from app.core import ProfileDatabase, SpeakerProfile, ProfileDBError, SpeakerEmbedding


class TestProfileDatabase(unittest.TestCase):
    """Test cases for the ProfileDatabase class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for the test database
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        
        # Create a test database file
        self.db_path = self.test_dir / "test_profiles.db"
        
        # Initialize the database
        self.db = ProfileDatabase(self.db_path)
        
        # Create some test embeddings
        self.test_embeddings = []
        for i in range(5):
            vector = np.random.rand(512).astype(np.float32)
            self.test_embeddings.append(SpeakerEmbedding(vector))
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.db.close()
        self.temp_dir.cleanup()
    
    def test_create_profile(self):
        """Test creating a new speaker profile."""
        # Create a profile
        profile = self.db.create_profile(name="Test Speaker")
        
        # Verify the profile was created
        self.assertIsNotNone(profile)
        self.assertIsInstance(profile, SpeakerProfile)
        self.assertEqual(profile.name, "Test Speaker")
        self.assertEqual(len(profile.embeddings), 0)
        
        # Verify the profile is in the database
        profiles = self.db.list_profiles()
        self.assertEqual(len(profiles), 1)
        self.assertEqual(profiles[0]['id'], profile.id)
        self.assertEqual(profiles[0]['name'], "Test Speaker")
    
    def test_get_profile(self):
        """Test retrieving a profile by ID."""
        # Create a profile
        profile = self.db.create_profile(name="Test Speaker")
        
        # Retrieve the profile
        retrieved = self.db.get_profile(profile.id)
        
        # Verify the retrieved profile matches
        self.assertEqual(retrieved.id, profile.id)
        self.assertEqual(retrieved.name, profile.name)
    
    def test_update_profile(self):
        """Test updating a profile."""
        # Create a profile
        profile = self.db.create_profile(name="Test Speaker")
        
        # Update the profile name
        profile.name = "Updated Name"
        self.db.update_profile(profile)
        
        # Retrieve the profile and verify the update
        retrieved = self.db.get_profile(profile.id)
        self.assertEqual(retrieved.name, "Updated Name")
    
    def test_delete_profile(self):
        """Test deleting a profile."""
        # Create a profile
        profile = self.db.create_profile(name="Test Speaker")
        
        # Delete the profile
        result = self.db.delete_profile(profile.id)
        
        # Verify the deletion was successful
        self.assertTrue(result)
        
        # Verify the profile is no longer in the database
        profiles = self.db.list_profiles()
        self.assertEqual(len(profiles), 0)
        
        # Verify get_profile returns None
        self.assertIsNone(self.db.get_profile(profile.id))
    
    def test_add_embedding_to_profile(self):
        """Test adding an embedding to a profile."""
        # Create a profile
        profile = self.db.create_profile(name="Test Speaker")
        
        # Add an embedding
        embedding = self.test_embeddings[0]
        self.db.add_embedding_to_profile(profile.id, embedding)
        
        # Retrieve the profile and verify the embedding was added
        retrieved = self.db.get_profile(profile.id)
        self.assertEqual(len(retrieved.embeddings), 1)
        self.assertTrue(np.array_equal(retrieved.embeddings[0].vector, embedding.vector))
    
    def test_add_multiple_embeddings(self):
        """Test adding multiple embeddings to a profile."""
        # Create a profile
        profile = self.db.create_profile(name="Test Speaker")
        
        # Add multiple embeddings
        for embedding in self.test_embeddings:
            self.db.add_embedding_to_profile(profile.id, embedding)
        
        # Retrieve the profile and verify all embeddings were added
        retrieved = self.db.get_profile(profile.id)
        self.assertEqual(len(retrieved.embeddings), len(self.test_embeddings))
    
    def test_remove_embedding_from_profile(self):
        """Test removing an embedding from a profile."""
        # Create a profile and add embeddings
        profile = self.db.create_profile(name="Test Speaker")
        for embedding in self.test_embeddings:
            self.db.add_embedding_to_profile(profile.id, embedding)
        
        # Retrieve the profile to get the embedding IDs
        retrieved = self.db.get_profile(profile.id)
        embedding_id = retrieved.embeddings[0].id
        
        # Remove one embedding
        result = self.db.remove_embedding_from_profile(profile.id, embedding_id)
        
        # Verify the removal was successful
        self.assertTrue(result)
        
        # Verify the embedding was removed
        updated = self.db.get_profile(profile.id)
        self.assertEqual(len(updated.embeddings), len(self.test_embeddings) - 1)
    
    def test_find_matching_profile(self):
        """Test finding a matching profile for an embedding."""
        # Create profiles with embeddings
        profile1 = self.db.create_profile(name="Speaker 1")
        profile2 = self.db.create_profile(name="Speaker 2")
        
        # Add embeddings to profile1
        for embedding in self.test_embeddings[:3]:
            self.db.add_embedding_to_profile(profile1.id, embedding)
        
        # Add embeddings to profile2
        for embedding in self.test_embeddings[3:]:
            self.db.add_embedding_to_profile(profile2.id, embedding)
        
        # Mock the similarity computation to return high similarity for profile1
        with patch('app.core.profile_db.ProfileDatabase._compute_similarity', 
                  side_effect=[0.9, 0.6]):  # High similarity for profile1, lower for profile2
            
            # Find the matching profile for an embedding similar to profile1
            test_embedding = self.test_embeddings[0]
            match = self.db.find_matching_profile(test_embedding)
            
            # Verify the match is profile1
            self.assertEqual(match.id, profile1.id)
    
    def test_error_handling(self):
        """Test error handling in the database operations."""
        # Test with invalid profile ID
        with self.assertRaises(ProfileDBError):
            self.db.get_profile("nonexistent_id")
        
        # Test with invalid embedding ID
        profile = self.db.create_profile(name="Test Speaker")
        with self.assertRaises(ProfileDBError):
            self.db.remove_embedding_from_profile(profile.id, "nonexistent_id")
    
    def test_database_backup_restore(self):
        """Test database backup and restore functionality."""
        # Create profiles with embeddings
        profile1 = self.db.create_profile(name="Speaker 1")
        profile2 = self.db.create_profile(name="Speaker 2")
        
        # Add embeddings to profiles
        for embedding in self.test_embeddings[:3]:
            self.db.add_embedding_to_profile(profile1.id, embedding)
        
        for embedding in self.test_embeddings[3:]:
            self.db.add_embedding_to_profile(profile2.id, embedding)
        
        # Create a backup
        backup_path = self.test_dir / "backup.db"
        self.db.backup_database(backup_path)
        
        # Verify the backup file exists
        self.assertTrue(backup_path.exists())
        
        # Create a new database from the backup
        restored_db = ProfileDatabase(backup_path)
        
        # Verify the restored database has the same profiles
        restored_profiles = restored_db.list_profiles()
        self.assertEqual(len(restored_profiles), 2)
        
        # Verify the profiles have the correct embeddings
        restored_profile1 = restored_db.get_profile(profile1.id)
        restored_profile2 = restored_db.get_profile(profile2.id)
        
        self.assertEqual(len(restored_profile1.embeddings), 3)
        self.assertEqual(len(restored_profile2.embeddings), 2)
        
        # Clean up
        restored_db.close()


if __name__ == '__main__':
    unittest.main() 