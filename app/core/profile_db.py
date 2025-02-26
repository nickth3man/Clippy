"""
Speaker Profile Database Module

This module implements a database for storing and retrieving speaker profiles.
It provides capabilities to manage speaker profiles across multiple recordings.

Key features:
1. Speaker profile storage and retrieval
2. Profile versioning and confidence scoring
3. Metadata storage
4. Backup and restoration
"""

import os
import logging
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import sqlite3
import pickle
import datetime

import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

# Import local modules
from app.core.embedding import SpeakerEmbedding
from app.core.clustering import SpeakerCluster


class ProfileDBError(Exception):
    """Exception raised for profile database errors."""
    pass


class SpeakerProfile:
    """
    Represents a speaker profile with embeddings and metadata.
    """
    
    def __init__(self, 
                 profile_id: str = None,
                 name: str = None,
                 embeddings: List[SpeakerEmbedding] = None,
                 metadata: Dict = None,
                 confidence: float = 1.0):
        """
        Initialize a speaker profile.
        
        Args:
            profile_id: Profile identifier (optional)
            name: Speaker name (optional)
            embeddings: List of speaker embeddings (optional)
            metadata: Additional metadata (optional)
            confidence: Confidence score (0-1)
        """
        self.profile_id = profile_id or str(uuid.uuid4())
        self.name = name
        self.embeddings = embeddings or []
        self.metadata = metadata or {}
        self.confidence = confidence
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.version = 1
        
        # Calculate representative embedding
        self.representative_embedding = None
        if self.embeddings:
            self._update_representative_embedding()
    
    def add_embedding(self, embedding: SpeakerEmbedding) -> None:
        """
        Add an embedding to the profile.
        
        Args:
            embedding: Speaker embedding to add
        """
        self.embeddings.append(embedding)
        self._update_representative_embedding()
        self.updated_at = time.time()
        self.version += 1
    
    def remove_embedding(self, embedding: SpeakerEmbedding) -> bool:
        """
        Remove an embedding from the profile.
        
        Args:
            embedding: Speaker embedding to remove
            
        Returns:
            True if the embedding was removed, False otherwise
        """
        try:
            self.embeddings.remove(embedding)
            self._update_representative_embedding()
            self.updated_at = time.time()
            self.version += 1
            return True
        except ValueError:
            return False
    
    def _update_representative_embedding(self) -> None:
        """Update the representative embedding based on current embeddings."""
        if not self.embeddings:
            self.representative_embedding = None
            return
            
        # For now, just use the average of all embeddings
        # In a real implementation, this could be more sophisticated
        vectors = np.vstack([e.vector for e in self.embeddings])
        avg_vector = np.mean(vectors, axis=0)
        
        # Normalize
        norm = np.linalg.norm(avg_vector)
        if norm > 0:
            avg_vector = avg_vector / norm
            
        # Create representative embedding
        self.representative_embedding = SpeakerEmbedding(
            vector=avg_vector,
            speaker_id=self.profile_id,
            confidence=self.confidence,
            metadata={'is_representative': True, 'version': self.version}
        )
    
    def get_embedding_count(self) -> int:
        """Get the number of embeddings in the profile."""
        return len(self.embeddings)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation (without embeddings)."""
        return {
            'profile_id': self.profile_id,
            'name': self.name,
            'embedding_count': self.get_embedding_count(),
            'confidence': self.confidence,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'version': self.version,
            'metadata': self.metadata
        }
        
    def __repr__(self) -> str:
        """String representation."""
        return (f"SpeakerProfile(id={self.profile_id}, "
                f"name={self.name}, "
                f"embeddings={self.get_embedding_count()}, "
                f"version={self.version}, "
                f"confidence={self.confidence:.2f})")


class ProfileDatabase:
    """
    Handles storage and retrieval of speaker profiles.
    
    Provides functionality to manage speaker profiles across multiple recordings.
    """
    
    def __init__(self, db_path: Union[str, Path] = None):
        """
        Initialize the ProfileDatabase.
        
        Args:
            db_path: Path to the database file (optional)
        """
        if db_path is None:
            # Use default location in user's home directory
            db_path = Path.home() / '.clippy' / 'profiles.db'
            
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize the database schema."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Create profiles table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS profiles (
                profile_id TEXT PRIMARY KEY,
                name TEXT,
                confidence REAL,
                created_at REAL,
                updated_at REAL,
                version INTEGER,
                metadata TEXT
            )
            ''')
            
            # Create embeddings table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                embedding_id TEXT PRIMARY KEY,
                profile_id TEXT,
                vector BLOB,
                confidence REAL,
                created_at REAL,
                metadata TEXT,
                FOREIGN KEY (profile_id) REFERENCES profiles (profile_id)
            )
            ''')
            
            # Create recordings table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS recordings (
                recording_id TEXT PRIMARY KEY,
                name TEXT,
                path TEXT,
                duration REAL,
                created_at REAL,
                metadata TEXT
            )
            ''')
            
            # Create appearances table (linking profiles to recordings)
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS appearances (
                appearance_id TEXT PRIMARY KEY,
                profile_id TEXT,
                recording_id TEXT,
                confidence REAL,
                duration REAL,
                metadata TEXT,
                FOREIGN KEY (profile_id) REFERENCES profiles (profile_id),
                FOREIGN KEY (recording_id) REFERENCES recordings (recording_id)
            )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info(f"Initialized profile database at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise ProfileDBError(f"Failed to initialize database: {e}")
    
    def add_profile(self, profile: SpeakerProfile) -> None:
        """
        Add a speaker profile to the database.
        
        Args:
            profile: Speaker profile to add
            
        Raises:
            ProfileDBError: If adding fails
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Insert profile
            cursor.execute('''
            INSERT OR REPLACE INTO profiles (
                profile_id, name, confidence, created_at, updated_at, version, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                profile.profile_id,
                profile.name,
                profile.confidence,
                profile.created_at,
                profile.updated_at,
                profile.version,
                json.dumps(profile.metadata)
            ))
            
            # Insert embeddings
            for embedding in profile.embeddings:
                cursor.execute('''
                INSERT OR REPLACE INTO embeddings (
                    embedding_id, profile_id, vector, confidence, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    str(uuid.uuid4()),
                    profile.profile_id,
                    pickle.dumps(embedding.vector),
                    embedding.confidence,
                    embedding.created_at,
                    json.dumps(embedding.metadata)
                ))
                
            conn.commit()
            conn.close()
            
            logger.info(f"Added profile {profile.profile_id} to database")
            
        except Exception as e:
            logger.error(f"Failed to add profile: {e}")
            raise ProfileDBError(f"Failed to add profile: {e}")
    
    def get_profile(self, profile_id: str) -> Optional[SpeakerProfile]:
        """
        Get a speaker profile from the database.
        
        Args:
            profile_id: Profile identifier
            
        Returns:
            Speaker profile or None if not found
            
        Raises:
            ProfileDBError: If retrieval fails
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Get profile
            cursor.execute('''
            SELECT profile_id, name, confidence, created_at, updated_at, version, metadata
            FROM profiles
            WHERE profile_id = ?
            ''', (profile_id,))
            
            profile_row = cursor.fetchone()
            if profile_row is None:
                conn.close()
                return None
                
            # Create profile object
            profile = SpeakerProfile(
                profile_id=profile_row[0],
                name=profile_row[1],
                confidence=profile_row[2],
                metadata=json.loads(profile_row[6])
            )
            profile.created_at = profile_row[3]
            profile.updated_at = profile_row[4]
            profile.version = profile_row[5]
            
            # Get embeddings
            cursor.execute('''
            SELECT vector, confidence, created_at, metadata
            FROM embeddings
            WHERE profile_id = ?
            ''', (profile_id,))
            
            embedding_rows = cursor.fetchall()
            for row in embedding_rows:
                vector = pickle.loads(row[0])
                confidence = row[1]
                created_at = row[2]
                metadata = json.loads(row[3])
                
                embedding = SpeakerEmbedding(
                    vector=vector,
                    speaker_id=profile_id,
                    confidence=confidence,
                    metadata=metadata
                )
                embedding.created_at = created_at
                
                profile.embeddings.append(embedding)
                
            # Update representative embedding
            profile._update_representative_embedding()
            
            conn.close()
            return profile
            
        except Exception as e:
            logger.error(f"Failed to get profile: {e}")
            raise ProfileDBError(f"Failed to get profile: {e}")
    
    def delete_profile(self, profile_id: str) -> bool:
        """
        Delete a speaker profile from the database.
        
        Args:
            profile_id: Profile identifier
            
        Returns:
            True if the profile was deleted, False otherwise
            
        Raises:
            ProfileDBError: If deletion fails
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Check if profile exists
            cursor.execute('''
            SELECT profile_id FROM profiles WHERE profile_id = ?
            ''', (profile_id,))
            
            if cursor.fetchone() is None:
                conn.close()
                return False
                
            # Delete embeddings
            cursor.execute('''
            DELETE FROM embeddings WHERE profile_id = ?
            ''', (profile_id,))
            
            # Delete appearances
            cursor.execute('''
            DELETE FROM appearances WHERE profile_id = ?
            ''', (profile_id,))
            
            # Delete profile
            cursor.execute('''
            DELETE FROM profiles WHERE profile_id = ?
            ''', (profile_id,))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Deleted profile {profile_id} from database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete profile: {e}")
            raise ProfileDBError(f"Failed to delete profile: {e}")
    
    def list_profiles(self) -> List[Dict]:
        """
        List all speaker profiles in the database.
        
        Returns:
            List of profile dictionaries
            
        Raises:
            ProfileDBError: If listing fails
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Get profiles
            cursor.execute('''
            SELECT profile_id, name, confidence, created_at, updated_at, version, metadata
            FROM profiles
            ORDER BY name, created_at
            ''')
            
            profiles = []
            for row in cursor.fetchall():
                profile_id = row[0]
                
                # Count embeddings
                cursor.execute('''
                SELECT COUNT(*) FROM embeddings WHERE profile_id = ?
                ''', (profile_id,))
                embedding_count = cursor.fetchone()[0]
                
                # Create profile dictionary
                profile_dict = {
                    'profile_id': profile_id,
                    'name': row[1],
                    'confidence': row[2],
                    'created_at': row[3],
                    'updated_at': row[4],
                    'version': row[5],
                    'metadata': json.loads(row[6]),
                    'embedding_count': embedding_count
                }
                
                profiles.append(profile_dict)
                
            conn.close()
            return profiles
            
        except Exception as e:
            logger.error(f"Failed to list profiles: {e}")
            raise ProfileDBError(f"Failed to list profiles: {e}")
    
    def search_profiles(self, 
                       embedding: SpeakerEmbedding,
                       threshold: float = 0.7,
                       limit: int = 5) -> List[Tuple[Dict, float]]:
        """
        Search for profiles matching an embedding.
        
        Args:
            embedding: Speaker embedding to search for
            threshold: Similarity threshold (0-1)
            limit: Maximum number of results
            
        Returns:
            List of (profile_dict, similarity) tuples
            
        Raises:
            ProfileDBError: If search fails
        """
        try:
            # Get all profiles
            profiles = []
            for profile_dict in self.list_profiles():
                profile = self.get_profile(profile_dict['profile_id'])
                if profile and profile.representative_embedding is not None:
                    profiles.append((profile, profile_dict))
                    
            # Calculate similarities
            similarities = []
            for profile, profile_dict in profiles:
                # Calculate similarity between embeddings
                vec1 = embedding.vector / np.linalg.norm(embedding.vector)
                vec2 = profile.representative_embedding.vector / np.linalg.norm(profile.representative_embedding.vector)
                similarity = np.dot(vec1, vec2)
                
                # Convert to range 0-1
                similarity = (similarity + 1) / 2
                
                if similarity >= threshold:
                    similarities.append((profile_dict, float(similarity)))
                    
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Limit results
            return similarities[:limit]
            
        except Exception as e:
            logger.error(f"Failed to search profiles: {e}")
            raise ProfileDBError(f"Failed to search profiles: {e}")
    
    def add_recording(self, 
                     recording_id: str,
                     name: str,
                     path: str,
                     duration: float,
                     metadata: Dict = None) -> None:
        """
        Add a recording to the database.
        
        Args:
            recording_id: Recording identifier
            name: Recording name
            path: Path to the recording file
            duration: Recording duration in seconds
            metadata: Additional metadata (optional)
            
        Raises:
            ProfileDBError: If adding fails
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Insert recording
            cursor.execute('''
            INSERT OR REPLACE INTO recordings (
                recording_id, name, path, duration, created_at, metadata
            ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                recording_id,
                name,
                path,
                duration,
                time.time(),
                json.dumps(metadata or {})
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Added recording {recording_id} to database")
            
        except Exception as e:
            logger.error(f"Failed to add recording: {e}")
            raise ProfileDBError(f"Failed to add recording: {e}")
    
    def add_appearance(self, 
                      profile_id: str,
                      recording_id: str,
                      confidence: float,
                      duration: float,
                      metadata: Dict = None) -> None:
        """
        Add a speaker appearance in a recording.
        
        Args:
            profile_id: Profile identifier
            recording_id: Recording identifier
            confidence: Confidence score (0-1)
            duration: Duration of appearance in seconds
            metadata: Additional metadata (optional)
            
        Raises:
            ProfileDBError: If adding fails
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Insert appearance
            cursor.execute('''
            INSERT OR REPLACE INTO appearances (
                appearance_id, profile_id, recording_id, confidence, duration, metadata
            ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()),
                profile_id,
                recording_id,
                confidence,
                duration,
                json.dumps(metadata or {})
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Added appearance of profile {profile_id} in recording {recording_id}")
            
        except Exception as e:
            logger.error(f"Failed to add appearance: {e}")
            raise ProfileDBError(f"Failed to add appearance: {e}")
    
    def get_profile_appearances(self, profile_id: str) -> List[Dict]:
        """
        Get all appearances of a profile.
        
        Args:
            profile_id: Profile identifier
            
        Returns:
            List of appearance dictionaries
            
        Raises:
            ProfileDBError: If retrieval fails
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Get appearances
            cursor.execute('''
            SELECT a.appearance_id, a.recording_id, a.confidence, a.duration, a.metadata,
                   r.name, r.path, r.duration as recording_duration, r.created_at, r.metadata as recording_metadata
            FROM appearances a
            JOIN recordings r ON a.recording_id = r.recording_id
            WHERE a.profile_id = ?
            ORDER BY r.created_at DESC
            ''', (profile_id,))
            
            appearances = []
            for row in cursor.fetchall():
                appearance_dict = {
                    'appearance_id': row[0],
                    'recording_id': row[1],
                    'confidence': row[2],
                    'duration': row[3],
                    'metadata': json.loads(row[4]),
                    'recording_name': row[5],
                    'recording_path': row[6],
                    'recording_duration': row[7],
                    'recording_created_at': row[8],
                    'recording_metadata': json.loads(row[9])
                }
                
                appearances.append(appearance_dict)
                
            conn.close()
            return appearances
            
        except Exception as e:
            logger.error(f"Failed to get profile appearances: {e}")
            raise ProfileDBError(f"Failed to get profile appearances: {e}")
    
    def get_recording(self, recording_id: str) -> Optional[Dict]:
        """
        Get a recording from the database.
        
        Args:
            recording_id: Recording identifier
            
        Returns:
            Recording dictionary or None if not found
            
        Raises:
            ProfileDBError: If retrieval fails
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Get recording
            cursor.execute('''
            SELECT recording_id, name, path, duration, created_at, metadata
            FROM recordings
            WHERE recording_id = ?
            ''', (recording_id,))
            
            row = cursor.fetchone()
            if row is None:
                conn.close()
                return None
                
            recording_dict = {
                'recording_id': row[0],
                'name': row[1],
                'file_path': row[2],
                'duration': row[3],
                'created_at': row[4],
                'metadata': json.loads(row[5])
            }
            
            conn.close()
            return recording_dict
            
        except Exception as e:
            logger.error(f"Failed to get recording: {e}")
            raise ProfileDBError(f"Failed to get recording: {e}")
    
    def get_recording_appearances(self, recording_id: str) -> List[Dict]:
        """
        Get all appearances in a recording.
        
        Args:
            recording_id: Recording identifier
            
        Returns:
            List of appearance dictionaries
            
        Raises:
            ProfileDBError: If retrieval fails
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Get appearances
            cursor.execute('''
            SELECT a.appearance_id, a.profile_id, a.confidence, a.duration, a.metadata,
                   p.name as profile_name
            FROM appearances a
            JOIN profiles p ON a.profile_id = p.profile_id
            WHERE a.recording_id = ?
            ORDER BY a.confidence DESC
            ''', (recording_id,))
            
            appearances = []
            for row in cursor.fetchall():
                appearance_dict = {
                    'appearance_id': row[0],
                    'profile_id': row[1],
                    'confidence': row[2],
                    'duration': row[3],
                    'metadata': json.loads(row[4]),
                    'profile_name': row[5]
                }
                
                appearances.append(appearance_dict)
                
            conn.close()
            return appearances
            
        except Exception as e:
            logger.error(f"Failed to get recording appearances: {e}")
            raise ProfileDBError(f"Failed to get recording appearances: {e}")
    
    def list_recordings(self) -> List[Dict]:
        """
        List all recordings in the database.
        
        Returns:
            List of recording dictionaries
            
        Raises:
            ProfileDBError: If listing fails
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Get recordings
            cursor.execute('''
            SELECT recording_id, name, path, duration, created_at, metadata
            FROM recordings
            ORDER BY name, created_at
            ''')
            
            recordings = []
            for row in cursor.fetchall():
                recording_id = row[0]
                
                # Create recording dictionary
                recording_dict = {
                    'recording_id': recording_id,
                    'name': row[1],
                    'file_path': row[2],
                    'duration': row[3],
                    'created_at': row[4],
                    'metadata': json.loads(row[5])
                }
                
                recordings.append(recording_dict)
                
            conn.close()
            return recordings
            
        except Exception as e:
            logger.error(f"Failed to list recordings: {e}")
            raise ProfileDBError(f"Failed to list recordings: {e}")
    
    def backup_database(self, backup_path: Union[str, Path] = None) -> str:
        """
        Create a backup of the database.
        
        Args:
            backup_path: Path to save the backup to (optional)
            
        Returns:
            Path to the backup file
            
        Raises:
            ProfileDBError: If backup fails
        """
        try:
            # Generate backup path if not provided
            if backup_path is None:
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = self.db_path.parent / f"profiles_backup_{timestamp}.db"
                
            backup_path = Path(backup_path)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create backup
            shutil.copy2(self.db_path, backup_path)
            
            logger.info(f"Created database backup at {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Failed to create database backup: {e}")
            raise ProfileDBError(f"Failed to create database backup: {e}")
    
    def restore_database(self, backup_path: Union[str, Path]) -> None:
        """
        Restore the database from a backup.
        
        Args:
            backup_path: Path to the backup file
            
        Raises:
            ProfileDBError: If restoration fails
        """
        try:
            backup_path = Path(backup_path)
            
            if not backup_path.exists():
                raise ProfileDBError(f"Backup file not found: {backup_path}")
                
            # Create backup of current database
            current_backup = self.backup_database()
            
            # Restore from backup
            shutil.copy2(backup_path, self.db_path)
            
            logger.info(f"Restored database from {backup_path} (current state backed up to {current_backup})")
            
        except Exception as e:
            logger.error(f"Failed to restore database: {e}")
            raise ProfileDBError(f"Failed to restore database: {e}")
    
    def get_database_stats(self) -> Dict:
        """
        Get statistics about the database.
        
        Returns:
            Dictionary with statistics
            
        Raises:
            ProfileDBError: If retrieval fails
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Get counts
            cursor.execute('SELECT COUNT(*) FROM profiles')
            profile_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM embeddings')
            embedding_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM recordings')
            recording_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM appearances')
            appearance_count = cursor.fetchone()[0]
            
            # Get database size
            db_size = self.db_path.stat().st_size
            
            # Get creation time
            cursor.execute('SELECT created_at FROM profiles ORDER BY created_at ASC LIMIT 1')
            row = cursor.fetchone()
            creation_time = row[0] if row else time.time()
            
            # Get last update time
            cursor.execute('SELECT updated_at FROM profiles ORDER BY updated_at DESC LIMIT 1')
            row = cursor.fetchone()
            last_update_time = row[0] if row else creation_time
            
            conn.close()
            
            return {
                'profile_count': profile_count,
                'embedding_count': embedding_count,
                'recording_count': recording_count,
                'appearance_count': appearance_count,
                'database_size_bytes': db_size,
                'creation_time': creation_time,
                'last_update_time': last_update_time,
                'database_path': str(self.db_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            raise ProfileDBError(f"Failed to get database stats: {e}")
    
    def create_profile_from_cluster(self, 
                                  cluster: SpeakerCluster,
                                  name: str = None,
                                  confidence: float = None) -> SpeakerProfile:
        """
        Create a speaker profile from a cluster.
        
        Args:
            cluster: Speaker cluster
            name: Profile name (optional)
            confidence: Confidence score (optional)
            
        Returns:
            Created speaker profile
            
        Raises:
            ProfileDBError: If creation fails
        """
        try:
            # Use cluster coherence as confidence if not provided
            if confidence is None:
                confidence = cluster.get_coherence()
                
            # Create profile
            profile = SpeakerProfile(
                name=name or f"Speaker {cluster.cluster_id}",
                embeddings=cluster.embeddings.copy(),
                confidence=confidence,
                metadata={'source_cluster_id': cluster.cluster_id}
            )
            
            # Add to database
            self.add_profile(profile)
            
            logger.info(f"Created profile {profile.profile_id} from cluster {cluster.cluster_id}")
            return profile
            
        except Exception as e:
            logger.error(f"Failed to create profile from cluster: {e}")
            raise ProfileDBError(f"Failed to create profile from cluster: {e}")
    
    def update_profile_from_cluster(self, 
                                  profile_id: str,
                                  cluster: SpeakerCluster,
                                  confidence: float = None) -> Optional[SpeakerProfile]:
        """
        Update a speaker profile from a cluster.
        
        Args:
            profile_id: Profile identifier
            cluster: Speaker cluster
            confidence: Confidence score (optional)
            
        Returns:
            Updated speaker profile or None if not found
            
        Raises:
            ProfileDBError: If update fails
        """
        try:
            # Get profile
            profile = self.get_profile(profile_id)
            if profile is None:
                return None
                
            # Update confidence if provided
            if confidence is not None:
                profile.confidence = confidence
                
            # Add embeddings from cluster
            for embedding in cluster.embeddings:
                profile.add_embedding(embedding)
                
            # Update profile in database
            self.add_profile(profile)
            
            logger.info(f"Updated profile {profile_id} from cluster {cluster.cluster_id}")
            return profile
            
        except Exception as e:
            logger.error(f"Failed to update profile from cluster: {e}")
            raise ProfileDBError(f"Failed to update profile from cluster: {e}")
    
    def close(self) -> None:
        """Close the database connection."""
        # SQLite connections are closed after each operation
        pass 