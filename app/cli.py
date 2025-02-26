"""
Command Line Interface for Clippy

This module provides a command-line interface for the Clippy application.
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.core import (
    Pipeline, 
    ProcessingResult, 
    PipelineError,
    SpeakerProfile
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def progress_callback(message: str, progress: float) -> None:
    """
    Callback function for progress updates.
    
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


def process_audio(args: argparse.Namespace) -> None:
    """
    Process an audio file.
    
    Args:
        args: Command-line arguments
    """
    file_path = Path(args.file)
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return
    
    logger.info(f"Processing audio file: {file_path}")
    
    # Initialize pipeline
    pipeline = Pipeline(
        models_dir=args.models_dir,
        db_path=args.db_path,
        device=args.device
    )
    
    try:
        # Process recording
        result = pipeline.process_recording(
            file_path,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            progress_callback=progress_callback if not args.quiet else None
        )
        
        # Print result summary
        if not args.quiet:
            print("\nProcessing Summary:")
            print(f"  Recording ID: {result.recording_id}")
            print(f"  Duration: {result.duration:.2f} seconds")
            print(f"  Number of speakers: {result.num_speakers}")
            print(f"  Number of segments: {len(result.segments)}")
            print(f"  Number of profiles: {len(result.profiles)}")
            print(f"  Processing time: {result.processing_time:.2f} seconds")
            
            if result.errors:
                print("\nErrors:")
                for error in result.errors:
                    print(f"  - {error}")
                    
            if result.warnings:
                print("\nWarnings:")
                for warning in result.warnings:
                    print(f"  - {warning}")
                    
            if result.profiles:
                print("\nProfiles:")
                for profile in result.profiles:
                    print(f"  - {profile.name} (ID: {profile.profile_id})")
                    print(f"    Confidence: {profile.confidence:.2f}")
                    print(f"    Embeddings: {len(profile.embeddings)}")
        
        # Save result to JSON if requested
        if args.output:
            import json
            
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
                
            logger.info(f"Result saved to {output_path}")
            
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        
    finally:
        # Close pipeline
        pipeline.close()


def list_profiles(args: argparse.Namespace) -> None:
    """
    List all speaker profiles.
    
    Args:
        args: Command-line arguments
    """
    # Initialize pipeline
    pipeline = Pipeline(
        models_dir=args.models_dir,
        db_path=args.db_path
    )
    
    try:
        # Get profiles
        profiles = pipeline.list_profiles()
        
        if not profiles:
            print("No profiles found.")
            return
            
        print(f"Found {len(profiles)} profiles:")
        
        for i, profile in enumerate(profiles, 1):
            print(f"{i}. {profile['name']} (ID: {profile['profile_id']})")
            print(f"   Created: {profile['created_at']}")
            print(f"   Embeddings: {profile['num_embeddings']}")
            print(f"   Appearances: {profile['num_appearances']}")
            print()
            
    except Exception as e:
        logger.error(f"Failed to list profiles: {e}")
        
    finally:
        # Close pipeline
        pipeline.close()


def show_profile(args: argparse.Namespace) -> None:
    """
    Show details of a speaker profile.
    
    Args:
        args: Command-line arguments
    """
    # Initialize pipeline
    pipeline = Pipeline(
        models_dir=args.models_dir,
        db_path=args.db_path
    )
    
    try:
        # Get profile
        profile = pipeline.get_profile(args.profile_id)
        
        if not profile:
            print(f"Profile not found: {args.profile_id}")
            return
            
        print(f"Profile: {profile.name} (ID: {profile.profile_id})")
        print(f"Created: {profile.created_at}")
        print(f"Updated: {profile.updated_at}")
        print(f"Confidence: {profile.confidence:.2f}")
        print(f"Embeddings: {len(profile.embeddings)}")
        
        # Get appearances
        appearances = pipeline.get_profile_appearances(args.profile_id)
        
        if appearances:
            print(f"\nAppearances ({len(appearances)}):")
            
            for i, appearance in enumerate(appearances, 1):
                print(f"{i}. Recording: {appearance['recording_name']}")
                print(f"   Duration: {appearance['duration']:.2f} seconds")
                print(f"   Confidence: {appearance['confidence']:.2f}")
                print(f"   Date: {appearance['created_at']}")
                print()
                
    except Exception as e:
        logger.error(f"Failed to show profile: {e}")
        
    finally:
        # Close pipeline
        pipeline.close()


def delete_profile(args: argparse.Namespace) -> None:
    """
    Delete a speaker profile.
    
    Args:
        args: Command-line arguments
    """
    # Initialize pipeline
    pipeline = Pipeline(
        models_dir=args.models_dir,
        db_path=args.db_path
    )
    
    try:
        # Get profile
        profile = pipeline.get_profile(args.profile_id)
        
        if not profile:
            print(f"Profile not found: {args.profile_id}")
            return
            
        # Confirm deletion
        if not args.force:
            confirm = input(f"Are you sure you want to delete profile '{profile.name}'? (y/n): ")
            
            if confirm.lower() != 'y':
                print("Deletion cancelled.")
                return
                
        # Delete profile
        success = pipeline.delete_profile(args.profile_id)
        
        if success:
            print(f"Profile '{profile.name}' deleted.")
        else:
            print(f"Failed to delete profile '{profile.name}'.")
            
    except Exception as e:
        logger.error(f"Failed to delete profile: {e}")
        
    finally:
        # Close pipeline
        pipeline.close()


def backup_database(args: argparse.Namespace) -> None:
    """
    Backup the database.
    
    Args:
        args: Command-line arguments
    """
    # Initialize pipeline
    pipeline = Pipeline(
        models_dir=args.models_dir,
        db_path=args.db_path
    )
    
    try:
        # Backup database
        backup_path = pipeline.backup_database(args.output)
        
        print(f"Database backed up to: {backup_path}")
            
    except Exception as e:
        logger.error(f"Failed to backup database: {e}")
        
    finally:
        # Close pipeline
        pipeline.close()


def restore_database(args: argparse.Namespace) -> None:
    """
    Restore the database from a backup.
    
    Args:
        args: Command-line arguments
    """
    backup_path = Path(args.backup)
    
    if not backup_path.exists():
        logger.error(f"Backup file not found: {backup_path}")
        return
    
    # Initialize pipeline
    pipeline = Pipeline(
        models_dir=args.models_dir,
        db_path=args.db_path
    )
    
    try:
        # Confirm restoration
        if not args.force:
            confirm = input("Are you sure you want to restore the database? This will overwrite the current database. (y/n): ")
            
            if confirm.lower() != 'y':
                print("Restoration cancelled.")
                return
                
        # Restore database
        pipeline.restore_database(backup_path)
        
        print(f"Database restored from: {backup_path}")
            
    except Exception as e:
        logger.error(f"Failed to restore database: {e}")
        
    finally:
        # Close pipeline
        pipeline.close()


def show_stats(args: argparse.Namespace) -> None:
    """
    Show database statistics.
    
    Args:
        args: Command-line arguments
    """
    # Initialize pipeline
    pipeline = Pipeline(
        models_dir=args.models_dir,
        db_path=args.db_path
    )
    
    try:
        # Get stats
        stats = pipeline.get_database_stats()
        
        print("Database Statistics:")
        print(f"  Profiles: {stats['num_profiles']}")
        print(f"  Recordings: {stats['num_recordings']}")
        print(f"  Embeddings: {stats['num_embeddings']}")
        print(f"  Appearances: {stats['num_appearances']}")
        print(f"  Total recording duration: {stats['total_duration']:.2f} seconds")
        print(f"  Database size: {stats['db_size'] / (1024 * 1024):.2f} MB")
        print(f"  Created: {stats['created_at']}")
        print(f"  Last modified: {stats['last_modified']}")
            
    except Exception as e:
        logger.error(f"Failed to show stats: {e}")
        
    finally:
        # Close pipeline
        pipeline.close()


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Clippy - Voice Database Builder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Global arguments
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Path to the models directory"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="clippy.db",
        help="Path to the database file"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to run models on"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    # Create subparsers
    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        help="Command to execute"
    )
    
    # Process command
    process_parser = subparsers.add_parser(
        "process",
        help="Process an audio file"
    )
    process_parser.add_argument(
        "file",
        type=str,
        help="Path to the audio file"
    )
    process_parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Minimum number of speakers"
    )
    process_parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Maximum number of speakers"
    )
    process_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the result to"
    )
    process_parser.set_defaults(func=process_audio)
    
    # List profiles command
    list_parser = subparsers.add_parser(
        "list",
        help="List all speaker profiles"
    )
    list_parser.set_defaults(func=list_profiles)
    
    # Show profile command
    show_parser = subparsers.add_parser(
        "show",
        help="Show details of a speaker profile"
    )
    show_parser.add_argument(
        "profile_id",
        type=str,
        help="Profile ID"
    )
    show_parser.set_defaults(func=show_profile)
    
    # Delete profile command
    delete_parser = subparsers.add_parser(
        "delete",
        help="Delete a speaker profile"
    )
    delete_parser.add_argument(
        "profile_id",
        type=str,
        help="Profile ID"
    )
    delete_parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation"
    )
    delete_parser.set_defaults(func=delete_profile)
    
    # Backup command
    backup_parser = subparsers.add_parser(
        "backup",
        help="Backup the database"
    )
    backup_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the backup to"
    )
    backup_parser.set_defaults(func=backup_database)
    
    # Restore command
    restore_parser = subparsers.add_parser(
        "restore",
        help="Restore the database from a backup"
    )
    restore_parser.add_argument(
        "backup",
        type=str,
        help="Path to the backup file"
    )
    restore_parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation"
    )
    restore_parser.set_defaults(func=restore_database)
    
    # Stats command
    stats_parser = subparsers.add_parser(
        "stats",
        help="Show database statistics"
    )
    stats_parser.set_defaults(func=show_stats)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 