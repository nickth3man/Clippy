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
            
        # Export separated voices if requested
        if args.export_voices:
            export_dir = args.export_voices
            try:
                output_files = pipeline.export_separated_voices(
                    result, 
                    export_dir,
                    format=args.export_format
                )
                
                if not args.quiet:
                    print(f"\nExported {len(output_files)} separated voice files to {export_dir}:")
                    for file_path in output_files:
                        print(f"  - {file_path.name}")
                        
            except Exception as e:
                logger.error(f"Failed to export separated voices: {e}")
            
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


def export_voices(args: argparse.Namespace) -> None:
    """
    Export separated voices from a previously processed recording.
    
    Args:
        args: Command-line arguments
    """
    # Initialize pipeline
    pipeline = Pipeline(
        models_dir=args.models_dir,
        db_path=args.db_path
    )
    
    try:
        # Get recording
        recording_id = args.recording_id
        
        # Get processing result
        result = pipeline.get_processing_result(recording_id)
        
        if not result:
            print(f"Recording not found: {recording_id}")
            return
            
        if not result.separated_voices:
            print(f"No separated voices found for recording: {recording_id}")
            return
            
        # Export voices
        output_files = pipeline.export_separated_voices(
            result,
            args.output_dir,
            format=args.format
        )
        
        print(f"Exported {len(output_files)} separated voice files to {args.output_dir}:")
        for file_path in output_files:
            print(f"  - {file_path.name}")
            
    except Exception as e:
        logger.error(f"Failed to export voices: {e}")
        
    finally:
        # Close pipeline
        pipeline.close()


def interactive_mode(args: argparse.Namespace) -> None:
    """
    Interactive mode for reviewing and correcting processing results.
    
    Args:
        args: Command line arguments
    """
    from app.core import Pipeline
    import readline  # For better input experience
    
    print("\n=== Clippy Interactive Mode ===")
    print("This mode allows you to review and correct processing results.\n")
    
    # Initialize pipeline
    pipeline = Pipeline(
        models_dir=args.models_dir,
        db_path=args.db_path,
        device=args.device
    )
    
    # Get all processing results
    results = pipeline.get_all_processing_results()
    
    if not results:
        print("No processing results found. Please process some recordings first.")
        return
    
    # Display available recordings
    print("Available recordings:")
    for i, result in enumerate(results):
        recording_path = result.recording_path or "Unknown path"
        print(f"{i+1}. {result.recording_id} - {os.path.basename(recording_path)}")
    
    # Select recording
    while True:
        try:
            selection = input("\nSelect a recording (number) or 'q' to quit: ")
            if selection.lower() == 'q':
                break
                
            idx = int(selection) - 1
            if 0 <= idx < len(results):
                selected_result = results[idx]
                _interactive_recording_menu(pipeline, selected_result)
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nExiting interactive mode...")
            break
    
    pipeline.close()


def _interactive_recording_menu(pipeline: 'Pipeline', result: 'ProcessingResult') -> None:
    """
    Interactive menu for a specific recording.
    
    Args:
        pipeline: Pipeline instance
        result: Processing result to work with
    """
    recording_path = result.recording_path or "Unknown path"
    print(f"\n=== Recording: {os.path.basename(recording_path)} ===")
    
    while True:
        print("\nOptions:")
        print("1. View detected speakers")
        print("2. Reassign speaker voices")
        print("3. Manage speaker profiles")
        print("4. Export separated voices")
        print("5. Back to recording selection")
        
        try:
            choice = input("\nEnter your choice: ")
            
            if choice == '1':
                _view_detected_speakers(pipeline, result)
            elif choice == '2':
                _reassign_speaker_voices(pipeline, result)
            elif choice == '3':
                _manage_speaker_profiles(pipeline, result)
            elif choice == '4':
                _export_voices_interactive(pipeline, result)
            elif choice == '5':
                break
            else:
                print("Invalid choice. Please try again.")
        except KeyboardInterrupt:
            print("\nReturning to recording selection...")
            break


def _view_detected_speakers(pipeline: 'Pipeline', result: 'ProcessingResult') -> None:
    """
    View detected speakers in a recording.
    
    Args:
        pipeline: Pipeline instance
        result: Processing result to work with
    """
    print("\n=== Detected Speakers ===")
    
    if not result.speakers:
        print("No speakers detected.")
        return
    
    for i, speaker in enumerate(result.speakers):
        speaker_id = speaker.get('id', 'Unknown')
        profile_id = speaker.get('profile_id', 'None')
        confidence = speaker.get('confidence', 0.0)
        speaking_time = speaker.get('speaking_time', 0.0)
        
        profile_name = "Unknown"
        if profile_id != 'None' and profile_id is not None:
            profile = pipeline.get_profile(profile_id)
            if profile:
                profile_name = profile.name or f"Profile {profile_id}"
        
        print(f"{i+1}. Speaker {speaker_id}")
        print(f"   Profile: {profile_name} ({profile_id})")
        print(f"   Confidence: {confidence:.2f}")
        print(f"   Speaking time: {speaking_time:.2f} seconds")


def _reassign_speaker_voices(pipeline: 'Pipeline', result: 'ProcessingResult') -> None:
    """
    Reassign speaker voices to different profiles.
    
    Args:
        pipeline: Pipeline instance
        result: Processing result to work with
    """
    print("\n=== Reassign Speaker Voices ===")
    
    if not result.speakers:
        print("No speakers detected.")
        return
    
    # List all speakers first
    _view_detected_speakers(pipeline, result)
    
    # List all available profiles
    profiles = pipeline.list_profiles()
    print("\nAvailable profiles:")
    for i, profile in enumerate(profiles):
        print(f"{i+1}. {profile.get('name', 'Unnamed')} ({profile.get('id')})")
    
    # Select speaker to reassign
    try:
        speaker_idx = int(input("\nSelect speaker to reassign (number): ")) - 1
        if not (0 <= speaker_idx < len(result.speakers)):
            print("Invalid speaker selection.")
            return
            
        profile_idx = int(input("Assign to profile (number) or 0 for new profile: ")) - 1
        
        if profile_idx == -1:
            # Create new profile
            name = input("Enter name for new profile: ")
            new_profile = pipeline.create_profile(name=name)
            profile_id = new_profile.id
        elif 0 <= profile_idx < len(profiles):
            profile_id = profiles[profile_idx].get('id')
        else:
            print("Invalid profile selection.")
            return
        
        # Update the speaker's profile
        speaker = result.speakers[speaker_idx]
        old_profile_id = speaker.get('profile_id')
        
        # Perform the reassignment
        success = pipeline.reassign_speaker(
            recording_id=result.recording_id,
            speaker_id=speaker.get('id'),
            new_profile_id=profile_id
        )
        
        if success:
            print(f"Speaker reassigned from profile {old_profile_id} to {profile_id}.")
        else:
            print("Failed to reassign speaker.")
            
    except ValueError:
        print("Please enter valid numbers.")
    except KeyboardInterrupt:
        print("\nCancelled.")


def _manage_speaker_profiles(pipeline: 'Pipeline', result: 'ProcessingResult') -> None:
    """
    Manage speaker profiles related to this recording.
    
    Args:
        pipeline: Pipeline instance
        result: Processing result to work with
    """
    print("\n=== Manage Speaker Profiles ===")
    
    while True:
        print("\nOptions:")
        print("1. Rename a profile")
        print("2. Merge profiles")
        print("3. View profile details")
        print("4. Back to recording menu")
        
        try:
            choice = input("\nEnter your choice: ")
            
            if choice == '1':
                # List all available profiles
                profiles = pipeline.list_profiles()
                print("\nAvailable profiles:")
                for i, profile in enumerate(profiles):
                    print(f"{i+1}. {profile.get('name', 'Unnamed')} ({profile.get('id')})")
                
                profile_idx = int(input("\nSelect profile to rename (number): ")) - 1
                if 0 <= profile_idx < len(profiles):
                    profile_id = profiles[profile_idx].get('id')
                    new_name = input("Enter new name: ")
                    success = pipeline.rename_profile(profile_id, new_name)
                    if success:
                        print(f"Profile renamed to {new_name}.")
                    else:
                        print("Failed to rename profile.")
                else:
                    print("Invalid profile selection.")
                    
            elif choice == '2':
                # List all available profiles
                profiles = pipeline.list_profiles()
                print("\nAvailable profiles:")
                for i, profile in enumerate(profiles):
                    print(f"{i+1}. {profile.get('name', 'Unnamed')} ({profile.get('id')})")
                
                source_idx = int(input("\nSelect source profile to merge (number): ")) - 1
                target_idx = int(input("Select target profile to merge into (number): ")) - 1
                
                if 0 <= source_idx < len(profiles) and 0 <= target_idx < len(profiles) and source_idx != target_idx:
                    source_id = profiles[source_idx].get('id')
                    target_id = profiles[target_idx].get('id')
                    
                    confirmation = input(f"Are you sure you want to merge {profiles[source_idx].get('name')} into {profiles[target_idx].get('name')}? (y/n): ")
                    if confirmation.lower() == 'y':
                        success = pipeline.merge_profiles(source_id, target_id)
                        if success:
                            print("Profiles merged successfully.")
                        else:
                            print("Failed to merge profiles.")
                else:
                    print("Invalid profile selection.")
                    
            elif choice == '3':
                # List all available profiles
                profiles = pipeline.list_profiles()
                print("\nAvailable profiles:")
                for i, profile in enumerate(profiles):
                    print(f"{i+1}. {profile.get('name', 'Unnamed')} ({profile.get('id')})")
                
                profile_idx = int(input("\nSelect profile to view (number): ")) - 1
                if 0 <= profile_idx < len(profiles):
                    profile_id = profiles[profile_idx].get('id')
                    profile = pipeline.get_profile(profile_id)
                    appearances = pipeline.get_profile_appearances(profile_id)
                    
                    print(f"\nProfile: {profile.name} ({profile.id})")
                    print(f"Created: {profile.created_at}")
                    print(f"Updated: {profile.updated_at}")
                    print(f"Quality: {profile.quality:.2f}")
                    print(f"Recordings: {len(appearances)}")
                    
                    print("\nAppearances:")
                    for i, appearance in enumerate(appearances):
                        recording_id = appearance.get('recording_id', 'Unknown')
                        confidence = appearance.get('confidence', 0.0)
                        speaking_time = appearance.get('speaking_time', 0.0)
                        print(f"{i+1}. Recording: {recording_id}")
                        print(f"   Confidence: {confidence:.2f}")
                        print(f"   Speaking time: {speaking_time:.2f} seconds")
                else:
                    print("Invalid profile selection.")
                    
            elif choice == '4':
                break
            else:
                print("Invalid choice. Please try again.")
                
        except ValueError:
            print("Please enter valid numbers.")
        except KeyboardInterrupt:
            print("\nReturning to recording menu...")
            break


def _export_voices_interactive(pipeline: 'Pipeline', result: 'ProcessingResult') -> None:
    """
    Export separated voices with interactive options.
    
    Args:
        pipeline: Pipeline instance
        result: Processing result to work with
    """
    print("\n=== Export Separated Voices ===")
    
    # Ask for export directory
    default_dir = os.path.join("data", "exports", result.recording_id)
    export_dir = input(f"Export directory [{default_dir}]: ") or default_dir
    
    # Ask for export format
    format_options = ["wav", "mp3", "flac"]
    format_str = ", ".join(format_options)
    format_choice = input(f"Export format [{format_str}] (default: wav): ") or "wav"
    
    if format_choice not in format_options:
        print(f"Invalid format. Using wav instead.")
        format_choice = "wav"
    
    # Perform the export
    try:
        exported_files = pipeline.export_separated_voices(
            result=result,
            output_dir=export_dir,
            format=format_choice
        )
        
        print(f"Successfully exported {len(exported_files)} voice files to {export_dir}")
        for i, file_path in enumerate(exported_files):
            print(f"{i+1}. {os.path.basename(file_path)}")
            
    except Exception as e:
        print(f"Error exporting voices: {str(e)}")


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
    process_parser.add_argument(
        "--export-voices",
        type=str,
        default=None,
        help="Directory to export separated voices to"
    )
    process_parser.add_argument(
        "--export-format",
        type=str,
        default="wav",
        choices=["wav", "mp3", "flac"],
        help="Format for exported voice files"
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
    
    # Export voices command
    export_parser = subparsers.add_parser(
        "export",
        help="Export separated voices from a recording"
    )
    export_parser.add_argument(
        "recording_id",
        type=str,
        help="Recording ID"
    )
    export_parser.add_argument(
        "--output-dir",
        type=str,
        default="data/exports",
        help="Directory to save the exported voices to"
    )
    export_parser.add_argument(
        "--format",
        type=str,
        default="wav",
        choices=["wav", "mp3", "flac"],
        help="Audio format for exported files"
    )
    export_parser.set_defaults(func=export_voices)
    
    # Interactive mode command
    interactive_parser = subparsers.add_parser(
        "interactive",
        help="Start interactive mode for reviewing and correcting results"
    )
    interactive_parser.set_defaults(func=interactive_mode)

    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 