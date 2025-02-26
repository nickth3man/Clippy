#!/usr/bin/env python3
"""
Clippy: Multi-speaker Voice Database Builder
Main application entry point
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path.home() / '.clippy' / 'clippy.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup the application environment."""
    # Create necessary directories if they don't exist
    user_data_dir = Path.home() / '.clippy'
    user_data_dir.mkdir(exist_ok=True)
    
    # Ensure logs directory exists
    logs_dir = user_data_dir / 'logs'
    logs_dir.mkdir(exist_ok=True)
    
    # Set up environment variables
    os.environ['CLIPPY_DATA_DIR'] = str(user_data_dir)
    
    logger.info(f"Environment setup complete. Data directory: {user_data_dir}")


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Clippy: Multi-speaker Voice Database Builder"
    )
    parser.add_argument(
        "--cli", 
        action="store_true", 
        help="Run in command-line interface mode"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode"
    )
    args, remaining_args = parser.parse_known_args()
    
    # Setup environment
    setup_environment()
    
    # Set debug level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    try:
        if args.cli:
            logger.info("Starting Clippy in CLI mode")
            from app.cli.main import cli_main
            return cli_main(remaining_args)
        else:
            logger.info("Starting Clippy in GUI mode")
            from app.gui.main import gui_main
            return gui_main(remaining_args)
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        print(f"Error: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
        return 1
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        print(f"An unexpected error occurred: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 