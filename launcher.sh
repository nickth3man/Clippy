#!/bin/bash

# Clippy Unix Launcher

# Set application root to the directory containing this script
APP_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$APP_ROOT"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Check for dependencies
python -c "import pkg_resources; pkg_resources.require(open('requirements.txt'))" >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Check for FFmpeg
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "FFmpeg not found in PATH. Checking for bundled version..."
    if [ -f "tools/ffmpeg" ]; then
        export PATH="$APP_ROOT/tools:$PATH"
        chmod +x "$APP_ROOT/tools/ffmpeg"
    else
        echo "WARNING: FFmpeg not found. Audio processing functionality may be limited."
    fi
fi

# Launch the application
echo "Starting Clippy..."
python -m app.main "$@"

# Handle exit
if [ $? -ne 0 ]; then
    echo
    echo "Clippy exited with error code $?"
    read -p "Press enter to continue..."
fi 