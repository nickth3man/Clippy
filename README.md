# Clippy: Multi-speaker Voice Database Builder

Clippy is a tool for building and managing a database of speaker profiles from multi-speaker audio recordings. It separates individual voices from conversations, extracts speaker embeddings, and clusters them to identify and track unique speakers across multiple recordings.

## Features

- **Voice Separation**: Extract individual voices from conversation recordings
- **Speaker Identification**: Recognize the same speakers across different recordings
- **Profile Management**: Build and maintain a database of speaker profiles
- **Cross-Recording Analysis**: Track speaker appearances across multiple recordings
- **User-Friendly Interface**: Both CLI and GUI options for interacting with the system

## Installation

### Portable Installation (Recommended)

1. Download the latest Clippy release package
2. Extract the package to any location
3. Run the launcher for your platform:
   - Windows: `launcher.bat`
   - macOS/Linux: `launcher.sh`

### Manual Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/clippy.git
   cd clippy
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

5. Install platform-specific dependencies:
   - FFmpeg (required for audio processing)
   - CUDA drivers (optional, for GPU acceleration)

## Quick Start

### Using the GUI

1. Run `python -m app.main`
2. Import an audio file using the "Import" button
3. Process the audio using the "Process" button
4. Explore the separated speakers and their profiles

### Using the CLI

Basic usage:
```
python -m app.cli process path/to/audio.mp3
```

For more options:
```
python -m app.cli --help
```

## System Requirements

- **Minimum**:
  - Python 3.8 or later
  - 4GB RAM
  - 1GB disk space
  - Dual-core CPU

- **Recommended**:
  - 8GB RAM
  - SSD with 10GB free space
  - Quad-core CPU
  - CUDA-compatible GPU with 4GB VRAM

## Documentation

For full documentation, see the `docs/` directory or visit [project website].

## License

[License information]

## Acknowledgments

This project builds upon several open-source libraries:
- SVoice for voice separation
- WhisperX for diarization
- [Other acknowledgments] 