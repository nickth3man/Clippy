@echo off
SETLOCAL EnableDelayedExpansion

:: Clippy Windows Launcher
TITLE Clippy: Multi-speaker Voice Database Builder

:: Set application root to the directory containing this script
SET "APP_ROOT=%~dp0"
CD /D "%APP_ROOT%"

:: Create virtual environment if it doesn't exist
IF NOT EXIST "venv" (
    ECHO Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment
CALL venv\Scripts\activate.bat

:: Check for dependencies
python -c "import pkg_resources; pkg_resources.require(open('requirements.txt'))" > NUL 2>&1
IF %ERRORLEVEL% NEQ 0 (
    ECHO Installing dependencies...
    pip install -r requirements.txt
)

:: Check for FFmpeg
WHERE ffmpeg > NUL 2>&1
IF %ERRORLEVEL% NEQ 0 (
    ECHO FFmpeg not found in PATH. Checking for bundled version...
    IF EXIST "tools\ffmpeg.exe" (
        SET "PATH=%APP_ROOT%tools;%PATH%"
    ) ELSE (
        ECHO WARNING: FFmpeg not found. Audio processing functionality may be limited.
    )
)

:: Launch the application
ECHO Starting Clippy...
python -m app.main %*

:: Handle exit
IF %ERRORLEVEL% NEQ 0 (
    ECHO.
    ECHO Clippy exited with error code %ERRORLEVEL%
    PAUSE
)

ENDLOCAL 