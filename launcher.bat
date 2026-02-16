@echo off
REM EVK4 Dashboard Launcher
REM Pixi handles the PYTHONPATH configuration from pixi.toml

echo ========================================
echo EVK4 Event Camera Dashboard
echo ========================================
echo.

cd /d "%~dp0"

REM Check if Pixi is installed
pixi --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Pixi is not installed or not in PATH
    echo.
    echo Please install Pixi first:
    echo https://pixi.sh/
    echo.
    pause
    exit /b 1
)

echo Starting dashboard...
echo.

pixi run start

REM Keep window open if there's an error
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Dashboard closed with error code: %ERRORLEVEL%
    pause
)