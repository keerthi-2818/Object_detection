@echo off
REM Real-Time Object Detection Using YOLOv8 - Windows Startup Script
REM Internship ID: CITS2432
REM Run this script to set up and start the project

setlocal enabledelayedexpansion
cls

echo.
echo ======================================================================
echo   Real-Time Object Detection Using YOLOv8
echo   Internship ID: CITS2432
echo   Windows Setup Script
echo ======================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo.
    echo Please install Python 3.8+ from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

python --version

REM Check if virtual environment exists
if not exist "venv\" (
    echo.
    echo [STEP 1/5] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [SUCCESS] Virtual environment created
) else (
    echo.
    echo [INFO] Virtual environment already exists
)

REM Activate virtual environment
echo.
echo [STEP 2/5] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo [SUCCESS] Virtual environment activated

REM Upgrade pip
echo.
echo [STEP 3/5] Upgrading pip...
python -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo [WARNING] Failed to upgrade pip, continuing anyway...
)
echo [SUCCESS] pip upgraded

REM Install dependencies
echo.
echo [STEP 4/5] Installing dependencies...
echo This may take a few minutes...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    echo.
    echo Try running manually:
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)
echo [SUCCESS] Dependencies installed

REM Create required directories
echo.
echo [STEP 5/5] Creating required directories...
if not exist "models\" mkdir models
if not exist "outputs\" mkdir outputs
if not exist "data\" mkdir data
if not exist "logs\" mkdir logs
if not exist "config\" mkdir config
if not exist "assets\" mkdir assets
echo [SUCCESS] Directories created

REM Verify setup
echo.
echo ======================================================================
echo   Setup Complete!
echo ======================================================================
echo.
echo Optional: Verify your setup by running:
echo   python verify_setup.py
echo.
echo Starting Streamlit dashboard...
echo.

REM Start the application
streamlit run app.py

pause
