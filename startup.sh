#!/bin/bash

# Real-Time Object Detection Using YOLOv8 - Linux/macOS Startup Script
# Internship ID: CITS2432
# Run this script to set up and start the project

clear

echo ""
echo "======================================================================"
echo "  Real-Time Object Detection Using YOLOv8"
echo "  Internship ID: CITS2432"
echo "  Linux/macOS Setup Script"
echo "======================================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed"
    echo ""
    echo "Please install Python 3.8+ from https://www.python.org/"
    exit 1
fi

python3 --version

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "[STEP 1/5] Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtual environment"
        exit 1
    fi
    echo "[SUCCESS] Virtual environment created"
else
    echo ""
    echo "[INFO] Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "[STEP 2/5] Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to activate virtual environment"
    exit 1
fi
echo "[SUCCESS] Virtual environment activated"

# Upgrade pip
echo ""
echo "[STEP 3/5] Upgrading pip..."
python -m pip install --upgrade pip --quiet
if [ $? -ne 0 ]; then
    echo "[WARNING] Failed to upgrade pip, continuing anyway..."
fi
echo "[SUCCESS] pip upgraded"

# Install dependencies
echo ""
echo "[STEP 4/5] Installing dependencies..."
echo "This may take a few minutes..."
pip install -r requirements.txt --quiet
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install dependencies"
    echo ""
    echo "Try running manually:"
    echo "  pip install -r requirements.txt"
    exit 1
fi
echo "[SUCCESS] Dependencies installed"

# Create required directories
echo ""
echo "[STEP 5/5] Creating required directories..."
mkdir -p models outputs data logs config assets
echo "[SUCCESS] Directories created"

# Verify setup (optional)
echo ""
echo "======================================================================"
echo "  Setup Complete!"
echo "======================================================================"
echo ""
echo "Optional: Verify your setup by running:"
echo "  python verify_setup.py"
echo ""
echo "Starting Streamlit dashboard..."
echo ""

# Start the application
streamlit run app.py
