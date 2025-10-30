#!/usr/bin/env bash
# Setup script for MurmurNet

set -e

echo "=================================================="
echo "MurmurNet Setup Script"
echo "=================================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.10"

echo "Checking Python version..."
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "✓ Python $PYTHON_VERSION detected"
else
    echo "✗ Python 3.10 or higher is required"
    exit 1
fi

echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""

# Copy .env.example to .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo "✓ .env file created"
    echo "  Please edit .env to configure your settings"
else
    echo "✓ .env file already exists"
fi

echo ""

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p logs
mkdir -p data/zim
mkdir -p data/vector_db
mkdir -p data/memory_db
mkdir -p evaluation/results
mkdir -p evaluation/benchmarks
mkdir -p models
echo "✓ Directories created"

echo ""
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Edit .env file to configure your settings"
echo ""
echo "3. (Optional) Download Wikipedia ZIM file for RAG:"
echo "   wget -P data/zim https://download.kiwix.org/zim/wikipedia/wikipedia_ja_all_maxi_latest.zim"
echo ""
echo "4. Start the server:"
echo "   python -m src.main"
echo ""
echo "5. Or run with Docker:"
echo "   docker-compose up -d"
echo ""
