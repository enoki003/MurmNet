#!/usr/bin/env bash
# Start MurmurNet server

set -e

echo "Starting MurmurNet server..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if .env exists
if [ ! -f ".env" ]; then
    echo ".env file not found. Copying from .env.example..."
    cp .env.example .env
fi

# Start the server
python -m src.main
