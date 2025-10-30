#!/usr/bin/env bash
# Run evaluation benchmarks

set -e

echo "Running MurmurNet Evaluation..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Run evaluation
python evaluation/run_evaluation.py "$@"
