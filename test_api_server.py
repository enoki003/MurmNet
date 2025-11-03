#!/usr/bin/env python
"""
Test API server startup and basic endpoints.
"""

import time
import sys
import subprocess
import requests
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger


def test_server_startup():
    """Test that the server can start."""
    print("=" * 70)
    print("API SERVER STARTUP TEST")
    print("=" * 70)
    
    # Start server in background
    print("\n[1/4] Starting server...")
    server_process = subprocess.Popen(
        [".venv/bin/python", "src/main.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd="/home/enoki003/study/MurmurNet"
    )
    
    # Wait for server to start
    print("  Waiting for server to initialize (10 seconds)...")
    time.sleep(10)
    
    # Check if process is still running
    if server_process.poll() is not None:
        stdout, stderr = server_process.communicate()
        print(f"  ✗ Server failed to start")
        print(f"  Error: {stderr.decode()[:500]}")
        return False, server_process
    
    print("  ✓ Server process started")
    
    # Test health endpoint
    print("\n[2/4] Testing health endpoint...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print(f"  ✓ Health check successful")
            print(f"    Response: {response.json()}")
        else:
            print(f"  ✗ Health check failed with status {response.status_code}")
            return False, server_process
    except Exception as e:
        print(f"  ✗ Failed to connect to health endpoint: {e}")
        return False, server_process
    
    # Test statistics endpoint
    print("\n[3/4] Testing statistics endpoint...")
    try:
        response = requests.get("http://localhost:8000/statistics", timeout=5)
        if response.status_code == 200:
            print(f"  ✓ Statistics endpoint successful")
            stats = response.json()
            print(f"    Total tasks: {stats.get('total_tasks', 0)}")
        else:
            print(f"  ✗ Statistics failed with status {response.status_code}")
            return False, server_process
    except Exception as e:
        print(f"  ✗ Failed to connect to statistics endpoint: {e}")
        return False, server_process
    
    # Test root endpoint
    print("\n[4/4] Testing root endpoint...")
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            print(f"  ✓ Root endpoint successful")
        else:
            print(f"  ✗ Root failed with status {response.status_code}")
            return False, server_process
    except Exception as e:
        print(f"  ✗ Failed to connect to root endpoint: {e}")
        return False, server_process
    
    return True, server_process


def main():
    """Run server tests."""
    success, server_process = test_server_startup()
    
    # Cleanup
    print("\n" + "=" * 70)
    print("CLEANUP")
    print("=" * 70)
    print("Stopping server...")
    
    if server_process:
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
            print("✓ Server stopped")
        except subprocess.TimeoutExpired:
            server_process.kill()
            print("✓ Server killed")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if success:
        print("✓ All API endpoint tests passed!")
        print("\nServer is ready for use:")
        print("  Start: python src/main.py")
        print("  Test: python scripts/client.py")
        return 0
    else:
        print("✗ Some tests failed")
        print("\nCheck the logs for details:")
        print("  tail -f logs/murmurnet.log")
        return 1


if __name__ == "__main__":
    sys.exit(main())
