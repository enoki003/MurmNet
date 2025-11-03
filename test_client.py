#!/usr/bin/env python
"""
Simple client to test MurmurNet API.
"""

import time
import requests
import json

API_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("\n=== Health Check ===")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


def test_statistics():
    """Test statistics endpoint."""
    print("\n=== Statistics ===")
    response = requests.get(f"{API_URL}/statistics")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


def test_query(query_text):
    """Test query endpoint."""
    print(f"\n=== Query: {query_text} ===")
    
    payload = {"query": query_text}
    
    print("Sending query...")
    start = time.time()
    response = requests.post(f"{API_URL}/query", json=payload, timeout=300)
    elapsed = time.time() - start
    
    print(f"Status: {response.status_code}")
    print(f"Time: {elapsed:.2f}s")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nTask ID: {result.get('task_id')}")
        print(f"Success: {result.get('success')}")
        print(f"Answer: {result.get('answer', 'No answer')[:200]}...")
        return result.get('task_id')
    else:
        print(f"Error: {response.text}")
        return None


def test_task_history(task_id):
    """Test task history endpoint."""
    print(f"\n=== Task History: {task_id} ===")
    response = requests.get(f"{API_URL}/task/{task_id}/history")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        history = response.json()
        print(f"History entries: {len(history.get('history', {}).get('entries', []))}")


def main():
    print("=" * 70)
    print("MURMURNET API CLIENT TEST")
    print("=" * 70)
    
    # Wait for server
    print("\nWaiting for server to be ready...")
    for i in range(30):
        try:
            response = requests.get(f"{API_URL}/health", timeout=2)
            if response.status_code == 200:
                print("✓ Server is ready!")
                break
        except:
            time.sleep(1)
    else:
        print("✗ Server not available")
        return 1
    
    # Run tests
    test_health()
    test_statistics()
    
    # Test a simple query
    task_id = test_query("What is 2 + 2?")
    
    if task_id:
        test_task_history(task_id)
    
    print("\n" + "=" * 70)
    print("✓ API CLIENT TEST COMPLETE")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
