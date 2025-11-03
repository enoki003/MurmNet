#!/usr/bin/env python3
"""
Test script for MurmurNet API client.
Tests query submission and retrieval.
"""

import json
import time
import requests

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    print("=" * 70)
    print("HEALTH CHECK TEST")
    print("=" * 70)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_query():
    """Test query submission."""
    print("=" * 70)
    print("QUERY SUBMISSION TEST")
    print("=" * 70)
    
    query = "What is the capital of France?"
    print(f"Query: {query}\n")
    
    # Submit query
    response = requests.post(
        f"{API_URL}/query",
        json={"query": query}
    )
    
    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    
    if response.status_code == 200:
        task_id = result.get("task_id")
        print(f"\n✓ Query submitted successfully")
        print(f"Task ID: {task_id}")
        
        # Wait and check status
        print("\nWaiting for processing...")
        for i in range(30):  # Wait up to 5 minutes
            time.sleep(10)
            status_response = requests.get(f"{API_URL}/query/{task_id}")
            status_result = status_response.json()
            
            current_status = status_result.get("status")
            print(f"  [{i*10}s] Status: {current_status}")
            
            if current_status == "completed":
                print("\n" + "=" * 70)
                print("QUERY RESULT")
                print("=" * 70)
                print(f"Answer: {status_result.get('answer', 'No answer')}")
                print(f"Execution Time: {status_result.get('execution_time', 0):.2f}s")
                return True
            elif current_status == "failed":
                print(f"\n✗ Query failed: {status_result.get('error', 'Unknown error')}")
                return False
        
        print("\n⚠ Timeout waiting for result")
        return False
    else:
        print(f"\n✗ Failed to submit query: {result}")
        return False

if __name__ == "__main__":
    test_health()
    print()
    test_query()
