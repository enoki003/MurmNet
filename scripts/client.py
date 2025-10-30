#!/usr/bin/env python3
"""
Simple client for testing MurmurNet API.
"""

import argparse
import json
import sys

import requests


def query(text: str, api_url: str = "http://localhost:8000") -> dict:
    """
    Send a query to MurmurNet API.
    
    Args:
        text: Query text
        api_url: API base URL
        
    Returns:
        Response dict
    """
    response = requests.post(
        f"{api_url}/query",
        json={"query": text},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def get_task_history(task_id: str, api_url: str = "http://localhost:8000") -> dict:
    """
    Get task history from API.
    
    Args:
        task_id: Task ID
        api_url: API base URL
        
    Returns:
        Task history dict
    """
    response = requests.get(
        f"{api_url}/task/{task_id}/history",
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def get_statistics(api_url: str = "http://localhost:8000") -> dict:
    """
    Get system statistics.
    
    Args:
        api_url: API base URL
        
    Returns:
        Statistics dict
    """
    response = requests.get(
        f"{api_url}/statistics",
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser(description="MurmurNet API Client")
    parser.add_argument(
        "action",
        choices=["query", "history", "stats"],
        help="Action to perform",
    )
    parser.add_argument(
        "--text",
        help="Query text (for 'query' action)",
    )
    parser.add_argument(
        "--task-id",
        help="Task ID (for 'history' action)",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="API base URL",
    )
    parser.add_argument(
        "--output",
        help="Output file path (JSON)",
    )
    
    args = parser.parse_args()
    
    try:
        if args.action == "query":
            if not args.text:
                print("Error: --text is required for 'query' action", file=sys.stderr)
                sys.exit(1)
            
            print(f"Sending query: {args.text}")
            result = query(args.text, args.api_url)
            
            print("\n" + "=" * 60)
            print("RESULT")
            print("=" * 60)
            print(f"Task ID: {result['task_id']}")
            print(f"Success: {result['success']}")
            print(f"Execution Time: {result['execution_time_seconds']:.2f}s")
            print("\nAnswer:")
            print(result.get('answer', result.get('error')))
            print("=" * 60)
            
        elif args.action == "history":
            if not args.task_id:
                print("Error: --task-id is required for 'history' action", file=sys.stderr)
                sys.exit(1)
            
            print(f"Fetching history for task: {args.task_id}")
            result = get_task_history(args.task_id, args.api_url)
            
        elif args.action == "stats":
            print("Fetching system statistics...")
            result = get_statistics(args.api_url)
        
        # Save to file if specified
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\nOutput saved to: {args.output}")
        
        # Print JSON for non-query actions
        if args.action != "query":
            print(json.dumps(result, ensure_ascii=False, indent=2))
        
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
