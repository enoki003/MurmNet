#!/usr/bin/env python3
"""
Visualize blackboard history for analysis.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


def load_history(file_path: str) -> Dict:
    """Load task history from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_timeline(history: Dict, output_path: str) -> None:
    """
    Plot timeline of agent activities.
    
    Args:
        history: Task history dict
        output_path: Output image path
    """
    entries = history.get("history", {}).get("entries", [])
    
    if not entries:
        print("No entries found in history")
        return
    
    # Extract data
    timestamps = []
    agents = []
    entry_types = []
    
    for entry in entries:
        timestamps.append(datetime.fromisoformat(entry["timestamp"]))
        agents.append(entry["agent_id"])
        entry_types.append(entry["entry_type"])
    
    # Get unique agents
    unique_agents = list(set(agents))
    agent_to_idx = {agent: i for i, agent in enumerate(unique_agents)}
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot each entry
    colors = plt.cm.tab10(range(len(unique_agents)))
    
    for i, (ts, agent, etype) in enumerate(zip(timestamps, agents, entry_types)):
        y = agent_to_idx[agent]
        color = colors[y]
        ax.scatter(ts, y, c=[color], s=100, alpha=0.7, edgecolors='black')
        
        # Add entry type label
        if i % 3 == 0:  # Show every 3rd label to avoid clutter
            ax.text(ts, y + 0.1, etype, fontsize=8, rotation=45, ha='left')
    
    # Format plot
    ax.set_yticks(range(len(unique_agents)))
    ax.set_yticklabels(unique_agents)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Agent", fontsize=12)
    ax.set_title(f"Agent Activity Timeline - Task {history.get('task_id', 'Unknown')}", fontsize=14)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.xticks(rotation=45)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=150)
    print(f"Timeline saved to: {output_path}")


def plot_entry_type_distribution(history: Dict, output_path: str) -> None:
    """
    Plot distribution of entry types.
    
    Args:
        history: Task history dict
        output_path: Output image path
    """
    entries = history.get("history", {}).get("entries", [])
    
    if not entries:
        print("No entries found in history")
        return
    
    # Count entry types
    entry_type_counts = {}
    for entry in entries:
        etype = entry["entry_type"]
        entry_type_counts[etype] = entry_type_counts.get(etype, 0) + 1
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    types = list(entry_type_counts.keys())
    counts = list(entry_type_counts.values())
    
    bars = ax.bar(types, counts, color='skyblue', edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    ax.set_xlabel("Entry Type", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Entry Type Distribution - Task {history.get('task_id', 'Unknown')}", fontsize=14)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=150)
    print(f"Distribution plot saved to: {output_path}")


def generate_text_report(history: Dict, output_path: str) -> None:
    """
    Generate text report of task execution.
    
    Args:
        history: Task history dict
        output_path: Output text file path
    """
    entries = history.get("history", {}).get("entries", [])
    
    lines = [
        "=" * 80,
        f"MurmurNet Task Execution Report",
        "=" * 80,
        "",
        f"Task ID: {history.get('task_id', 'Unknown')}",
        f"Status: {history.get('status', 'Unknown')}",
        f"Created: {history.get('created_at', 'Unknown')}",
        f"Updated: {history.get('updated_at', 'Unknown')}",
        f"Total Entries: {len(entries)}",
        "",
        "=" * 80,
        "Entry Timeline",
        "=" * 80,
        "",
    ]
    
    for i, entry in enumerate(entries, 1):
        lines.append(f"[{i}] {entry['timestamp']}")
        lines.append(f"    Agent: {entry['agent_id']}")
        lines.append(f"    Type: {entry['entry_type']}")
        lines.append(f"    Content: {str(entry['content'])[:200]}...")
        lines.append("")
    
    lines.append("=" * 80)
    
    report = "\n".join(lines)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Text report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize MurmurNet blackboard history")
    parser.add_argument(
        "history_file",
        help="Path to task history JSON file",
    )
    parser.add_argument(
        "--output-dir",
        default="./visualization_output",
        help="Output directory for visualizations",
    )
    
    args = parser.parse_args()
    
    # Load history
    history = load_history(args.history_file)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    task_id = history.get("task_id", "unknown")
    
    # Generate visualizations
    print("Generating visualizations...")
    
    plot_timeline(
        history,
        str(output_dir / f"{task_id}_timeline.png")
    )
    
    plot_entry_type_distribution(
        history,
        str(output_dir / f"{task_id}_distribution.png")
    )
    
    generate_text_report(
        history,
        str(output_dir / f"{task_id}_report.txt")
    )
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
