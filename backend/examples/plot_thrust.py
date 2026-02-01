"""
Simple thrust visualization - reads NDJSON and plots thrust over time.

Usage (from backend directory):

    uv run ../scenarios/debug.py > output.ndjson
    uv run -m examples.plot_thrust output.ndjson

    # Save to file instead of showing:
    uv run -m examples.plot_thrust output.ndjson --output thrust.png
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt

from .ndjson_reader import load_flight


def main():
    parser = argparse.ArgumentParser(
        description="Plot thrust from simulation messages"
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        type=str,
        help="Input NDJSON file (if omitted, reads from stdin)",
    )
    parser.add_argument(
        "--output", "-o", type=str, help="Output file (PNG, PDF, etc.)"
    )
    args = parser.parse_args()

    data = load_flight(args.input_file)

    if len(data.times) == 0:
        print("No thrust data found in messages.", file=sys.stderr)
        return

    thrust_mag = np.linalg.norm(data.thrust_body, axis=1)

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(data.times, thrust_mag, "r-", linewidth=2, label="Thrust")
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Thrust (N)", fontsize=12)
    plt.title("Rocket Thrust Over Time", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150)
        print(f"Plot saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
