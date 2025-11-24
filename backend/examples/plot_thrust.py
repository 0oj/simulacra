"""
Simple thrust visualization - reads messages from stdin and plots thrust over time.

Usage (from backend directory):
    uv run ../scenarios/debug.py | uv run -m examples.plot_thrust
    
    # Save to file instead of showing:
    uv run ../scenarios/debug.py | uv run -m examples.plot_thrust --output thrust.png
"""

import sys
import os
import matplotlib.pyplot as plt
from typing import Optional
import numpy as np

# Add the src directory to the path so we can import simulacra
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(script_dir)
src_dir = os.path.join(backend_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from simulacra.schema import read_message, VariableMessage, MetadataMessage, SCHEMA_VERSION # noqa: E402


def plot_thrust(output_file: Optional[str] = None):
    """
    Read simulation messages from stdin and plot thrust over time.
    
    Args:
        output_file: Optional path to save plot (PNG, PDF, etc.)
    """
    times = []
    thrust_magnitudes = []
    
    # Read all messages
    try:
        while True:
            message = read_message()
            
            # Check schema version from metadata
            if isinstance(message, MetadataMessage):
                if message.schema_version != SCHEMA_VERSION:
                    print(f"Warning: Schema version mismatch. Expected {SCHEMA_VERSION}, got {message.schema_version}", file=sys.stderr)
            
            # Extract thrust data from variable messages
            if isinstance(message, VariableMessage):
                if message.thrust_vector is not None:
                    # Compute magnitude from thrust vector
                    thrust_mag = float(np.linalg.norm(message.thrust_vector))
                    times.append(message.timestamp)
                    thrust_magnitudes.append(thrust_mag)
                elif message.timestamp is not None:
                    # No thrust vector, but we have a timestamp - record as zero
                    times.append(message.timestamp)
                    thrust_magnitudes.append(0.0)
    
    except EOFError:
        # End of input - that's fine
        pass
    except Exception as e:
        print(f"Error reading messages: {e}", file=sys.stderr)
        return
    
    if len(times) == 0:
        print("No thrust data found in messages.", file=sys.stderr)
        return
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(times, thrust_magnitudes, 'r-', linewidth=2, label='Thrust')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Thrust (N)', fontsize=12)
    plt.title('Rocket Thrust Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse  # noqa: E402
    
    parser = argparse.ArgumentParser(description='Plot thrust from simulation messages')
    parser.add_argument('--output', '-o', type=str, help='Output file (PNG, PDF, etc.)')
    args = parser.parse_args()
    
    plot_thrust(args.output)
