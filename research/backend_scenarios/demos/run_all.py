#!/usr/bin/env python3
"""
Run all demo scenarios and generate output files.

Usage:
    uv run scenarios/demos/run_all.py

Output:
    /tmp/simulacra/demo_*.ndjson

Then view in browser:
    http://localhost:5174/?file=demo_freefall.ndjson&file=demo_hover_balance.ndjson
"""

import sys
import os
import pathlib
import subprocess

# List of demo scripts to run
DEMOS = [
    # Clean physics demos
    "freefall.py",
    "hover_balance.py",
    "tilt_tipover.py",
    "thrust_torque.py",
    "gimbal_pulse.py",
    "spin.py",
    "landing.py",
    "tvc_recovery.py",
    # Noisy physics demos
    "noisy_hover.py",
    "thrust_jitter.py",
    "sensor_drift.py",
]

def main():
    script_dir = pathlib.Path(__file__).parent
    output_dir = pathlib.Path("/tmp/simulacra")
    output_dir.mkdir(exist_ok=True)

    print("Running all demo scenarios...\n")

    for demo in DEMOS:
        demo_path = script_dir / demo
        print(f"  Running {demo}...", end=" ", flush=True)

        try:
            result = subprocess.run(
                ["uv", "run", str(demo_path)],
                capture_output=True,
                text=True,
                cwd=script_dir.parent.parent,  # backend/
            )
            if result.returncode == 0:
                print("OK")
            else:
                print(f"FAILED\n    {result.stderr.strip()}")
        except Exception as e:
            print(f"ERROR: {e}")

    print("\n" + "=" * 60)
    print("Generated files:")
    for f in sorted(output_dir.glob("demo_*.ndjson")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:30s} ({size_kb:.1f} KB)")

    print("\n" + "=" * 60)
    print("View in browser:")
    print()

    # Individual demos
    print("Individual demos:")
    for demo in DEMOS:
        name = demo.replace(".py", "")
        print(f"  http://localhost:5174/?file=demo_{name}.ndjson")

    print()
    print("All demos (compare):")
    files = "&".join(f"file=demo_{d.replace('.py', '')}.ndjson" for d in DEMOS)
    print(f"  http://localhost:5174/?{files}")

    print()
    print("Physics comparisons:")
    print()
    print("  Freefall vs Hover:")
    print("    http://localhost:5173/?file=demo_freefall.ndjson&file=demo_hover_balance.ndjson")
    print()
    print("  Uncontrolled vs Controlled (tilt):")
    print("    http://localhost:5173/?file=demo_tilt_tipover.ndjson&file=demo_tvc_recovery.ndjson")
    print()
    print("  Clean vs Noisy hover:")
    print("    http://localhost:5173/?file=demo_hover_balance.ndjson&file=demo_noisy_hover.ndjson")
    print()
    print("  Thrust jitter without vs with TVC:")
    print("    http://localhost:5173/?file=demo_thrust_jitter.ndjson&file=demo_noisy_hover.ndjson")
    print()
    print("  Sensor drift effect:")
    print("    http://localhost:5173/?file=demo_sensor_drift.ndjson")


if __name__ == "__main__":
    main()
