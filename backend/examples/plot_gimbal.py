"""
Body-frame gimbal deflection animator.

Shows the thrust vector's transverse deflection in the body Y-Z plane,
as if looking down the rocket's nose toward the tail.

Center = thrust aligned with body +X (no deflection).
Displacement = gimbal deflection angle in degrees.

Usage (from backend directory):

    uv run ../scenarios/debug.py > output.ndjson
    uv run -m examples.plot_gimbal output.ndjson
    uv run -m examples.plot_gimbal output.ndjson --speed 2.0
    uv run -m examples.plot_gimbal output.ndjson --output gimbal.mp4
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .ndjson_reader import load_flight


def main():
    parser = argparse.ArgumentParser(
        description="Body-frame gimbal deflection animator"
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        type=str,
        help="Input NDJSON file (if omitted, reads from stdin)",
    )
    parser.add_argument(
        "--output", "-o", type=str, help="Save animation to file (e.g. gimbal.mp4)"
    )
    parser.add_argument(
        "--speed",
        "-s",
        type=float,
        default=1.0,
        help="Playback speed multiplier (default: 1.0 = real time)",
    )
    args = parser.parse_args()

    # ── Read data ────────────────────────────────────────────────────────
    data = load_flight(args.input_file)
    n_frames = len(data.times)
    if n_frames == 0:
        print("No trajectory data found.", file=sys.stderr)
        return

    # ── Compute deflection angles in degrees ─────────────────────────────
    # thrust_body columns: 0=Tx (axial), 1=Ty, 2=Tz
    Tx = data.thrust_body[:, 0]
    Ty = data.thrust_body[:, 1]
    Tz = data.thrust_body[:, 2]

    defl_y = np.degrees(np.arctan2(Ty, Tx))  # deflection in body Y
    defl_z = np.degrees(np.arctan2(Tz, Tx))  # deflection in body Z

    # Zero out deflection when there is no thrust (avoids meaningless 0/0).
    no_thrust = np.linalg.norm(data.thrust_body, axis=1) < 1e-6
    defl_y[no_thrust] = 0.0
    defl_z[no_thrust] = 0.0

    max_gimbal_deg = 15.0  # from scenario's max_outer_angle / max_inner_angle

    # ── Stage labels per frame ───────────────────────────────────────────
    debug_times = [t for t, _ in data.debug_msgs]
    debug_texts = [txt for _, txt in data.debug_msgs]
    frame_labels = [""] * n_frames
    if debug_times:
        j = 0
        for i in range(n_frames):
            t = data.times[i]
            while j < len(debug_times) - 1 and debug_times[j + 1] <= t:
                j += 1
            if debug_times[j] <= t:
                frame_labels[i] = debug_texts[j]

    # ── Figure setup ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    limit = max_gimbal_deg * 1.3
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel("Body Z deflection (\u00b0)")
    ax.set_ylabel("Body Y deflection (\u00b0)")
    ax.set_title("Gimbal Deflection \u2014 Looking Down the Nose")

    # Reference circle at max gimbal angle.
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(
        max_gimbal_deg * np.cos(theta),
        max_gimbal_deg * np.sin(theta),
        "--",
        color="gray",
        linewidth=1,
        alpha=0.6,
        label=f"Max gimbal ({max_gimbal_deg}\u00b0)",
    )
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.3)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # Dynamic elements.
    trail_len = max(1, int(2.0 / data.dt))  # ~2 seconds of trail
    (trail_line,) = ax.plot([], [], "-", color="steelblue", linewidth=1, alpha=0.4)
    (dot,) = ax.plot([], [], "o", color="orangered", markersize=8)

    time_text = fig.text(0.5, 0.95, "", ha="center", fontsize=11)
    stage_text = fig.text(
        0.5, 0.91, "", ha="center", fontsize=11, color="tab:green", fontweight="bold"
    )

    # ── Animation ────────────────────────────────────────────────────────
    def update(i):
        j0 = max(0, i - trail_len)
        trail_line.set_data(defl_z[j0 : i + 1], defl_y[j0 : i + 1])
        dot.set_data([defl_z[i]], [defl_y[i]])

        time_text.set_text(f"t = {data.times[i]:.2f} s")
        stage_text.set_text(frame_labels[i])

        return trail_line, dot, time_text, stage_text

    interval_ms = (data.dt / args.speed) * 1000
    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=interval_ms,
        repeat=True,
        blit=False,
    )

    if args.output:
        anim.save(args.output, writer="ffmpeg", fps=int(1 / data.dt))
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
