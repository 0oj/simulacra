"""
Stacked time-series telemetry plot.

Base panels (always shown):
  1. Velocity (world frame) — magnitude + components
  2. Acceleration (derived via finite difference) — magnitude + components
  3. Thrust magnitude
  4. Angular velocity (body frame) — components

Mass model panels (shown when mass data is present and varies):
  5. Rocket mass (kg)
  6. Fuel fraction (0..1)

Stage transitions from debug messages are shown as vertical dashed lines.

Usage (from backend directory):

    uv run ../scenarios/debug.py > output.ndjson
    uv run -m examples.plot_telemetry output.ndjson
    uv run -m examples.plot_telemetry output.ndjson --output telemetry.png
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt

from .ndjson_reader import load_flight


def main():
    parser = argparse.ArgumentParser(
        description="Stacked time-series telemetry plotter"
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        type=str,
        help="Input NDJSON file (if omitted, reads from stdin)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Save plot to file (e.g. telemetry.png)",
    )
    args = parser.parse_args()

    # ── Read data ────────────────────────────────────────────────────────
    data = load_flight(args.input_file)
    n = len(data.times)
    if n == 0:
        print("No trajectory data found.", file=sys.stderr)
        return

    t = data.times

    # ── Derived quantities ───────────────────────────────────────────────
    vel_mag = np.linalg.norm(data.velocities, axis=1)

    # Acceleration via finite difference of velocity w.r.t. time.
    if n > 1:
        accel = np.gradient(data.velocities, t, axis=0)
    else:
        accel = np.zeros_like(data.velocities)
    accel_mag = np.linalg.norm(accel, axis=1)

    thrust_mag = np.linalg.norm(data.thrust_body, axis=1)

    # Check if mass model data is present and varies over time.
    has_mass = (
        len(data.masses) == n
        and data.masses.max() - data.masses.min() > 1e-6
    )

    # ── Figure layout ────────────────────────────────────────────────────
    n_panels = 6 if has_mass else 4
    fig_height = 2.5 * n_panels
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, fig_height), sharex=True)
    fig.suptitle("Flight Telemetry", fontsize=14, fontweight="bold")

    comp_colors = {"x": "tab:red", "y": "tab:green", "z": "tab:blue"}

    # Panel 1: Velocity
    ax = axes[0]
    ax.plot(t, vel_mag, "k-", linewidth=1.5, label="|V|")
    ax.plot(
        t, data.velocities[:, 0],
        color=comp_colors["x"], linewidth=0.8, alpha=0.7, label="Vx",
    )
    ax.plot(
        t, data.velocities[:, 1],
        color=comp_colors["y"], linewidth=0.8, alpha=0.7, label="Vy",
    )
    ax.plot(
        t, data.velocities[:, 2],
        color=comp_colors["z"], linewidth=0.8, alpha=0.7, label="Vz",
    )
    ax.set_ylabel("Velocity (m/s)")
    ax.legend(loc="upper right", fontsize=8, ncol=4)
    ax.grid(True, alpha=0.3)

    # Panel 2: Acceleration
    ax = axes[1]
    ax.plot(t, accel_mag, "k-", linewidth=1.5, label="|a|")
    ax.plot(
        t, accel[:, 0],
        color=comp_colors["x"], linewidth=0.8, alpha=0.7, label="ax",
    )
    ax.plot(
        t, accel[:, 1],
        color=comp_colors["y"], linewidth=0.8, alpha=0.7, label="ay",
    )
    ax.plot(
        t, accel[:, 2],
        color=comp_colors["z"], linewidth=0.8, alpha=0.7, label="az",
    )
    ax.set_ylabel("Acceleration (m/s\u00b2)")
    ax.legend(loc="upper right", fontsize=8, ncol=4)
    ax.grid(True, alpha=0.3)

    # Panel 3: Thrust magnitude
    ax = axes[2]
    ax.plot(t, thrust_mag, "r-", linewidth=1.5, label="Thrust")
    ax.set_ylabel("Thrust (N)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Angular velocity
    ax = axes[3]
    ax.plot(
        t, data.angular_vel_body[:, 0],
        color=comp_colors["x"], linewidth=1, label="\u03c9x (roll)",
    )
    ax.plot(
        t, data.angular_vel_body[:, 1],
        color=comp_colors["y"], linewidth=1, label="\u03c9y (pitch)",
    )
    ax.plot(
        t, data.angular_vel_body[:, 2],
        color=comp_colors["z"], linewidth=1, label="\u03c9z (yaw)",
    )
    ax.set_ylabel("Angular vel (rad/s)")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)

    # Panels 5-6: Mass model (conditional)
    if has_mass:
        ax = axes[4]
        ax.plot(t, data.masses, "k-", linewidth=1.5, label="Mass")
        ax.set_ylabel("Mass (kg)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[5]
        ax.plot(t, data.fuel_fractions * 100, "tab:orange", linewidth=1.5, label="Fuel")
        ax.set_ylabel("Fuel (%)")
        ax.set_ylim(-5, 105)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    # X-axis label on the bottom panel.
    axes[-1].set_xlabel("Time (s)")

    # ── Stage annotations ────────────────────────────────────────────────
    for ts, text in data.debug_msgs:
        for ax in axes:
            ax.axvline(ts, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        # Label at top of the first panel using axes-relative y coordinate.
        axes[0].text(
            ts,
            1.0,
            f" {text}",
            transform=axes[0].get_xaxis_transform(),
            fontsize=7,
            color="gray",
            va="top",
            ha="left",
        )

    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150)
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
