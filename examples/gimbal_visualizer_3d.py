"""
Interactive 3D visualizer for the nested gimbal actuator.

This lets you adjust the outer and inner gimbal angles and see the
resulting nozzle/thrust direction in the rocket *body* frame.

Usage (from backend directory):

    uv run -m examples.gimbal_visualizer_3d

Close the window to exit.
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

# Add the src directory to the path so we can import simulacra
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(script_dir)
src_dir = os.path.join(backend_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from simulacra.phenomena.actuators import _nozzle_direction_body  # noqa: E402


def compute_dir_body(outer_deg: float, inner_deg: float) -> np.ndarray:
    """
    Convenience wrapper around `_nozzle_direction_body` using degrees.

    Args:
        outer_deg: Outer gimbal yaw about body +Z (degrees).
        inner_deg: Inner gimbal rotation about its +Y axis (degrees).
    """
    outer = np.deg2rad(outer_deg)
    inner = np.deg2rad(inner_deg)
    return _nozzle_direction_body(outer, inner)


def _set_equal_3d(ax) -> None:
    """Set equal aspect for a 3D axis."""
    max_range = 1.0
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    ax.set_box_aspect((1, 1, 1))


def create_plot() -> Tuple[plt.Figure, plt.Axes, Slider, Slider]:
    # Initial angles in degrees
    outer_init = 0.0
    inner_init = 0.0

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Title and axis labels
    ax.set_title("Nested Gimbal Visualizer (Body Frame)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Body X (nose)", color="r")
    ax.set_ylabel("Body Y", color="g")
    ax.set_zlabel("Body Z", color="b")

    # Draw body axes
    ax.quiver(0, 0, 0, 1, 0, 0, color="r", linewidth=2, arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 1, 0, color="g", linewidth=2, arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, 1, color="b", linewidth=2, arrow_length_ratio=0.1)

    # Initial nozzle direction
    dir0 = compute_dir_body(outer_init, inner_init)
    thrust_quiver = ax.quiver(
        0,
        0,
        0,
        dir0[0],
        dir0[1],
        dir0[2],
        color="magenta",
        linewidth=3,
        arrow_length_ratio=0.15,
    )

    # View: look approximately down +X so Y is horizontal and Z is vertical.
    # Matplotlib's 3D view is specified by elev (about X-Y plane) and azim (about Z).
    # This combination gives a "front-on" ball-style view where the nose (+X)
    # is roughly pointing out of the screen.
    ax.view_init(elev=0, azim=0)
    _set_equal_3d(ax)

    # HUD text (angles + direction components)
    hud = fig.text(
        0.02,
        0.95,
        "",
        fontsize=10,
        family="monospace",
        va="top",
    )

    def update_hud(outer_deg: float, inner_deg: float, d: np.ndarray) -> None:
        hud.set_text(
            "HUD (body frame)\n"
            f"  outer_yaw_z  = {outer_deg:6.2f}°\n"
            f"  inner_pitch_y= {inner_deg:6.2f}°\n"
            f"  dir_body     = [{d[0]:+6.3f}, {d[1]:+6.3f}, {d[2]:+6.3f}]"
        )

    update_hud(outer_init, inner_init, dir0)

    # Sliders: place them below the plot
    slider_ax_outer = fig.add_axes([0.15, 0.08, 0.7, 0.03])
    slider_ax_inner = fig.add_axes([0.15, 0.03, 0.7, 0.03])

    outer_slider = Slider(
        slider_ax_outer,
        "Outer yaw (deg, +Z)",
        valmin=-60.0,
        valmax=60.0,
        valinit=outer_init,
    )
    inner_slider = Slider(
        slider_ax_inner,
        "Inner pitch (deg, +Y in outer frame)",
        valmin=-45.0,
        valmax=45.0,
        valinit=inner_init,
    )

    # Update callback
    def on_change(_val: float) -> None:
        nonlocal thrust_quiver

        outer_deg = outer_slider.val
        inner_deg = inner_slider.val
        d = compute_dir_body(outer_deg, inner_deg)

        # Remove old arrow and draw new one
        thrust_quiver.remove()
        thrust_quiver = ax.quiver(
            0,
            0,
            0,
            d[0],
            d[1],
            d[2],
            color="magenta",
            linewidth=3,
            arrow_length_ratio=0.15,
        )

        update_hud(outer_deg, inner_deg, d)
        fig.canvas.draw_idle()

    outer_slider.on_changed(on_change)
    inner_slider.on_changed(on_change)

    return fig, ax, outer_slider, inner_slider


def main(output: str | None = None) -> None:
    fig, _ax, _outer_slider, _inner_slider = create_plot()
    if output:
        # Save a static snapshot at the initial angles.
        fig.savefig(output, dpi=150)
        print(f"Saved gimbal visualisation to {output}")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse  # noqa: E402

    parser = argparse.ArgumentParser(
        description="Interactive nested gimbal visualizer (3D body frame)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Optional output file (PNG, PDF, etc.) for a static snapshot",
    )
    args = parser.parse_args()

    main(args.output)



