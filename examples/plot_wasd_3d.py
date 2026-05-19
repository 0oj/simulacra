"""
Real-time 3D trajectory animator for rocket simulations.

Reads NDJSON messages from a file (or stdin), then plays back the flight
as a looping 3D animation with two views:
- Left:  world view — full trajectory, trail, and rocket dot
- Right: close-up — rocket body, thrust vector, orientation detail

Keyboard shortcuts:
    R  — reset both view orientations

Usage (from backend directory):

    uv run ../scenarios/debug.py > output.ndjson
    uv run -m examples.plot_wasd_3d output.ndjson

    # Adjust playback speed:
    uv run -m examples.plot_wasd_3d output.ndjson --speed 2.0

    # Save as MP4 instead of showing:
    uv run -m examples.plot_wasd_3d output.ndjson --output flight.mp4
"""

from __future__ import annotations

import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection
from scipy.spatial.transform import Rotation

from .ndjson_reader import load_flight


def main():
    parser = argparse.ArgumentParser(
        description="Real-time 3D trajectory animator for rocket simulations"
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        type=str,
        help="Input NDJSON file (if omitted, reads from stdin)",
    )
    parser.add_argument(
        "--output", "-o", type=str, help="Save animation to file (e.g. flight.mp4)"
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

    n_frames = len(data.positions)
    if n_frames == 0:
        print("No trajectory data found.", file=sys.stderr)
        return

    dt = data.dt
    rocket_length = data.rocket_length
    positions = data.positions
    quats_wxyz = data.quats_wxyz
    thrust_body = data.thrust_body

    # ── Precompute rotations ─────────────────────────────────────────────
    # Scipy expects (x, y, z, w); schema stores (w, x, y, z).
    quats_xyzw = quats_wxyz[:, [1, 2, 3, 0]]
    rotations = Rotation.from_quat(quats_xyzw)

    # Body +X is the nose direction.
    body_x = np.array([1.0, 0.0, 0.0])
    nose_dirs = rotations.apply(body_x)  # (N, 3) world-frame nose unit vectors

    # Thrust in world frame for each timestep.
    thrust_world = rotations.apply(thrust_body)  # (N, 3)
    thrust_mags = np.linalg.norm(thrust_world, axis=1)
    max_thrust = thrust_mags.max() if thrust_mags.max() > 0 else 1.0

    # Scale thrust arrows so the largest is rocket_length long.
    thrust_scale = rocket_length / max_thrust

    # ── Precompute per-frame stage labels from debug messages ──────────
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

    # ── Remap to plot coords ─────────────────────────────────────────────
    # Matplotlib always renders Z as vertical.  The simulation uses Y-up,
    # so swap Y↔Z: world (X, Y, Z) → plot (X, Z, Y).
    #   plot X = world X,  plot Y = world Z,  plot Z = world Y (up)
    _YZ = [0, 2, 1]
    positions = positions[:, _YZ]
    nose_dirs = nose_dirs[:, _YZ]
    thrust_world = thrust_world[:, _YZ]

    # ── Figure setup: two side-by-side 3D axes ───────────────────────────
    fig = plt.figure(figsize=(16, 7))
    ax_world = fig.add_subplot(121, projection="3d")
    ax_local = fig.add_subplot(122, projection="3d")

    # ── World-view axis limits ───────────────────────────────────────────
    pad = rocket_length * 2
    world_mins = positions.min(axis=0) - pad
    world_maxs = positions.max(axis=0) + pad
    ranges = world_maxs - world_mins
    max_range = ranges.max()
    centers = (world_mins + world_maxs) / 2
    world_mins = centers - max_range / 2
    world_maxs = centers + max_range / 2

    ax_world.set_xlim(world_mins[0], world_maxs[0])
    ax_world.set_ylim(world_mins[1], world_maxs[1])
    ax_world.set_zlim(world_mins[2], world_maxs[2])
    ax_world.set_xlabel("X (m)")
    ax_world.set_ylabel("Z (m)")
    ax_world.set_zlabel("Y (m) — up")
    ax_world.set_title("World View")

    ax_local.set_xlabel("X (m)")
    ax_local.set_ylabel("Z (m)")
    ax_local.set_zlabel("Y (m) — up")
    ax_local.set_title("Close-up")

    # Default view angles.
    default_elev, default_azim = 20, -60
    ax_world.view_init(elev=default_elev, azim=default_azim)
    ax_local.view_init(elev=default_elev, azim=default_azim)

    lock_radius = rocket_length * 3

    def on_key(event):
        if event.key == "r":
            ax_world.view_init(elev=default_elev, azim=default_azim)
            ax_local.view_init(elev=default_elev, azim=default_azim)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)

    # ── World-view static elements ───────────────────────────────────────
    ax_world.plot(
        positions[:, 0], positions[:, 1], positions[:, 2],
        color="steelblue", alpha=0.2, linewidth=1,
    )

    # ── World-view dynamic elements ──────────────────────────────────────
    (w_trail,) = ax_world.plot([], [], [], color="steelblue", linewidth=2)
    (w_dot,) = ax_world.plot([], [], [], "o", color="black", markersize=4)

    # ── Close-up dynamic elements ────────────────────────────────────────
    (l_trail,) = ax_local.plot([], [], [], color="steelblue", linewidth=1, alpha=0.4)
    (l_rocket,) = ax_local.plot(
        [], [], [], color="black", linewidth=4, solid_capstyle="round",
    )
    l_thrust = ax_local.quiver(
        0, 0, 0, 0, 0, 0, color="orangered", arrow_length_ratio=0.15, linewidth=2,
    )

    # ── Shared text overlays (on figure, not on axes) ────────────────────
    time_text = fig.text(0.5, 0.95, "", ha="center", fontsize=11)
    stage_text = fig.text(0.5, 0.91, "", ha="center", fontsize=11,
                          color="tab:green", fontweight="bold")
    fig.text(0.5, 0.02, "R = reset view angles", ha="center",
             fontsize=8, color="gray")

    # ── Animation ────────────────────────────────────────────────────────
    def update(i):
        nonlocal l_thrust

        pos = positions[i]
        nose = pos + rocket_length * nose_dirs[i]

        # ── World view: trail + dot ──────────────────────────────────
        w_trail.set_data_3d(
            positions[:i + 1, 0], positions[:i + 1, 1], positions[:i + 1, 2],
        )
        w_dot.set_data_3d([pos[0]], [pos[1]], [pos[2]])

        # ── Close-up: rocket body + thrust arrow ─────────────────────
        # Show a short trailing segment for context (last ~2 seconds).
        trail_len = max(1, int(2.0 / dt))
        j0 = max(0, i - trail_len)
        l_trail.set_data_3d(
            positions[j0:i + 1, 0], positions[j0:i + 1, 1], positions[j0:i + 1, 2],
        )

        l_rocket.set_data_3d(
            [pos[0], nose[0]], [pos[1], nose[1]], [pos[2], nose[2]],
        )

        l_thrust.remove()
        tv = thrust_world[i] * thrust_scale
        l_thrust = ax_local.quiver(
            pos[0], pos[1], pos[2],
            -tv[0], -tv[1], -tv[2],
            color="orangered", arrow_length_ratio=0.15, linewidth=2,
        )

        # Re-center close-up axes on the rocket every frame.
        r = lock_radius
        ax_local.set_xlim(pos[0] - r, pos[0] + r)
        ax_local.set_ylim(pos[1] - r, pos[1] + r)
        ax_local.set_zlim(pos[2] - r, pos[2] + r)

        # Text overlays.
        time_text.set_text(f"t = {data.times[i]:.2f} s")
        stage_text.set_text(frame_labels[i])

        return w_trail, w_dot, l_trail, l_rocket, l_thrust, time_text

    interval_ms = (dt / args.speed) * 1000
    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=interval_ms,
        repeat=True,
        blit=False,
    )

    if args.output:
        anim.save(args.output, writer="ffmpeg", fps=int(1 / dt))
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
