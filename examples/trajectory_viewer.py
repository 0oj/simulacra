"""
Live 3D trajectory viewer.

Reads NDJSON from one or more streams (files, pipes, stdin) and shows a
growing 3D trajectory in real time. Two views: a world overview and a
close-up tracking the rocket's current position.

Shares the streaming infrastructure with viewer.py (StreamReader, etc.).

Usage (from backend directory):

    # One-shot: read a file, show trajectory, keep window open
    uv run -m examples.trajectory_viewer ../temp/debug.ndjson

    # Pipe from a live simulation
    uv run ../scenarios/debug.py | uv run -m examples.trajectory_viewer

    # Compare two scenarios
    uv run -m examples.trajectory_viewer debug=../temp/debug.ndjson sounding=../temp/sounding.ndjson

Keyboard shortcuts:
    Space — pause / resume playback
    R     — reset view angles
    +/=   — speed up playback (2x, 4x, 8x)
    -     — slow down playback
"""

from __future__ import annotations

import argparse
import sys
import time as _time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection
from scipy.spatial.transform import Rotation

from .viewer import StreamReader, ScenarioEntry, parse_file_arg, TAB10


# Matplotlib 3D always renders Z as vertical. Simulation uses Y-up.
# Swap: world (X, Y, Z) → plot (X, Z, Y).
_YZ = [0, 2, 1]


class TrajectoryViewer:
    """Live 3D trajectory viewer polling StreamReaders via FuncAnimation."""

    def __init__(self, entries: list[ScenarioEntry]):
        self._entries = entries

        self._fig = plt.figure(figsize=(16, 7))
        self._fig.suptitle("Trajectory Viewer", fontsize=14, fontweight="bold")
        self._ax_world = self._fig.add_subplot(121, projection="3d")
        self._ax_local = self._fig.add_subplot(122, projection="3d")

        self._default_elev, self._default_azim = 20, -60
        self._ax_world.view_init(elev=self._default_elev, azim=self._default_azim)
        self._ax_local.view_init(elev=self._default_elev, azim=self._default_azim)

        self._fig.canvas.mpl_connect("key_press_event", self._on_key)

        # Playback state
        self._playback_active = False
        self._playback_paused = False
        self._playback_cursor = 0.0     # simulation-time cursor
        self._playback_speed = 1.0
        self._playback_max_time = 0.0
        self._last_tick_wall = _time.monotonic()

        # Status line
        labels = ", ".join(e.label for e in self._entries)
        self._status_text = self._fig.text(
            0.5, 0.01, f"Watching: {labels}", ha="center", fontsize=8, color="gray"
        )

        # Waiting text (centered in world view)
        self._waiting_text = self._ax_world.text2D(
            0.5, 0.5, "Waiting for data...",
            transform=self._ax_world.transAxes, ha="center", va="center",
            fontsize=14, color="lightgray",
        )

        self._keyhint_text = self._fig.text(
            0.5, 0.95, "Space = pause  |  +/- = speed  |  R = reset view",
            ha="center", fontsize=8, color="gray",
        )

        self._anim = FuncAnimation(
            self._fig, self._tick, interval=50, cache_frame_data=False,
        )

    def _on_key(self, event):
        if event.key == "r":
            self._ax_world.view_init(elev=self._default_elev, azim=self._default_azim)
            self._ax_local.view_init(elev=self._default_elev, azim=self._default_azim)
            self._fig.canvas.draw_idle()
        elif event.key == " ":
            self._playback_paused = not self._playback_paused
            self._last_tick_wall = _time.monotonic()  # avoid time jump on resume
        elif event.key in ("+", "="):
            self._playback_speed = min(self._playback_speed * 2, 16.0)
        elif event.key == "-":
            self._playback_speed = max(self._playback_speed / 2, 0.25)

    def _tick(self, frame):
        now = _time.monotonic()
        wall_dt = now - self._last_tick_wall
        self._last_tick_wall = now

        # Poll streams for new data
        changed = False
        all_eof = True
        for entry in self._entries:
            data, version, alive = entry.reader.snapshot()
            if alive:
                all_eof = False
            if version != entry.last_version:
                entry.last_version = version
                entry.data = data
                changed = True

        # Compute max time across all loaded scenarios
        max_t = 0.0
        has_data = False
        for entry in self._entries:
            if entry.data is not None and len(entry.data.times) > 0:
                has_data = True
                max_t = max(max_t, entry.data.times[-1])
        self._playback_max_time = max_t

        if changed and not self._playback_active:
            # Still streaming — show latest data as it arrives
            self._playback_cursor = max_t
            self._redraw()
            return

        # Enter playback mode once all streams are done and we have data
        if all_eof and has_data and not self._playback_active:
            self._playback_active = True
            self._playback_cursor = 0.0
            self._last_tick_wall = now  # reset so first step is clean

        if not self._playback_active:
            return

        # Advance cursor
        if not self._playback_paused:
            self._playback_cursor += wall_dt * self._playback_speed
            if self._playback_cursor > self._playback_max_time:
                self._playback_cursor = 0.0  # loop

        self._redraw()

    def _redraw(self):
        # Preserve current view angles across redraws
        w_elev, w_azim = self._ax_world.elev, self._ax_world.azim
        l_elev, l_azim = self._ax_local.elev, self._ax_local.azim

        self._ax_world.clear()
        self._ax_local.clear()

        has_any_data = False
        # Track the "focus" position for close-up (use first scenario with data)
        focus_pos = None
        focus_length = 6.0

        cursor = self._playback_cursor

        for entry in self._entries:
            if entry.data is None:
                continue
            d = entry.data
            color = entry.color
            n_total = len(d.times)

            # Slice data to playback cursor
            n = int(np.searchsorted(d.times, cursor, side="right"))
            n = max(1, min(n, n_total))

            has_any_data = True

            # Remap Y↔Z for matplotlib
            positions = d.positions[:n, _YZ]

            # Current position at cursor
            pos = positions[-1]

            # --- World view: trail up to cursor + current dot ---
            self._ax_world.plot(
                positions[:, 0], positions[:, 1], positions[:, 2],
                color=color, linewidth=1.5, alpha=0.8, label=entry.label,
            )
            self._ax_world.plot(
                [pos[0]], [pos[1]], [pos[2]],
                "o", color=color, markersize=5,
            )

            # --- Compute orientation for the current frame ---
            cur = n - 1  # index of current frame
            if len(d.quats_wxyz) > cur:
                quat_wxyz = d.quats_wxyz[cur]
                # Scipy expects (x, y, z, w)
                quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
                rot = Rotation.from_quat(quat_xyzw)
                # Body +X is the nose direction
                nose_dir = rot.apply([1.0, 0.0, 0.0])
                nose_dir = nose_dir[_YZ]
                nose = pos + d.rocket_length * nose_dir

                # Thrust in world frame (current frame)
                if len(d.thrust_body) > cur:
                    thrust_w = rot.apply(d.thrust_body[cur])
                    thrust_w = thrust_w[_YZ]
                    thrust_mag = np.linalg.norm(thrust_w)
                else:
                    thrust_w = np.zeros(3)
                    thrust_mag = 0.0

                # --- Close-up: rocket body + thrust arrow ---
                # Short trailing segment (last ~2 seconds)
                trail_len = max(1, int(2.0 / d.dt))
                j0 = max(0, n - trail_len)
                self._ax_local.plot(
                    positions[j0:, 0], positions[j0:, 1], positions[j0:, 2],
                    color=color, linewidth=1, alpha=0.4,
                )

                # Rocket body line
                self._ax_local.plot(
                    [pos[0], nose[0]], [pos[1], nose[1]], [pos[2], nose[2]],
                    color=color, linewidth=4, solid_capstyle="round",
                )

                # Thrust arrow (pointing opposite to thrust direction)
                if thrust_mag > 1.0:
                    scale = d.rocket_length / max(thrust_mag, 1.0)
                    tv = thrust_w * scale
                    self._ax_local.quiver(
                        pos[0], pos[1], pos[2],
                        -tv[0], -tv[1], -tv[2],
                        color=color, arrow_length_ratio=0.15, linewidth=2, alpha=0.7,
                    )

                # Also draw rocket body in world view
                self._ax_world.plot(
                    [pos[0], nose[0]], [pos[1], nose[1]], [pos[2], nose[2]],
                    color="black", linewidth=3, solid_capstyle="round",
                )

            # Track focus for close-up centering
            if focus_pos is None:
                focus_pos = pos
                focus_length = d.rocket_length

            # Stage annotations: only those up to current time
            for ts, text in d.debug_msgs:
                if ts > cursor:
                    break
                idx = np.searchsorted(d.times, ts)
                idx = min(idx, n - 1)
                ann_pos = d.positions[idx, _YZ]
                self._ax_world.text(
                    ann_pos[0], ann_pos[1], ann_pos[2],
                    f" {text}", fontsize=7, color=color, alpha=0.7,
                )

        # --- Axis styling ---
        self._ax_world.set_xlabel("X (m)")
        self._ax_world.set_ylabel("Z (m)")
        self._ax_world.set_zlabel("Y (m) — up")
        self._ax_world.set_title("World View")

        self._ax_local.set_xlabel("X (m)")
        self._ax_local.set_ylabel("Z (m)")
        self._ax_local.set_zlabel("Y (m) — up")
        self._ax_local.set_title("Close-up")

        if has_any_data:
            self._ax_world.legend(loc="upper left", fontsize=8)
            self._waiting_text.set_visible(False)

            # Auto-scale world view to FULL trajectory (not just cursor slice)
            # so the camera stays stable during playback
            all_positions = []
            for entry in self._entries:
                if entry.data is not None:
                    all_positions.append(entry.data.positions[:, _YZ])
            if all_positions:
                all_pos = np.vstack(all_positions)
                pad = focus_length * 2
                mins = all_pos.min(axis=0) - pad
                maxs = all_pos.max(axis=0) + pad
                ranges = maxs - mins
                max_range = ranges.max()
                centers = (mins + maxs) / 2
                self._ax_world.set_xlim(centers[0] - max_range / 2, centers[0] + max_range / 2)
                self._ax_world.set_ylim(centers[1] - max_range / 2, centers[1] + max_range / 2)
                self._ax_world.set_zlim(centers[2] - max_range / 2, centers[2] + max_range / 2)

            # Center close-up on the focus position
            if focus_pos is not None:
                r = focus_length * 3
                self._ax_local.set_xlim(focus_pos[0] - r, focus_pos[0] + r)
                self._ax_local.set_ylim(focus_pos[1] - r, focus_pos[1] + r)
                self._ax_local.set_zlim(focus_pos[2] - r, focus_pos[2] + r)
        else:
            self._waiting_text = self._ax_world.text2D(
                0.5, 0.5, "Waiting for data...",
                transform=self._ax_world.transAxes, ha="center", va="center",
                fontsize=14, color="lightgray",
            )

        # Restore view angles
        self._ax_world.view_init(elev=w_elev, azim=w_azim)
        self._ax_local.view_init(elev=l_elev, azim=l_azim)

        # Update status line
        labels = ", ".join(e.label for e in self._entries)
        if self._playback_active:
            paused = "PAUSED" if self._playback_paused else "playing"
            self._status_text.set_text(
                f"{labels}  |  {paused} {self._playback_speed:.2g}x"
                f"  |  t = {cursor:.2f} / {self._playback_max_time:.2f} s"
            )
        else:
            n_active = sum(1 for e in self._entries if e.reader.snapshot()[2])
            n_total = len(self._entries)
            now = datetime.now().strftime("%H:%M:%S")
            if n_active == 0:
                stream_status = "all streams ended"
            else:
                stream_status = f"{n_active}/{n_total} active"
            self._status_text.set_text(
                f"Watching: {labels}  |  Streams: {stream_status}  |  Last update: {now}"
            )

        self._fig.canvas.draw_idle()

    def show(self):
        plt.show()


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Live 3D trajectory viewer. Reads NDJSON from files or stdin.",
        epilog=(
            "examples:\n"
            "  uv run -m examples.trajectory_viewer output.ndjson\n"
            "  uv run -m examples.trajectory_viewer baseline=a.ndjson tweaked=b.ndjson\n"
            "  uv run ../scenarios/debug.py | uv run -m examples.trajectory_viewer\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "files",
        nargs="*",
        help=(
            "NDJSON sources to read. Use label=path for custom labels. "
            "If omitted, reads from stdin."
        ),
    )
    args = parser.parse_args()

    entries: list[ScenarioEntry] = []

    if args.files:
        for i, raw in enumerate(args.files):
            label, path = parse_file_arg(raw)
            try:
                stream = open(path, "r")
            except OSError as e:
                print(f"Error opening {path}: {e}", file=sys.stderr)
                continue
            color = TAB10[i % len(TAB10)]
            entries.append(ScenarioEntry(
                label=label,
                color=color,
                reader=StreamReader(stream),
            ))
    else:
        entries.append(ScenarioEntry(
            label="stdin",
            color=TAB10[0],
            reader=StreamReader(sys.stdin),
        ))

    if not entries:
        print("No valid sources to read.", file=sys.stderr)
        sys.exit(1)

    viewer = TrajectoryViewer(entries)
    viewer.show()


if __name__ == "__main__":
    main()
