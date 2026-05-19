"""
Persistent stream-based flight viewer with scenario comparison.

Reads NDJSON from one or more streams (files, pipes, stdin), plots simple
scalar panels for comparison, and updates in place as new data arrives.
The viewer is a pure stream consumer — the user manages the streams.

Usage (from backend directory):

    # One-shot: read files, plot, keep window open
    uv run -m examples.viewer temp/a.ndjson temp/b.ndjson

    # Pipe from a live simulation
    uv run ../scenarios/debug.py | uv run -m examples.viewer

    # Custom labels
    uv run -m examples.viewer debug=temp/debug.ndjson sounding=temp/sounding.ndjson

    # Named pipes for re-runnable streams
    mkfifo /tmp/sim_a /tmp/sim_b
    uv run -m examples.viewer baseline=/tmp/sim_a tweaked=/tmp/sim_b
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import IO

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation

# Ensure simulacra is importable even when not installed as a package.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.dirname(_script_dir)
_src_dir = os.path.join(_backend_dir, "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from simulacra.schema import (  # noqa: E402
    MetadataMessage,
    ConstantMessage,
    VariableMessage,
    DebugMessage,
    MessageType,
)

from .ndjson_reader import FlightData  # noqa: E402


# ============================================================================
# Line parser
# ============================================================================


def _parse_line(line: str):
    """Parse a single NDJSON line into a schema message, or None on failure."""
    try:
        data = json.loads(line.strip())
    except (json.JSONDecodeError, ValueError):
        return None
    msg_type = data.get("message_type")
    try:
        if msg_type == MessageType.METADATA:
            return MetadataMessage(**data)
        elif msg_type == MessageType.CONSTANT:
            return ConstantMessage(**data)
        elif msg_type == MessageType.VARIABLE:
            return VariableMessage(**data)
        elif msg_type == MessageType.DEBUG:
            return DebugMessage(**data)
    except Exception:
        return None
    return None


# ============================================================================
# StreamReader
# ============================================================================


class StreamReader:
    """Background thread that incrementally reads NDJSON and accumulates FlightData."""

    def __init__(self, stream: IO):
        self._stream = stream
        self._lock = threading.Lock()
        self._version = 0
        self._alive = True

        # Accumulated state
        self._dt = 0.02
        self._rocket_length = 6.0
        self._times: list[float] = []
        self._positions: list[list[float]] = []
        self._velocities: list[list[float]] = []
        self._quats_wxyz: list[list[float]] = []
        self._thrust_body: list[list[float]] = []
        self._angular_vel_body: list[list[float]] = []
        self._masses: list[float] = []
        self._fuel_fractions: list[float] = []
        self._debug_msgs: list[tuple[float, str]] = []

        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _reset(self):
        """Clear accumulated data for a new run (called under lock)."""
        self._times.clear()
        self._positions.clear()
        self._velocities.clear()
        self._quats_wxyz.clear()
        self._thrust_body.clear()
        self._angular_vel_body.clear()
        self._masses.clear()
        self._fuel_fractions.clear()
        self._debug_msgs.clear()

    def _read_loop(self):
        """Target for the daemon thread. Reads until EOF."""
        try:
            while True:
                line = self._stream.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                msg = _parse_line(line)
                if msg is None:
                    continue
                with self._lock:
                    if isinstance(msg, MetadataMessage):
                        self._reset()
                        if msg.dt is not None:
                            self._dt = msg.dt
                    elif isinstance(msg, ConstantMessage):
                        if msg.rocket_length is not None:
                            self._rocket_length = msg.rocket_length
                    elif isinstance(msg, DebugMessage):
                        self._debug_msgs.append((msg.timestamp, msg.text))
                    elif isinstance(msg, VariableMessage):
                        if msg.position is None or msg.orientation_quat is None:
                            continue
                        self._times.append(msg.timestamp)
                        p = msg.position
                        self._positions.append([p.x, p.y, p.z])
                        if msg.velocity is not None:
                            v = msg.velocity
                            self._velocities.append([v.x, v.y, v.z])
                        else:
                            self._velocities.append([0.0, 0.0, 0.0])
                        self._quats_wxyz.append(list(msg.orientation_quat))
                        if msg.thrust_vector_body is not None:
                            t = msg.thrust_vector_body
                            self._thrust_body.append([t.x, t.y, t.z])
                        else:
                            self._thrust_body.append([0.0, 0.0, 0.0])
                        if msg.angular_velocity_body is not None:
                            w = msg.angular_velocity_body
                            self._angular_vel_body.append([w.x, w.y, w.z])
                        else:
                            self._angular_vel_body.append([0.0, 0.0, 0.0])
                        if msg.rocket_mass is not None:
                            self._masses.append(msg.rocket_mass)
                        if msg.fuel_fraction is not None:
                            self._fuel_fractions.append(msg.fuel_fraction)
                    self._version += 1
        finally:
            with self._lock:
                self._alive = False

    def snapshot(self) -> tuple[FlightData | None, int, bool]:
        """Thread-safe: return (data, version, alive)."""
        with self._lock:
            if not self._times:
                return None, self._version, self._alive
            n = len(self._times)
            data = FlightData(
                dt=self._dt,
                rocket_length=self._rocket_length,
                times=np.array(self._times),
                positions=np.array(self._positions),
                velocities=np.array(self._velocities) if self._velocities else np.empty((0, 3)),
                quats_wxyz=np.array(self._quats_wxyz) if self._quats_wxyz else np.empty((0, 4)),
                thrust_body=np.array(self._thrust_body) if self._thrust_body else np.empty((0, 3)),
                angular_vel_body=np.array(self._angular_vel_body) if self._angular_vel_body else np.empty((0, 3)),
                masses=np.array(self._masses) if len(self._masses) == n else np.empty((0,)),
                fuel_fractions=np.array(self._fuel_fractions) if len(self._fuel_fractions) == n else np.empty((0,)),
                debug_msgs=list(self._debug_msgs),
            )
            return data, self._version, self._alive


# ============================================================================
# ScenarioEntry
# ============================================================================


@dataclass
class ScenarioEntry:
    label: str
    color: str
    reader: StreamReader
    last_version: int = 0
    data: FlightData | None = None


# ============================================================================
# Viewer
# ============================================================================


class Viewer:
    """Owns the matplotlib figure and polls StreamReaders via FuncAnimation."""

    PANEL_LABELS = [
        "Altitude (m)", "Speed (m/s)", "Thrust (N)", "Mass (kg)",
        "Tilt from vertical (deg)", "Angular vel (rad/s)",
    ]

    def __init__(self, entries: list[ScenarioEntry]):
        self._entries = entries

        self._fig, self._axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
        self._fig.suptitle("Flight Viewer", fontsize=14, fontweight="bold")

        for ax, label in zip(self._axes, self.PANEL_LABELS):
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
        self._axes[-1].set_xlabel("Time (s)")

        # Status line at the bottom of the figure
        labels = ", ".join(e.label for e in self._entries)
        self._status_text = self._fig.text(
            0.5, 0.01, f"Watching: {labels}", ha="center", fontsize=8, color="gray"
        )

        # "Waiting for data" text objects, one per panel
        self._waiting_texts = []
        for ax in self._axes:
            txt = ax.text(
                0.5, 0.5, "Waiting for data...",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color="lightgray",
            )
            self._waiting_texts.append(txt)

        self._fig.tight_layout(rect=[0, 0.03, 1, 0.96])

        self._anim = FuncAnimation(
            self._fig, self._tick, interval=500, cache_frame_data=False,
        )

    def _tick(self, frame):
        changed = False
        for entry in self._entries:
            data, version, alive = entry.reader.snapshot()
            if version != entry.last_version:
                entry.last_version = version
                entry.data = data
                changed = True
        if changed:
            self._redraw()

    def _redraw(self):
        for ax in self._axes:
            ax.clear()

        has_any_data = False
        for entry in self._entries:
            if entry.data is None:
                continue
            has_any_data = True
            d = entry.data
            t = d.times
            color = entry.color

            # Panel 0: Altitude (Y component = up)
            altitude = d.positions[:, 1]
            self._axes[0].plot(t, altitude, color=color, linewidth=1.5, label=entry.label)

            # Panel 1: Speed (velocity magnitude)
            if len(d.velocities) == len(t):
                speed = np.linalg.norm(d.velocities, axis=1)
            else:
                speed = np.zeros(len(t))
            self._axes[1].plot(t, speed, color=color, linewidth=1.5, label=entry.label)

            # Panel 2: Thrust (body-frame magnitude)
            if len(d.thrust_body) == len(t):
                thrust_mag = np.linalg.norm(d.thrust_body, axis=1)
            else:
                thrust_mag = np.zeros(len(t))
            self._axes[2].plot(t, thrust_mag, color=color, linewidth=1.5, label=entry.label)

            # Panel 3: Mass
            if len(d.masses) == len(t):
                self._axes[3].plot(t, d.masses, color=color, linewidth=1.5, label=entry.label)
            else:
                self._axes[3].text(
                    0.5, 0.5, "N/A",
                    transform=self._axes[3].transAxes, ha="center", va="center",
                    fontsize=11, color="lightgray",
                )

            # Panel 4: Tilt from vertical (degrees)
            # Nose is body +X. Transform to world, measure angle from +Y (up).
            if len(d.quats_wxyz) == len(t):
                quats_xyzw = d.quats_wxyz[:, [1, 2, 3, 0]]
                rots = Rotation.from_quat(quats_xyzw)
                nose_world = rots.apply([1.0, 0.0, 0.0])  # (N, 3)
                # cos(tilt) = nose_world · [0, 1, 0] = nose_world[:, 1]
                cos_tilt = np.clip(nose_world[:, 1], -1.0, 1.0)
                tilt_deg = np.degrees(np.arccos(cos_tilt))
                self._axes[4].plot(t, tilt_deg, color=color, linewidth=1.5, label=entry.label)

            # Panel 5: Angular velocity components (body-frame)
            if len(d.angular_vel_body) == len(t):
                labels_omega = ["ωx", "ωy", "ωz"]
                # Use the scenario color but vary lightness for the 3 components
                for k, (comp_label, ls) in enumerate(zip(labels_omega, ["-", "--", ":"])):
                    lbl = f"{entry.label} {comp_label}" if len(self._entries) > 1 else comp_label
                    self._axes[5].plot(
                        t, d.angular_vel_body[:, k],
                        color=color, linestyle=ls, linewidth=1.2, label=lbl,
                    )

            # Stage annotations on altitude panel
            self._draw_annotations(entry)

        # Restore axis labels and grid after clear
        for ax, label in zip(self._axes, self.PANEL_LABELS):
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
        self._axes[-1].set_xlabel("Time (s)")

        if has_any_data:
            self._axes[0].legend(loc="upper right", fontsize=8)
            self._axes[5].legend(loc="upper right", fontsize=7, ncol=3)
            # Remove waiting text
            for txt in self._waiting_texts:
                txt.set_visible(False)
        else:
            for ax, txt in zip(self._axes, self._waiting_texts):
                # Re-add waiting text after clear
                new_txt = ax.text(
                    0.5, 0.5, "Waiting for data...",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=11, color="lightgray",
                )

        # Update status line
        n_active = sum(1 for e in self._entries if e.reader.snapshot()[2])
        n_total = len(self._entries)
        now = datetime.now().strftime("%H:%M:%S")
        if n_active == 0:
            stream_status = "all streams ended"
        else:
            stream_status = f"{n_active}/{n_total} active"
        labels = ", ".join(e.label for e in self._entries)
        self._status_text.set_text(
            f"Watching: {labels}  |  Streams: {stream_status}  |  Last update: {now}"
        )

        self._fig.canvas.draw_idle()

    def _draw_annotations(self, entry: ScenarioEntry):
        """Draw stage-transition vertical lines on the altitude panel."""
        if entry.data is None:
            return
        idx = self._entries.index(entry)
        ax = self._axes[0]
        for ts, text in entry.data.debug_msgs:
            ax.axvline(ts, color=entry.color, linestyle="--", linewidth=0.8, alpha=0.3)
            # Stagger label vertically by scenario index
            y_offset = 0.95 - 0.06 * idx
            ax.text(
                ts, y_offset, f" {entry.label}: {text}",
                transform=ax.get_xaxis_transform(),
                fontsize=7, color=entry.color, alpha=0.7,
                va="top", ha="left",
            )

    def show(self):
        plt.show()


# ============================================================================
# CLI helpers
# ============================================================================


def parse_file_arg(arg: str) -> tuple[str, str]:
    """Parse 'label=path' or plain 'path'. Returns (label, path)."""
    if "=" in arg:
        label, path = arg.split("=", 1)
        return label, path
    return Path(arg).stem, arg


TAB10 = plt.cm.tab10.colors  # type: ignore[attr-defined]


def main():
    parser = argparse.ArgumentParser(
        description="Persistent flight data viewer. Reads NDJSON from files or stdin.",
        epilog=(
            "examples:\n"
            "  uv run -m examples.viewer output.ndjson\n"
            "  uv run -m examples.viewer baseline=a.ndjson tweaked=b.ndjson\n"
            "  uv run ../scenarios/debug.py | uv run -m examples.viewer\n"
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
        # Read from stdin
        entries.append(ScenarioEntry(
            label="stdin",
            color=TAB10[0],
            reader=StreamReader(sys.stdin),
        ))

    if not entries:
        print("No valid sources to read.", file=sys.stderr)
        sys.exit(1)

    viewer = Viewer(entries)
    viewer.show()


if __name__ == "__main__":
    main()
