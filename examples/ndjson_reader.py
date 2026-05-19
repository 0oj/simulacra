"""
Shared NDJSON flight data loader.

Every plotter imports ``load_flight(path)`` and gets back a FlightData
dataclass with all the arrays ready to use.

Usage::

    from .ndjson_reader import load_flight

    data = load_flight("output.ndjson")   # or None for stdin
    print(data.positions.shape)           # (N, 3)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

# Ensure simulacra is importable even when not installed as a package.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.dirname(_script_dir)
_src_dir = os.path.join(_backend_dir, "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from simulacra.schema import (  # noqa: E402
    read_message,
    MetadataMessage,
    ConstantMessage,
    VariableMessage,
    DebugMessage,
    SCHEMA_VERSION,
)


@dataclass
class FlightData:
    dt: float
    rocket_length: float
    times: np.ndarray  # (N,) — timestamp per frame
    positions: np.ndarray  # (N,3) world XYZ
    velocities: np.ndarray  # (N,3) world XYZ
    quats_wxyz: np.ndarray  # (N,4)
    thrust_body: np.ndarray  # (N,3) rocket frame
    angular_vel_body: np.ndarray  # (N,3) rocket frame
    masses: np.ndarray  # (N,) kg per frame (empty if not logged)
    fuel_fractions: np.ndarray  # (N,) 0..1 per frame (empty if not logged)
    debug_msgs: list[tuple[float, str]]


def load_flight(input_file: str | None) -> FlightData:
    """
    Read all NDJSON messages from a file (or stdin) and return FlightData.

    Skips VariableMessage frames missing position or orientation_quat.
    """
    dt = 0.02  # fallback
    rocket_length = 6.0  # fallback

    times: list[float] = []
    positions: list[list[float]] = []
    velocities: list[list[float]] = []
    quats_wxyz: list[list[float]] = []
    thrust_body: list[list[float]] = []
    angular_vel_body: list[list[float]] = []
    masses: list[float] = []
    fuel_fractions: list[float] = []
    debug_msgs: list[tuple[float, str]] = []

    stream = open(input_file, "r") if input_file else sys.stdin
    try:
        while True:
            msg = read_message(stream)

            if isinstance(msg, MetadataMessage):
                if msg.schema_version != SCHEMA_VERSION:
                    print(
                        f"Warning: schema version mismatch "
                        f"(expected {SCHEMA_VERSION}, got {msg.schema_version})",
                        file=sys.stderr,
                    )
                if msg.dt is not None:
                    dt = msg.dt

            elif isinstance(msg, ConstantMessage):
                if msg.rocket_length is not None:
                    rocket_length = msg.rocket_length

            elif isinstance(msg, DebugMessage):
                debug_msgs.append((msg.timestamp, msg.text))

            elif isinstance(msg, VariableMessage):
                if msg.position is None or msg.orientation_quat is None:
                    continue

                times.append(msg.timestamp)

                p = msg.position
                positions.append([p.x, p.y, p.z])

                if msg.velocity is not None:
                    v = msg.velocity
                    velocities.append([v.x, v.y, v.z])
                else:
                    velocities.append([0.0, 0.0, 0.0])

                q = msg.orientation_quat  # (w, x, y, z)
                quats_wxyz.append(list(q))

                if msg.thrust_vector_body is not None:
                    t = msg.thrust_vector_body
                    thrust_body.append([t.x, t.y, t.z])
                else:
                    thrust_body.append([0.0, 0.0, 0.0])

                if msg.angular_velocity_body is not None:
                    w = msg.angular_velocity_body
                    angular_vel_body.append([w.x, w.y, w.z])
                else:
                    angular_vel_body.append([0.0, 0.0, 0.0])

                if msg.rocket_mass is not None:
                    masses.append(msg.rocket_mass)
                if msg.fuel_fraction is not None:
                    fuel_fractions.append(msg.fuel_fraction)

    except EOFError:
        pass
    finally:
        if input_file and stream:
            stream.close()

    debug_msgs.sort(key=lambda x: x[0])

    return FlightData(
        dt=dt,
        rocket_length=rocket_length,
        times=np.array(times),
        positions=np.array(positions) if positions else np.empty((0, 3)),
        velocities=np.array(velocities) if velocities else np.empty((0, 3)),
        quats_wxyz=np.array(quats_wxyz) if quats_wxyz else np.empty((0, 4)),
        thrust_body=np.array(thrust_body) if thrust_body else np.empty((0, 3)),
        angular_vel_body=(
            np.array(angular_vel_body) if angular_vel_body else np.empty((0, 3))
        ),
        masses=np.array(masses) if masses else np.empty((0,)),
        fuel_fractions=np.array(fuel_fractions) if fuel_fractions else np.empty((0,)),
        debug_msgs=debug_msgs,
    )
