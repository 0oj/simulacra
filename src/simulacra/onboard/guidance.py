"""Guidance: turn a Mission into a stream of Setpoints for the Controller.

A Mission is a scenario-level objective (e.g. a list of waypoints to
visit). A Guidance implementation owns the logic for converting that
objective plus the live sensor state into a Setpoint at each controller
tick.

For v0.1 only WaypointMission + WaypointGuidance ship. The Mission
Protocol is polymorphic so future mission types (sounding-rocket
profile, controlled descent, abort) can be added without changing
the Controller interface.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from simulacra.physics.quat import Orientation
from simulacra.physics.vec3 import Vec3

from .controller import SensorFrame, VehicleParams


@dataclass(frozen=True)
class Waypoint:
    """A target the rocket should reach and optionally hold."""
    position: Vec3
    hold_time: float = 0.0
    acceptance_radius: float = 1.0   # m
    max_speed: float = 5.0           # m/s


class Mission(Protocol):
    """Scenario-level objective consumed by a Guidance implementation."""


@dataclass(frozen=True)
class WaypointMission:
    waypoints: list[Waypoint] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.waypoints) == 0:
            raise ValueError("WaypointMission requires at least one waypoint")


@dataclass(frozen=True, slots=True)
class Setpoint:
    """What Guidance wants the rocket to be doing right now.

    Controllers may ignore individual fields. A pure-position tracker
    can ignore `velocity` and `accel_ff`; an attitude-only controller
    can ignore everything except `attitude`.
    """
    position: Vec3
    velocity: Vec3
    accel_ff: Vec3
    attitude: Orientation


class Guidance(Protocol):
    """A guidance implementation.

    `reset` is called once before the scenario starts. `update` is
    called every controller tick (so guidance runs at controller rate).
    Guidance sees the same SensorFrame the controller does, which lets
    it react to sensed state (e.g. detect arrival from sensed position,
    abort on stale GPS).
    """
    def reset(self, mission: Mission, vehicle: VehicleParams) -> None: ...
    def update(self, frame: SensorFrame, t_now: float) -> Setpoint: ...


def _upright_quat() -> Orientation:
    """Upright in this project means body +X aligned with world +Y."""
    return Orientation.from_wxyz(
        math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4),
    )


class WaypointGuidance:
    """Sequence a WaypointMission into position setpoints.

    Detects arrival from sensed position (uses GPS if available, falls
    back to baro for altitude only). Advances after the active waypoint's
    hold_time elapses since arrival.

    v0.1 emits position-only setpoints (zero velocity and accel_ff).
    Higher-order trajectory shaping is a known evolution noted in the
    spec.
    """

    def __init__(self) -> None:
        self._mission: WaypointMission | None = None
        self._idx: int = 0
        self._arrival_t: float | None = None
        self._upright = _upright_quat()

    def reset(self, mission: WaypointMission, vehicle: VehicleParams) -> None:
        if not isinstance(mission, WaypointMission):
            raise TypeError(
                f"WaypointGuidance requires WaypointMission, got {type(mission).__name__}"
            )
        self._mission = mission
        self._idx = 0
        self._arrival_t = None

    def update(self, frame: SensorFrame, t_now: float) -> Setpoint:
        if self._mission is None:
            raise RuntimeError("WaypointGuidance.update called before reset()")
        wp = self._mission.waypoints[self._idx]

        # Sensed position: GPS if available, else baro for altitude only.
        if frame.gps is not None:
            sensed_pos = frame.gps.value
        else:
            sensed_pos = Vec3(0.0, frame.baro.value, 0.0)

        # Arrival detection (Euclidean).
        dx = float(sensed_pos[0] - wp.position[0])
        dy = float(sensed_pos[1] - wp.position[1])
        dz = float(sensed_pos[2] - wp.position[2])
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        if dist <= wp.acceptance_radius:
            if self._arrival_t is None:
                self._arrival_t = t_now
            elif (t_now - self._arrival_t) >= wp.hold_time:
                if self._idx + 1 < len(self._mission.waypoints):
                    self._idx += 1
                    self._arrival_t = None
                    wp = self._mission.waypoints[self._idx]
        else:
            # Drifted out — reset arrival timer.
            self._arrival_t = None

        return Setpoint(
            position=wp.position,
            velocity=Vec3(0.0, 0.0, 0.0),
            accel_ff=Vec3(0.0, 0.0, 0.0),
            attitude=self._upright,
        )
