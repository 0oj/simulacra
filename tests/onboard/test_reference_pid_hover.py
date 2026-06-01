"""Tests for the PID hover reference controller."""
from __future__ import annotations

import io
import json

from simulacra.noumena import (
    Environment, Rocket, RocketProps, RocketState,
    get_upright_orientation,
)
from simulacra.onboard.guidance import Waypoint, WaypointMission
from simulacra.onboard.reference import PIDHoverController
from simulacra.physics.vec3 import Vec3, RocketVec3, WorldVec3
from simulacra.runtime import Scenario


def _build_setup(start_alt: float, target_alt: float):
    props = RocketProps(
        length=4.0, mass=140.0,
        center_of_mass_offset=RocketVec3(2.0, 0, 0),
        inertia_diag=(10.0, 360.0, 360.0),
        thrust_application_offset=RocketVec3(0, 0, 0),
    )
    state = RocketState(
        position=WorldVec3(0, start_alt, 0),
        velocity=WorldVec3(0, 0, 0),
        orientation=get_upright_orientation(),
        angular_velocity_body=RocketVec3(0, 0, 0),
        thrust_body=RocketVec3(0, 0, 0),
    )
    rocket = Rocket(props=props, state=state)
    env = Environment(gravity=WorldVec3(0, -9.81, 0), wind=WorldVec3(0, 0, 0))
    mission = WaypointMission(waypoints=[
        Waypoint(position=Vec3(0, target_alt, 0), hold_time=5.0),
    ])
    return rocket, env, mission


def test_pid_hover_runs_and_does_not_crash():
    rocket, env, mission = _build_setup(start_alt=10.0, target_alt=10.0)
    out = io.StringIO()
    Scenario(
        rocket=rocket, env=env, mission=mission,
        controller=PIDHoverController(),
        duration=2.0, seed=42,
    ).run(out)
    lines = out.getvalue().rstrip("\n").split("\n")
    assert len(lines) > 1
    assert json.loads(lines[0])["message_type"] == "metadata"


def test_pid_hover_stays_near_target_altitude():
    """Started at target alt, PID should hold roughly there."""
    rocket, env, mission = _build_setup(start_alt=20.0, target_alt=20.0)
    out = io.StringIO()
    Scenario(
        rocket=rocket, env=env, mission=mission,
        controller=PIDHoverController(Kp=200.0, Kd=80.0),
        duration=3.0, seed=42,
    ).run(out)
    lines = out.getvalue().rstrip("\n").split("\n")
    variables = [json.loads(L) for L in lines if json.loads(L).get("message_type") == "variable"]
    # Sample altitudes after a brief transient.
    altitudes_after_settle = [v["position"][1] for v in variables[100:]]
    # Should remain within ±5 m of the target (loose because barometer noise + PID transient).
    assert all(15.0 < y < 25.0 for y in altitudes_after_settle), \
        f"Altitudes drifted: min={min(altitudes_after_settle):.2f}, max={max(altitudes_after_settle):.2f}"
