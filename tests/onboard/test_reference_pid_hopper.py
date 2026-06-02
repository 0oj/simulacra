"""Tests for the PIDHopperController reference (cascaded PID)."""
from __future__ import annotations

import io
import json

import pytest

from simulacra.noumena import (
    Environment, Rocket, RocketProps, RocketState,
    get_upright_orientation,
)
from simulacra.onboard.guidance import Waypoint, WaypointMission
from simulacra.onboard.reference import PIDHopperController
from simulacra.physics.vec3 import Vec3, RocketVec3, WorldVec3
from simulacra.runtime import Scenario


def _build(start_alt: float, mission: WaypointMission):
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
    return rocket, env, mission


def test_pid_hopper_runs_without_crash():
    """Smoke: a single waypoint hop is produced as well-formed NDJSON."""
    mission = WaypointMission(waypoints=[
        Waypoint(position=Vec3(0, 10, 0), hold_time=2.0, acceptance_radius=1.0),
    ])
    rocket, env, mission = _build(start_alt=1.0, mission=mission)
    out = io.StringIO()
    Scenario(
        rocket=rocket, env=env, mission=mission,
        controller=PIDHopperController(),
        duration=3.0, seed=42,
    ).run(out)
    lines = out.getvalue().rstrip("\n").split("\n")
    assert len(lines) > 1
    assert json.loads(lines[0])["message_type"] == "metadata"


def test_pid_hopper_ascends_toward_target():
    """Started low, with a 10 m waypoint, should gain altitude."""
    mission = WaypointMission(waypoints=[
        Waypoint(position=Vec3(0, 10, 0), hold_time=3.0, acceptance_radius=1.0),
    ])
    rocket, env, mission = _build(start_alt=1.0, mission=mission)
    out = io.StringIO()
    Scenario(
        rocket=rocket, env=env, mission=mission,
        controller=PIDHopperController(),
        duration=4.0, seed=42,
    ).run(out)
    lines = out.getvalue().rstrip("\n").split("\n")
    variables = [json.loads(L) for L in lines if json.loads(L).get("message_type") == "variable"]
    max_y = max(v["position"][1] for v in variables)
    # Should at least get above 5 m within 4 seconds of TVC climbing.
    assert max_y > 5.0, f"max altitude reached only {max_y:.2f} m"
