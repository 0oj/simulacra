"""End-to-end smoke: full stack produces parseable NDJSON."""
from __future__ import annotations

import io
import json

import pytest

from simulacra.noumena import (
    Environment, Rocket, RocketProps, RocketState,
    get_upright_orientation,
)
from simulacra.onboard.controller import IdentityController
from simulacra.onboard.guidance import Waypoint, WaypointMission
from simulacra.physics.vec3 import Vec3, RocketVec3, WorldVec3
from simulacra.runtime import Scenario


def _build_setup():
    props = RocketProps(
        length=4.0, mass=140.0,
        center_of_mass_offset=RocketVec3(2.0, 0, 0),
        inertia_diag=(10.0, 360.0, 360.0),
        thrust_application_offset=RocketVec3(0, 0, 0),
    )
    state = RocketState(
        position=WorldVec3(0, 50.0, 0),
        velocity=WorldVec3(0, 0, 0),
        orientation=get_upright_orientation(),
        angular_velocity_body=RocketVec3(0, 0, 0),
        thrust_body=RocketVec3(0, 0, 0),
    )
    rocket = Rocket(props=props, state=state)
    env = Environment(gravity=WorldVec3(0, -9.81, 0), wind=WorldVec3(0, 0, 0))
    mission = WaypointMission(waypoints=[Waypoint(position=Vec3(0, 5, 0), hold_time=0.0)])
    return rocket, env, mission


def test_smoke_produces_well_formed_ndjson():
    rocket, env, mission = _build_setup()
    out = io.StringIO()
    Scenario(
        rocket=rocket, env=env, mission=mission,
        controller=IdentityController(),
        duration=1.0, seed=42,
    ).run(out)
    lines = out.getvalue().rstrip("\n").split("\n")
    parsed = [json.loads(L) for L in lines]
    assert parsed[0]["message_type"] == "metadata"
    n_var = sum(1 for p in parsed if p.get("message_type") == "variable")
    assert n_var > 0


def test_smoke_rocket_falls_without_thrust():
    """IdentityController gives zero thrust → rocket descends."""
    rocket, env, mission = _build_setup()
    out = io.StringIO()
    Scenario(
        rocket=rocket, env=env, mission=mission,
        controller=IdentityController(),
        duration=1.0, seed=42,
    ).run(out)
    lines = out.getvalue().rstrip("\n").split("\n")
    variables = [json.loads(L) for L in lines if json.loads(L).get("message_type") == "variable"]
    final_y = variables[-1]["position"][1]
    # Started at 50 m, no thrust, 1 s of gravity → should be lower.
    assert final_y < 50.0
