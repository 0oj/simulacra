"""End-to-end Scenario tests using IdentityController."""
from __future__ import annotations

import io
import json

import pytest

from simulacra.noumena import (
    Environment, Rocket, RocketProps, RocketState,
    get_upright_orientation,
)
from simulacra.onboard.controller import IdentityController
from simulacra.onboard.guidance import Waypoint, WaypointGuidance, WaypointMission
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
        position=WorldVec3(0, 10, 0),
        velocity=WorldVec3(0, 0, 0),
        orientation=get_upright_orientation(),
        angular_velocity_body=RocketVec3(0, 0, 0),
        thrust_body=RocketVec3(0, 0, 0),
    )
    rocket = Rocket(props=props, state=state)
    env = Environment(gravity=WorldVec3(0, -9.81, 0), wind=WorldVec3(0, 0, 0))
    mission = WaypointMission(waypoints=[Waypoint(position=Vec3(0, 5, 0), hold_time=0.0)])
    return rocket, env, mission


def test_scenario_runs_with_identity_controller():
    rocket, env, mission = _build_setup()
    out = io.StringIO()
    Scenario(
        rocket=rocket, env=env, mission=mission,
        controller=IdentityController(), duration=0.5, seed=42,
    ).run(out)
    text = out.getvalue()
    assert len(text) > 0
    lines = text.rstrip("\n").split("\n")
    assert json.loads(lines[0])["message_type"] == "metadata"
    assert any(json.loads(L).get("message_type") == "variable" for L in lines[1:])


def test_scenario_uses_provided_guidance():
    rocket, env, mission = _build_setup()
    out = io.StringIO()
    Scenario(
        rocket=rocket, env=env, mission=mission,
        controller=IdentityController(),
        guidance=WaypointGuidance(),
        duration=0.5, seed=42,
    ).run(out)
    assert len(out.getvalue()) > 0


def test_two_runs_with_same_seed_produce_identical_output():
    rocket1, env1, mission1 = _build_setup()
    rocket2, env2, mission2 = _build_setup()

    def run_one(rocket, env, mission):
        out = io.StringIO()
        Scenario(
            rocket=rocket, env=env, mission=mission,
            controller=IdentityController(),
            duration=0.5, seed=42,
        ).run(out)
        return out.getvalue()

    text1 = run_one(rocket1, env1, mission1)
    text2 = run_one(rocket2, env2, mission2)
    assert text1 == text2, "Same seed must yield identical NDJSON"
