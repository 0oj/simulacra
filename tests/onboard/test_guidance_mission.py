"""Tests for mission data types and Setpoint."""
from __future__ import annotations

import pytest

from simulacra.physics.vec3 import Vec3
from simulacra.physics.quat import Orientation
from simulacra.onboard.guidance import Setpoint, Waypoint, WaypointMission


def test_waypoint_holds_position_and_hold_time():
    wp = Waypoint(position=Vec3(0.0, 10.0, 0.0), hold_time=2.0)
    assert wp.position[1] == 10.0
    assert wp.hold_time == 2.0
    assert wp.acceptance_radius == 1.0
    assert wp.max_speed == 5.0


def test_waypoint_mission_holds_list():
    wps = [
        Waypoint(position=Vec3(0.0,  5.0, 0.0), hold_time=1.0),
        Waypoint(position=Vec3(0.0, 10.0, 0.0), hold_time=2.0),
        Waypoint(position=Vec3(0.0,  1.0, 0.0), hold_time=0.5),
    ]
    mission = WaypointMission(waypoints=wps)
    assert len(mission.waypoints) == 3
    assert mission.waypoints[1].position[1] == 10.0


def test_waypoint_mission_rejects_empty():
    with pytest.raises(ValueError):
        WaypointMission(waypoints=[])


def test_setpoint_holds_pos_vel_accel_attitude():
    sp = Setpoint(
        position=Vec3(0.0, 10.0, 0.0),
        velocity=Vec3(0.0, 0.0, 0.0),
        accel_ff=Vec3(0.0, 0.0, 0.0),
        attitude=Orientation.identity(),
    )
    assert sp.position[1] == 10.0
    assert sp.velocity[0] == 0.0
    assert sp.accel_ff[0] == 0.0
