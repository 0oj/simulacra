"""Tests for WaypointGuidance: arrival detection, sequencing, setpoint output."""
from __future__ import annotations

import pytest

from simulacra.physics.vec3 import Vec3
from simulacra.physics.quat import Orientation

from simulacra.onboard.controller import SensorFrame, Stamped, VehicleParams
from simulacra.onboard.guidance import Waypoint, WaypointMission, WaypointGuidance


def make_frame(t: float, pos: Vec3, vel: Vec3 = None) -> SensorFrame:
    if vel is None:
        vel = Vec3(0, 0, 0)
    return SensorFrame(
        t_now=t,
        imu_accel=Stamped(value=Vec3(0, 0, 0), t=t, seq=0),
        imu_gyro=Stamped(value=Vec3(0, 0, 0), t=t, seq=0),
        baro=Stamped(value=float(pos[1]), t=t, seq=0),
        gps=Stamped(value=pos, t=t, seq=0),
        velocity=Stamped(value=vel, t=t, seq=0),
        orientation=Stamped(value=Orientation.identity(), t=t, seq=0),
    )


def make_vehicle() -> VehicleParams:
    return VehicleParams(
        dry_mass=60.0, full_fuel_mass=80.0, inertia_dry=Vec3(10, 360, 360),
        length=4.0, max_thrust=2000.0, max_gimbal_angle=0.349, max_gimbal_rate=2.0,
    )


def test_initial_setpoint_targets_first_waypoint():
    mission = WaypointMission(waypoints=[
        Waypoint(position=Vec3(0.0, 10.0, 0.0), hold_time=1.0),
        Waypoint(position=Vec3(0.0,  1.0, 0.0), hold_time=0.0),
    ])
    g = WaypointGuidance()
    g.reset(mission, make_vehicle())
    sp = g.update(make_frame(t=0.0, pos=Vec3(0, 0, 0)), t_now=0.0)
    assert float(sp.position[1]) == pytest.approx(10.0)


def test_arrival_within_radius_does_not_advance_before_hold_time():
    mission = WaypointMission(waypoints=[
        Waypoint(position=Vec3(0.0, 10.0, 0.0), hold_time=1.0, acceptance_radius=0.5),
        Waypoint(position=Vec3(0.0, 20.0, 0.0), hold_time=0.0),
    ])
    g = WaypointGuidance()
    g.reset(mission, make_vehicle())
    sp = g.update(make_frame(t=2.0, pos=Vec3(0, 10.0, 0)), t_now=2.0)
    assert float(sp.position[1]) == pytest.approx(10.0)
    sp = g.update(make_frame(t=2.5, pos=Vec3(0, 10.0, 0)), t_now=2.5)
    assert float(sp.position[1]) == pytest.approx(10.0)


def test_after_hold_time_advances_to_next_waypoint():
    mission = WaypointMission(waypoints=[
        Waypoint(position=Vec3(0.0, 10.0, 0.0), hold_time=1.0, acceptance_radius=0.5),
        Waypoint(position=Vec3(0.0, 20.0, 0.0), hold_time=0.0),
    ])
    g = WaypointGuidance()
    g.reset(mission, make_vehicle())
    g.update(make_frame(t=2.0, pos=Vec3(0, 10.0, 0)), t_now=2.0)
    g.update(make_frame(t=2.5, pos=Vec3(0, 10.0, 0)), t_now=2.5)
    sp = g.update(make_frame(t=3.5, pos=Vec3(0, 10.0, 0)), t_now=3.5)
    assert float(sp.position[1]) == pytest.approx(20.0)


def test_final_waypoint_does_not_advance_past_end():
    mission = WaypointMission(waypoints=[
        Waypoint(position=Vec3(0.0, 10.0, 0.0), hold_time=1.0, acceptance_radius=0.5),
    ])
    g = WaypointGuidance()
    g.reset(mission, make_vehicle())
    g.update(make_frame(t=2.0, pos=Vec3(0, 10.0, 0)), t_now=2.0)
    sp = g.update(make_frame(t=10.0, pos=Vec3(0, 10.0, 0)), t_now=10.0)
    assert float(sp.position[1]) == pytest.approx(10.0)


def test_setpoint_velocity_zero_for_simple_position_target():
    mission = WaypointMission(waypoints=[
        Waypoint(position=Vec3(0.0, 10.0, 0.0), hold_time=0.0),
    ])
    g = WaypointGuidance()
    g.reset(mission, make_vehicle())
    sp = g.update(make_frame(t=0.0, pos=Vec3(0, 0, 0)), t_now=0.0)
    assert float(sp.velocity[0]) == 0.0
    assert float(sp.velocity[1]) == 0.0
    assert float(sp.velocity[2]) == 0.0
    assert float(sp.accel_ff[0]) == 0.0
