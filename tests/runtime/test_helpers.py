"""Tests for runtime helpers: sensors_for, actuators_for."""
from __future__ import annotations

import pytest

from simulacra.noumena import (
    Environment, Kernel, Rocket, RocketProps, RocketState,
    get_upright_orientation,
)
from simulacra.physics.vec3 import RocketVec3, WorldVec3
from simulacra.runtime.helpers import actuators_for, sensors_for


def _build_rocket() -> Rocket:
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
    return Rocket(props=props, state=state)


def test_sensors_for_returns_named_sensor_container():
    rocket = _build_rocket()
    env = Environment(gravity=WorldVec3(0, -9.81, 0), wind=WorldVec3(0, 0, 0))
    kernel = Kernel(rocket=rocket, environment=env)
    sensors = sensors_for(kernel)
    assert hasattr(sensors, "GPS")
    assert hasattr(sensors, "Barometer")
    assert hasattr(sensors, "IMU")
    assert hasattr(sensors, "VelocitySensor")
    assert hasattr(sensors, "OrientationSensor")


def test_actuators_for_returns_throttle_and_gimbal():
    rocket = _build_rocket()
    actuators = actuators_for(rocket)
    # Actuators bundles throttle + gimbal — the new SDK uses both.
    assert actuators._throttle is not None
    assert actuators._gimbal is not None
