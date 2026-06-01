"""Unit tests for the SDK's plain-data types in onboard/controller.py."""
from __future__ import annotations

import pytest

from simulacra.physics.vec3 import Vec3
from simulacra.physics.quat import Orientation

from simulacra.onboard.controller import (
    Stamped,
    SensorFrame,
    ActuatorCommand,
    VehicleParams,
)


def test_stamped_carries_value_and_timestamp():
    s = Stamped(value=1.5, t=0.42, seq=7)
    assert s.value == 1.5
    assert s.t == 0.42
    assert s.seq == 7


def test_stamped_is_frozen():
    s = Stamped(value=1.0, t=0.0, seq=0)
    with pytest.raises((AttributeError, Exception)):
        s.value = 2.0


def _upright() -> Orientation:
    import math
    return Orientation.from_wxyz(math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))


def test_sensor_frame_holds_per_topic_stamps():
    accel = Vec3(0.0, -9.81, 0.0)
    gyro = Vec3(0.0, 0.0, 0.0)
    pos = Vec3(0.0, 100.0, 0.0)
    vel = Vec3(0.0, -5.0, 0.0)
    ori = _upright()
    frame = SensorFrame(
        t_now=1.234,
        imu_accel=Stamped(value=accel, t=1.230, seq=10),
        imu_gyro=Stamped(value=gyro, t=1.230, seq=10),
        baro=Stamped(value=100.0, t=1.220, seq=5),
        gps=Stamped(value=pos, t=1.000, seq=2),
        velocity=Stamped(value=vel, t=1.220, seq=5),
        orientation=Stamped(value=ori, t=1.230, seq=10),
    )
    assert frame.t_now == 1.234
    assert frame.gps.t == 1.000
    assert frame.t_now - frame.gps.t == pytest.approx(0.234)


def test_sensor_frame_gps_may_be_none():
    frame = SensorFrame(
        t_now=0.0,
        imu_accel=Stamped(value=Vec3(0.0, 0.0, 0.0), t=0.0, seq=0),
        imu_gyro=Stamped(value=Vec3(0.0, 0.0, 0.0), t=0.0, seq=0),
        baro=Stamped(value=0.0, t=0.0, seq=0),
        gps=None,
        velocity=Stamped(value=Vec3(0.0, 0.0, 0.0), t=0.0, seq=0),
        orientation=Stamped(value=_upright(), t=0.0, seq=0),
    )
    assert frame.gps is None


def test_actuator_command_holds_throttle_and_gimbal():
    cmd = ActuatorCommand(throttle=0.7, gimbal_pitch=0.05, gimbal_yaw=-0.03)
    assert cmd.throttle == 0.7
    assert cmd.gimbal_pitch == 0.05
    assert cmd.gimbal_yaw == -0.03


def test_actuator_command_is_frozen():
    cmd = ActuatorCommand(throttle=0.0, gimbal_pitch=0.0, gimbal_yaw=0.0)
    with pytest.raises((AttributeError, Exception)):
        cmd.throttle = 1.0


def test_vehicle_params_holds_design_time_constants():
    inertia = Vec3(10.0, 360.0, 360.0)
    vp = VehicleParams(
        dry_mass=60.0,
        full_fuel_mass=80.0,
        inertia_dry=inertia,
        length=4.0,
        max_thrust=2000.0,
        max_gimbal_angle=0.349,
        max_gimbal_rate=2.0,
    )
    assert vp.dry_mass == 60.0
    assert vp.full_fuel_mass == 80.0
    assert vp.max_thrust == 2000.0


def test_identity_controller_returns_zero_command():
    from simulacra.onboard.controller import IdentityController
    ctrl = IdentityController()
    assert ctrl.rate_hz == 100.0
    vp = VehicleParams(
        dry_mass=60.0, full_fuel_mass=80.0, inertia_dry=Vec3(10.0, 360.0, 360.0),
        length=4.0, max_thrust=2000.0, max_gimbal_angle=0.349, max_gimbal_rate=2.0,
    )
    ctrl.reset(mission=None, vehicle=vp)
    frame = SensorFrame(
        t_now=0.0,
        imu_accel=Stamped(value=Vec3(0.0, 0.0, 0.0), t=0.0, seq=0),
        imu_gyro=Stamped(value=Vec3(0.0, 0.0, 0.0), t=0.0, seq=0),
        baro=Stamped(value=0.0, t=0.0, seq=0),
        gps=Stamped(value=Vec3(0.0, 0.0, 0.0), t=0.0, seq=0),
        velocity=Stamped(value=Vec3(0.0, 0.0, 0.0), t=0.0, seq=0),
        orientation=Stamped(value=_upright(), t=0.0, seq=0),
    )
    cmd = ctrl.step(frame, setpoint=None, dt=0.01)
    assert cmd.throttle == 0.0
    assert cmd.gimbal_pitch == 0.0
    assert cmd.gimbal_yaw == 0.0
