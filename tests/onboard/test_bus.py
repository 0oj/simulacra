"""Unit tests for SensorBus push/snapshot semantics."""
from __future__ import annotations

import pytest

from simulacra.physics.vec3 import Vec3
from simulacra.physics.quat import Orientation

from simulacra.onboard.bus import SensorBus
from simulacra.onboard.controller import SensorFrame


def test_snapshot_before_any_push_raises():
    bus = SensorBus()
    with pytest.raises(RuntimeError):
        bus.snapshot(t_now=0.0)


def test_single_push_then_snapshot_yields_value():
    bus = SensorBus()
    bus.push_imu(t=0.01, accel=Vec3(0.0, -9.81, 0.0), gyro=Vec3(0.0, 0.0, 0.0))
    bus.push_baro(t=0.02, altitude=100.0)
    bus.push_gps(t=0.05, position=Vec3(0.0, 100.0, 0.0))
    bus.push_velocity(t=0.02, velocity=Vec3(0.0, -5.0, 0.0))
    bus.push_orientation(t=0.02, orientation=Orientation.identity())
    frame = bus.snapshot(t_now=0.10)
    assert isinstance(frame, SensorFrame)
    assert frame.t_now == 0.10
    assert float(frame.imu_accel.value[1]) == pytest.approx(-9.81)
    assert frame.baro.value == 100.0
    assert frame.gps is not None
    assert float(frame.gps.value[1]) == pytest.approx(100.0)


def test_repeated_push_increments_seq():
    bus = SensorBus()
    bus.push_baro(t=0.01, altitude=100.0)
    bus.push_baro(t=0.02, altitude=101.0)
    bus.push_baro(t=0.03, altitude=102.0)
    bus.push_imu(t=0.01, accel=Vec3(0, 0, 0), gyro=Vec3(0, 0, 0))
    bus.push_velocity(t=0.01, velocity=Vec3(0, 0, 0))
    bus.push_orientation(t=0.01, orientation=Orientation.identity())
    bus.push_gps(t=0.01, position=Vec3(0, 0, 0))

    frame = bus.snapshot(t_now=0.04)
    assert frame.baro.value == 102.0
    assert frame.baro.seq == 2   # 0-indexed: pushes 0, 1, 2
    assert frame.baro.t == 0.03


def test_gps_unavailable_yields_none():
    bus = SensorBus()
    bus.push_imu(t=0.01, accel=Vec3(0, 0, 0), gyro=Vec3(0, 0, 0))
    bus.push_baro(t=0.01, altitude=0.0)
    bus.push_velocity(t=0.01, velocity=Vec3(0, 0, 0))
    bus.push_orientation(t=0.01, orientation=Orientation.identity())
    # No GPS push.
    frame = bus.snapshot(t_now=0.02)
    assert frame.gps is None
