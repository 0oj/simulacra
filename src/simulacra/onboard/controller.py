"""SDK data types and Controller Protocol.

These types are the public contract that engineers writing controllers
program against. They are deliberately plain — frozen dataclasses with
no methods — so they map directly to C structs when the controller is
later ported to a flight computer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from simulacra.physics.quat import Orientation
from simulacra.physics.vec3 import Vec3


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Stamped(Generic[T]):
    """A sensor sample with its production timestamp.

    `t` is seconds since simulator/flight boot. `seq` is a monotonically
    increasing counter the controller can use to detect "did this update
    since I last looked?" without comparing timestamps directly.
    """
    value: T
    t: float
    seq: int


@dataclass(frozen=True, slots=True)
class SensorFrame:
    """Snapshot of all sensor topics as the controller sees them at one tick.

    Each field carries its own `Stamped` so the controller knows how
    fresh each reading is. `gps` may be None when GPS lock is lost.
    """
    t_now: float
    imu_accel: Stamped[Vec3]
    imu_gyro: Stamped[Vec3]
    baro: Stamped[float]
    gps: Stamped[Vec3] | None
    velocity: Stamped[Vec3]
    orientation: Stamped[Orientation]


@dataclass(frozen=True, slots=True)
class ActuatorCommand:
    """Single command output from the controller for one tick."""
    throttle: float          # 0..1
    gimbal_pitch: float      # rad
    gimbal_yaw: float        # rad


@dataclass(frozen=True, slots=True)
class VehicleParams:
    """Vehicle properties the controller may know at design time.

    Runtime mass changes as fuel burns and must be sensed or estimated;
    those values are not here.
    """
    dry_mass: float          # kg, structure + payload, no fuel
    full_fuel_mass: float    # kg, fuel at t=0
    inertia_dry: Vec3        # kg·m², principal-axis diagonal, dry
    length: float            # m, nose-to-base
    max_thrust: float        # N, at full throttle
    max_gimbal_angle: float  # rad
    max_gimbal_rate: float   # rad/s


class Controller(Protocol):
    """Protocol for any controller.

    Implementations must:
    - Declare a fixed tick rate via the class-level `rate_hz` attribute.
    - Implement `reset(mission, vehicle)`, called once before the scenario starts.
    - Implement `step(frame, setpoint, dt)`, called every 1/rate_hz seconds.

    `step` must be pure-ish: same inputs should give the same outputs
    given the same internal state. No I/O, no time-of-day reads, no
    allocations in the hot loop (in C, this becomes a timer ISR).
    """
    rate_hz: float

    def reset(self, mission, vehicle: VehicleParams) -> None: ...
    def step(self, frame: SensorFrame, setpoint, dt: float) -> ActuatorCommand: ...


class IdentityController:
    """A controller that outputs zero throttle and zero gimbal every tick.

    Useful for plumbing tests: it lets a scenario run end-to-end without
    a real control algorithm and proves the runtime calls the controller
    correctly. The rocket will of course fall down.
    """
    rate_hz: float = 100.0

    def reset(self, mission, vehicle: VehicleParams) -> None:
        return None

    def step(self, frame: SensorFrame, setpoint, dt: float) -> ActuatorCommand:
        return ActuatorCommand(throttle=0.0, gimbal_pitch=0.0, gimbal_yaw=0.0)
