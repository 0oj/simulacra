"""PD altitude controller — the simplest possible reference controller.

Holds the rocket at the altitude given by the current setpoint. No
gimbal control — assumes vertical flight only and outputs zero gimbal.

This is the most useful starting point for an engineer reading the
codebase: it's about 30 lines of actual control logic, all in `step`.
"""
from __future__ import annotations

from simulacra.onboard.controller import (
    ActuatorCommand, SensorFrame, VehicleParams,
)
from simulacra.onboard.guidance import Mission, Setpoint


class PIDHoverController:
    """PD altitude hold using barometer altitude and velocity.y."""

    rate_hz: float = 100.0

    def __init__(self, Kp: float = 10.0, Kd: float = 5.0) -> None:
        self.Kp = Kp
        self.Kd = Kd
        self._weight_N: float = 0.0
        self._max_thrust: float = 1.0   # set in reset

    def reset(self, mission: Mission, vehicle: VehicleParams) -> None:
        # Feed-forward weight = (dry + full fuel) × g. Approximate; the
        # actual weight decreases as fuel burns.
        g = 9.81
        self._weight_N = (vehicle.dry_mass + vehicle.full_fuel_mass) * g
        self._max_thrust = vehicle.max_thrust

    def step(
        self,
        frame: SensorFrame,
        setpoint: Setpoint,
        dt: float,
    ) -> ActuatorCommand:
        target_alt = float(setpoint.position[1])
        alt = float(frame.baro.value)
        vy = float(frame.velocity.value[1])

        error = target_alt - alt
        thrust_N = self._weight_N + self.Kp * error - self.Kd * vy
        thrust_N = max(0.0, min(thrust_N, self._max_thrust))

        return ActuatorCommand(
            throttle=thrust_N / self._max_thrust if self._max_thrust > 0 else 0.0,
            gimbal_pitch=0.0,
            gimbal_yaw=0.0,
        )
