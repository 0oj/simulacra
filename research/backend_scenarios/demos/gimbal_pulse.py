"""
Demo: Gimbal Pulse

A single gimbal deflection pulse to show how TVC creates attitude change.
Demonstrates: Gimbal → torque → angular velocity → attitude change.

Setup:
- Upright hovering rocket
- At t=1s, deflect gimbal 10° for 0.5s, then return to center
- No closed-loop control, just open-loop pulse

Expected behavior:
- Before pulse: hovering, no rotation
- During pulse: angular acceleration, rocket starts tilting
- After pulse: constant angular velocity (no more torque)
- Rocket continues rotating at that rate

Physics:
- Gimbal 10° with 981N thrust, 2m arm → torque = 981 * 2 * sin(10°) ≈ 341 N·m
- With Iyy = 50 kg·m², angular accel = 6.8 rad/s²
- After 0.5s: ω = 3.4 rad/s ≈ 195°/s

Usage:
    uv run scenarios/demos/gimbal_pulse.py
"""

import sys
import os
import math
import pathlib

script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(os.path.dirname(script_dir))
src_dir = os.path.join(backend_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from simulacra.noumena import (
    Rocket,
    RocketProps,
    RocketState,
    Environment,
    Kernel,
    Scenario,
    get_upright_orientation,
)
from simulacra.phenomena.actuators import (
    ThrottleActuator,
    GimbalActuator,
    Actuators,
    ThrottleCommand,
    GimbalCommand,
)
from simulacra.phenomena.sensors import Sensors
from simulacra.physics import RocketVec3, WorldVec3
from simulacra.onboard.controller import Controller


class GimbalPulseController(Controller):
    """Applies a gimbal pulse at a specific time."""

    def __init__(self, throttle, gimbal, thrust_N, pulse_start, pulse_duration, pulse_angle_deg):
        self.throttle = throttle
        self.gimbal = gimbal
        self.thrust_N = thrust_N
        self.pulse_start = pulse_start
        self.pulse_end = pulse_start + pulse_duration
        self.pulse_angle = math.radians(pulse_angle_deg)
        self.sensors = None
        self._time = 0.0

    def with_sensors(self, sensors):
        self.sensors = sensors
        return self

    def run(self):
        # Update time from sensors (they track simulation time)
        if self.sensors:
            self._time = self.sensors._time

        # Constant thrust
        self.throttle.apply(ThrottleCommand(thrust_N=self.thrust_N))

        # Gimbal pulse
        if self.pulse_start <= self._time < self.pulse_end:
            self.gimbal.apply(GimbalCommand(
                desired_outer_angle=self.pulse_angle,
                desired_inner_angle=0.0,
            ))
        else:
            self.gimbal.apply(GimbalCommand(
                desired_outer_angle=0.0,
                desired_inner_angle=0.0,
            ))


def make_scenario() -> Scenario:
    mass = 100.0
    gravity = 9.81
    weight = mass * gravity

    rocket = Rocket(
        props=RocketProps(
            length=4.0,
            mass=mass,
            center_of_mass_offset=RocketVec3(2.0, 0.0, 0.0),
            inertia_diag=(50.0, 50.0, 10.0),
            thrust_application_offset=RocketVec3(0.0, 0.0, 0.0),
        ),
        state=RocketState(
            position=WorldVec3(0.0, 10.0, 0.0),
            velocity=WorldVec3(0.0, 0.0, 0.0),
            orientation=get_upright_orientation(),
            angular_velocity_body=RocketVec3(0.0, 0.0, 0.0),
            thrust_body=RocketVec3(weight, 0.0, 0.0),
        ),
    )

    env = Environment(gravity=WorldVec3(0.0, -gravity, 0.0))
    kernel = Kernel(rocket=rocket, environment=env)
    sensors = Sensors(kernel)

    throttle = ThrottleActuator(rocket)
    gimbal = GimbalActuator(
        rocket,
        max_outer_angle=math.radians(15.0),
        max_inner_angle=math.radians(15.0),
    )
    actuators = Actuators(throttle=throttle, gimbal=gimbal)

    controller = GimbalPulseController(
        throttle=throttle,
        gimbal=gimbal,
        thrust_N=weight,
        pulse_start=1.0,      # Start pulse at t=1s
        pulse_duration=0.5,   # Pulse for 0.5s
        pulse_angle_deg=10.0, # 10° deflection
    )

    return Scenario(
        kernel=kernel,
        controller=controller,
        sensors=sensors,
        actuators=actuators,
        dt=0.02,
    )


if __name__ == "__main__":
    scenario = make_scenario()

    output_dir = pathlib.Path("/tmp/simulacra")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "demo_gimbal_pulse.ndjson"

    with open(output_path, "w") as f:
        scenario.run(duration=4.0, output=f)

    print(f"Wrote {output_path}", file=sys.stderr)
