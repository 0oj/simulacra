"""
Demo: Thrust Misalignment Torque

Thrust applied off-axis creates a torque that rotates the rocket.
Demonstrates: τ = r × F, angular acceleration from offset thrust.

Setup:
- Upright rocket with hover thrust
- Thrust has 2° yaw misalignment (constant bias)
- No attitude control

Expected behavior:
- Rocket starts rotating about its long axis (body Z)
- Angular velocity increases linearly (constant torque)
- Eventually the rocket spirals out of control

Physics:
- Thrust = 981 N, moment arm = 2m
- Misalignment 2° → lateral component = 981 * sin(2°) ≈ 34 N
- Torque = 34 N * 2 m = 68 N·m
- With Izz = 10 kg·m², angular accel = 6.8 rad/s²

Usage:
    uv run scenarios/demos/thrust_torque.py
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
from simulacra.phenomena.actuators import ThrottleActuator, Actuators
from simulacra.phenomena.sensors import Sensors
from simulacra.phenomena.errors import Bias
from simulacra.physics import RocketVec3, WorldVec3
from simulacra.onboard.controller import Controller


class ConstantThrustController(Controller):
    """Maintains constant thrust."""

    def __init__(self, throttle, thrust_N):
        self.throttle = throttle
        self.thrust_N = thrust_N
        self.sensors = None

    def with_sensors(self, sensors):
        self.sensors = sensors
        return self

    def run(self):
        from simulacra.phenomena.actuators import ThrottleCommand
        self.throttle.apply(ThrottleCommand(thrust_N=self.thrust_N))


def make_scenario(misalignment_deg: float = 2.0) -> Scenario:
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

    # Thrust with constant misalignment
    throttle = ThrottleActuator(
        rocket,
        alignment_errors={
            "yaw": Bias(offset=math.radians(misalignment_deg)),
        }
    )
    actuators = Actuators(throttle=throttle)
    controller = ConstantThrustController(throttle, thrust_N=weight)

    return Scenario(
        kernel=kernel,
        controller=controller,
        sensors=sensors,
        actuators=actuators,
        dt=0.02,
    )


if __name__ == "__main__":
    scenario = make_scenario(misalignment_deg=2.0)

    output_dir = pathlib.Path("/tmp/simulacra")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "demo_thrust_torque.ndjson"

    with open(output_path, "w") as f:
        scenario.run(duration=3.0, output=f)

    print(f"Wrote {output_path}", file=sys.stderr)
