"""
Demo: Thrust Jitter (Random Walk)

Hovering rocket with random thrust direction noise but NO attitude control.
Demonstrates: How small random torques accumulate into large attitude errors.

This is why rockets need active attitude control - even tiny thrust
misalignments cause angular velocity to random-walk.

Setup:
- Upright hover, no gimbal control
- Thrust has 0.3° std random jitter each timestep
- No attitude feedback

Expected behavior:
- Angular velocity does a random walk
- Attitude slowly drifts
- Eventually tips over (after several seconds)

Compare with demo_noisy_hover to see how TVC prevents this.

Usage:
    uv run scenarios/demos/thrust_jitter.py
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
from simulacra.phenomena.actuators import ThrottleActuator, Actuators, ThrottleCommand
from simulacra.phenomena.sensors import Sensors
from simulacra.phenomena.errors import GaussianNoise
from simulacra.physics import RocketVec3, WorldVec3
from simulacra.onboard.controller import Controller


class ThrustOnlyController(Controller):
    """Maintains hover thrust but no attitude control."""

    def __init__(self, throttle, thrust_N):
        self.throttle = throttle
        self.thrust_N = thrust_N
        self.sensors = None

    def with_sensors(self, sensors):
        self.sensors = sensors
        return self

    def run(self):
        self.throttle.apply(ThrottleCommand(thrust_N=self.thrust_N))


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

    # Thrust with random jitter (no bias, just noise)
    throttle = ThrottleActuator(
        rocket,
        alignment_errors={
            "yaw": GaussianNoise(std=math.radians(0.3), seed=42),
            "pitch": GaussianNoise(std=math.radians(0.3), seed=43),
        }
    )
    actuators = Actuators(throttle=throttle)

    controller = ThrustOnlyController(throttle, thrust_N=weight)

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
    output_path = output_dir / "demo_thrust_jitter.ndjson"

    with open(output_path, "w") as f:
        scenario.run(duration=10.0, output=f)

    print(f"Wrote {output_path}", file=sys.stderr)
