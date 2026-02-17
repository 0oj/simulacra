"""
Demo: Free Fall

A rocket at 10m altitude with no thrust. Pure gravitational acceleration.
Demonstrates: F = ma, constant acceleration, parabolic trajectory.

Expected behavior:
- Position: y(t) = 10 - 0.5*g*t² = 10 - 4.905*t²
- Velocity: v(t) = -g*t = -9.81*t
- Hits ground at t = sqrt(20/9.81) ≈ 1.43s
- Impact speed: ~14 m/s

Usage:
    uv run scenarios/demos/freefall.py
"""

import sys
import os
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
from simulacra.phenomena.actuators import Actuators
from simulacra.phenomena.sensors import Sensors
from simulacra.physics import RocketVec3, WorldVec3
from simulacra.onboard.controller import Controller


class NoController(Controller):
    """Does nothing - no thrust, no gimbal commands."""

    def with_sensors(self, sensors):
        return self

    def run(self):
        pass


def make_scenario() -> Scenario:
    mass = 100.0  # kg
    gravity = 9.81  # m/s²

    rocket = Rocket(
        props=RocketProps(
            length=4.0,
            mass=mass,
            center_of_mass_offset=RocketVec3(2.0, 0.0, 0.0),
            inertia_diag=(50.0, 50.0, 10.0),
            thrust_application_offset=RocketVec3(0.0, 0.0, 0.0),
        ),
        state=RocketState(
            position=WorldVec3(0.0, 10.0, 0.0),  # Start at 10m
            velocity=WorldVec3(0.0, 0.0, 0.0),   # At rest
            orientation=get_upright_orientation(),
            angular_velocity_body=RocketVec3(0.0, 0.0, 0.0),
            thrust_body=RocketVec3(0.0, 0.0, 0.0),  # No thrust
        ),
    )

    env = Environment(gravity=WorldVec3(0.0, -gravity, 0.0))
    kernel = Kernel(rocket=rocket, environment=env)
    sensors = Sensors(kernel)
    actuators = Actuators()
    controller = NoController()

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
    output_path = output_dir / "demo_freefall.ndjson"

    with open(output_path, "w") as f:
        scenario.run(duration=2.0, output=f)

    print(f"Wrote {output_path}", file=sys.stderr)
