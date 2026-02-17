"""
Demo: Spin (Angular Momentum)

Rocket with initial spin about its long axis (roll).
No thrust, no external torques. Angular momentum conserved.
Demonstrates: Conservation of angular momentum in free fall.

Setup:
- Rocket at 50m, no thrust
- Initial roll rate of 2 rad/s (about body X axis)
- Falls while spinning

Expected behavior:
- Constant angular velocity throughout (no torques)
- Falls under gravity
- Spin rate unchanged when it hits ground

Physics:
- L = I * ω = 10 * 2 = 20 kg·m²/s (about roll axis)
- This is conserved (no external torques about roll axis)

Usage:
    uv run scenarios/demos/spin.py
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
    def with_sensors(self, sensors):
        return self

    def run(self):
        pass


def make_scenario(initial_spin_rate: float = 2.0) -> Scenario:
    mass = 100.0
    gravity = 9.81

    rocket = Rocket(
        props=RocketProps(
            length=4.0,
            mass=mass,
            center_of_mass_offset=RocketVec3(2.0, 0.0, 0.0),
            inertia_diag=(50.0, 50.0, 10.0),  # Ixx=50, Iyy=50, Izz=10
            thrust_application_offset=RocketVec3(0.0, 0.0, 0.0),
        ),
        state=RocketState(
            position=WorldVec3(0.0, 50.0, 0.0),  # High up
            velocity=WorldVec3(0.0, 0.0, 0.0),
            orientation=get_upright_orientation(),
            angular_velocity_body=RocketVec3(initial_spin_rate, 0.0, 0.0),  # Roll
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
    scenario = make_scenario(initial_spin_rate=2.0)

    output_dir = pathlib.Path("/tmp/simulacra")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "demo_spin.ndjson"

    with open(output_path, "w") as f:
        scenario.run(duration=4.0, output=f)

    print(f"Wrote {output_path}", file=sys.stderr)
