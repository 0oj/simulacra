"""
Demo: Tilt and Tipover

Rocket starts tilted 15° from vertical with hover thrust but no attitude control.
Gravity torque causes it to tip over.
Demonstrates: Gravity torque, rotational dynamics, unstable equilibrium.

Expected behavior:
- Initially tilted 15°
- Gravity torque accelerates rotation
- Tips over completely (reaches 90°+ tilt)
- Eventually crashes

Physics:
- Gravity torque τ = m*g*r*sin(θ) where r = CoM offset
- For 100kg, g=9.81, r=2m, θ=15°: τ ≈ 508 N·m
- This torque increases as tilt increases (until 90°)

Usage:
    uv run scenarios/demos/tilt_tipover.py
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

import numpy as np
from scipy.spatial.transform import Rotation as R

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
from simulacra.physics import RocketVec3, WorldVec3, Orientation
from simulacra.onboard.controller import Controller


class ConstantThrustController(Controller):
    """Maintains constant thrust (no attitude control)."""

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


def make_scenario(initial_tilt_deg: float = 15.0) -> Scenario:
    mass = 100.0
    gravity = 9.81
    weight = mass * gravity

    # Create tilted orientation
    upright = get_upright_orientation()
    r_upright = upright._as_scipy_rotation()
    r_tilt = R.from_rotvec(math.radians(initial_tilt_deg) * np.array([0, 0, 1]))
    r_combined = r_tilt * r_upright
    x, y, z, w = r_combined.as_quat()
    orientation = Orientation.from_wxyz(w, x, y, z)

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
            orientation=orientation,
            angular_velocity_body=RocketVec3(0.0, 0.0, 0.0),
            thrust_body=RocketVec3(weight, 0.0, 0.0),  # Hover thrust
        ),
    )

    env = Environment(gravity=WorldVec3(0.0, -gravity, 0.0))
    kernel = Kernel(rocket=rocket, environment=env)
    sensors = Sensors(kernel)

    throttle = ThrottleActuator(rocket)
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
    scenario = make_scenario(initial_tilt_deg=15.0)

    output_dir = pathlib.Path("/tmp/simulacra")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "demo_tilt_tipover.ndjson"

    with open(output_path, "w") as f:
        scenario.run(duration=3.0, output=f)

    print(f"Wrote {output_path}", file=sys.stderr)
