"""
Demo: Sensor Drift

GPS sensor with random walk drift - position estimate slowly diverges from truth.
Demonstrates: How sensor biases accumulate over time.

The controller uses drifting GPS, so the rocket slowly drifts from where
it thinks it is to where it actually is.

Setup:
- Hovering rocket with TVC control
- GPS has random walk drift (accumulating bias)
- Controller tries to hold position based on drifting GPS

Expected behavior:
- Initially holds position well
- Over time, the GPS reading drifts from truth
- Rocket physically drifts to compensate for "perceived" position error
- Watch the GPS error chart grow over time

Usage:
    uv run scenarios/demos/sensor_drift.py
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
)
from simulacra.phenomena.sensors import Sensors
from simulacra.phenomena.errors import RandomWalk, CompositeError, GaussianNoise
from simulacra.physics import RocketVec3, WorldVec3
from simulacra.onboard.hover_controller import HoverController, HoverControllerConfig


def make_scenario() -> Scenario:
    mass = 100.0
    gravity = 9.81
    weight = mass * gravity
    max_thrust = 2000.0
    com_offset = 2.0

    rocket = Rocket(
        props=RocketProps(
            length=4.0,
            mass=mass,
            center_of_mass_offset=RocketVec3(com_offset, 0.0, 0.0),
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

    # GPS with random walk drift on altitude
    # sigma=0.3 means drift accumulates at 0.3 m/sqrt(s)
    sensors = Sensors(
        kernel,
        gps_errors={
            "y": CompositeError([
                GaussianNoise(std=0.1, seed=42),  # Small measurement noise
                RandomWalk(sigma=0.3, seed=43),  # Accumulating drift
            ]),
        },
    )

    throttle = ThrottleActuator(rocket)
    gimbal = GimbalActuator(
        rocket,
        max_outer_angle=math.radians(15.0),
        max_inner_angle=math.radians(15.0),
    )
    actuators = Actuators(throttle=throttle, gimbal=gimbal)

    controller = HoverController(
        throttle=throttle,
        gimbal=gimbal,
        config=HoverControllerConfig(
            target_alt=10.0,
            weight=weight,
            max_thrust=max_thrust,
            gravity=gravity,
            att_Kp=500.0,
            att_Kd=200.0,
            thrust_moment_arm=com_offset,
        ),
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
    output_path = output_dir / "demo_sensor_drift.ndjson"

    with open(output_path, "w") as f:
        scenario.run(duration=20.0, output=f)

    print(f"Wrote {output_path}", file=sys.stderr)
