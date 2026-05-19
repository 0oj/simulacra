"""
Demo: Noisy Hover

Same hover scenario but with realistic sensor noise and thrust jitter.
Demonstrates: How noise propagates through the control loop.

Compare with demo_hover_balance to see the effect of noise.

Noise sources:
- GPS: 0.5m std noise on altitude
- Velocity: 0.2 m/s std noise
- Gyro: 1°/s std noise
- Thrust alignment: 0.5° std jitter

Expected behavior:
- Altitude oscillates around target (can't hold perfectly still)
- Small attitude wobble from thrust jitter
- Controller fights the noise

Usage:
    uv run scenarios/demos/noisy_hover.py
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
from simulacra.phenomena.errors import GaussianNoise
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

    # Noisy sensors
    sensors = Sensors(
        kernel,
        gps_errors={
            "x": GaussianNoise(std=0.3, seed=42),
            "y": GaussianNoise(std=0.5, seed=43),  # Altitude noise
            "z": GaussianNoise(std=0.3, seed=44),
        },
        velocity_errors={
            "x": GaussianNoise(std=0.1, seed=45),
            "y": GaussianNoise(std=0.2, seed=46),
            "z": GaussianNoise(std=0.1, seed=47),
        },
        gyro_errors={
            "x": GaussianNoise(std=math.radians(1.0), seed=48),
            "y": GaussianNoise(std=math.radians(1.0), seed=49),
            "z": GaussianNoise(std=math.radians(1.0), seed=50),
        },
        orientation_errors={
            "roll": GaussianNoise(std=math.radians(0.5), seed=51),
            "pitch": GaussianNoise(std=math.radians(0.5), seed=52),
            "yaw": GaussianNoise(std=math.radians(0.5), seed=53),
        },
    )

    # Noisy thrust (alignment jitter)
    throttle = ThrottleActuator(
        rocket,
        alignment_errors={
            "yaw": GaussianNoise(std=math.radians(0.5), seed=54),
            "pitch": GaussianNoise(std=math.radians(0.5), seed=55),
        }
    )
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
    output_path = output_dir / "demo_noisy_hover.ndjson"

    with open(output_path, "w") as f:
        scenario.run(duration=10.0, output=f)

    print(f"Wrote {output_path}", file=sys.stderr)
