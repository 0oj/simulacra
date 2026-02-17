"""
Demo: TVC Recovery

Rocket starts with a large tilt (30°) and uses TVC to recover to vertical.
Demonstrates: Closed-loop attitude control, gravity feedforward.

This is the TVC controller in action - showing how gimbal deflection
creates corrective torque to fight gravity and stabilize attitude.

Expected behavior:
- Initial 30° tilt
- Gimbal deflects to create counter-torque
- Rocket recovers to vertical in ~1 second
- Settles at target altitude

Usage:
    uv run scenarios/demos/tvc_recovery.py
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
from simulacra.phenomena.actuators import (
    ThrottleActuator,
    GimbalActuator,
    Actuators,
)
from simulacra.phenomena.sensors import Sensors
from simulacra.physics import RocketVec3, WorldVec3, Orientation
from simulacra.onboard.hover_controller import HoverController, HoverControllerConfig


def make_scenario(initial_tilt_deg: float = 30.0) -> Scenario:
    mass = 100.0
    gravity = 9.81
    weight = mass * gravity
    max_thrust = 2000.0
    com_offset = 2.0

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
            center_of_mass_offset=RocketVec3(com_offset, 0.0, 0.0),
            inertia_diag=(50.0, 50.0, 10.0),
            thrust_application_offset=RocketVec3(0.0, 0.0, 0.0),
        ),
        state=RocketState(
            position=WorldVec3(0.0, 20.0, 0.0),  # Start at 20m
            velocity=WorldVec3(0.0, 0.0, 0.0),
            orientation=orientation,
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

    controller = HoverController(
        throttle=throttle,
        gimbal=gimbal,
        config=HoverControllerConfig(
            target_alt=20.0,
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
    scenario = make_scenario(initial_tilt_deg=30.0)

    output_dir = pathlib.Path("/tmp/simulacra")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "demo_tvc_recovery.ndjson"

    with open(output_path, "w") as f:
        scenario.run(duration=5.0, output=f)

    print(f"Wrote {output_path}", file=sys.stderr)
