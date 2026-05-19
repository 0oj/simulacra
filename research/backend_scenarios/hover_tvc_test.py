"""
Full hover test with TVC (Thrust Vector Control).

Uses the HoverController which has both altitude and attitude control:
- Altitude PD → thrust magnitude
- Quaternion PD → gimbal angles (TVC)

This serves as a baseline for RL comparison.

Usage:
    uv run scenarios/hover_tvc_test.py
    uv run scenarios/hover_tvc_test.py --noisy
    uv run scenarios/hover_tvc_test.py --disturbed  # with thrust misalignment
"""

import sys
import os
import math
import pathlib

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(script_dir)
src_dir = os.path.join(backend_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import numpy as np

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
from simulacra.phenomena.errors import GaussianNoise, Bias, CompositeError
from simulacra.physics import RocketVec3, WorldVec3
from simulacra.onboard.hover_controller import HoverController, HoverControllerConfig


def make_scenario(
    noisy: bool = False,
    disturbed: bool = False,
    initial_tilt_deg: float = 0.0,
) -> Scenario:
    """
    Create a hover scenario with full TVC control.

    Args:
        noisy: Add sensor noise
        disturbed: Add thrust misalignment (tests controller robustness)
        initial_tilt_deg: Start with rocket tilted (tests attitude recovery)
    """
    mass = 100.0  # kg
    gravity = 9.81  # m/s^2
    weight = mass * gravity
    max_thrust = 2000.0  # N
    rocket_length = 4.0  # m
    com_offset = 2.0  # m from base

    # Initial orientation: upright, possibly with tilt
    from simulacra.physics import Orientation
    from scipy.spatial.transform import Rotation as R
    if initial_tilt_deg != 0.0:
        # Tilt about world Z axis (sideways lean)
        upright = get_upright_orientation()
        # Compose rotations using scipy
        r_upright = upright._as_scipy_rotation()
        r_tilt = R.from_rotvec(math.radians(initial_tilt_deg) * np.array([0, 0, 1]))
        r_combined = r_tilt * r_upright  # Apply tilt in world frame first
        x, y, z, w = r_combined.as_quat()
        orientation = Orientation.from_wxyz(w, x, y, z)
    else:
        orientation = get_upright_orientation()

    rocket = Rocket(
        props=RocketProps(
            length=rocket_length,
            mass=mass,
            center_of_mass_offset=RocketVec3(com_offset, 0.0, 0.0),
            inertia_diag=(50.0, 50.0, 10.0),
            thrust_application_offset=RocketVec3(0.0, 0.0, 0.0),
        ),
        state=RocketState(
            position=WorldVec3(0.0, 5.0, 0.0),  # Start at 5m
            velocity=WorldVec3(0.0, 0.0, 0.0),
            orientation=orientation,
            angular_velocity_body=RocketVec3(0.0, 0.0, 0.0),
            thrust_body=RocketVec3(weight, 0.0, 0.0),  # Start at hover thrust
        ),
    )

    env = Environment(gravity=WorldVec3(0.0, -gravity, 0.0))
    kernel = Kernel(rocket=rocket, environment=env)

    # Sensors
    if noisy:
        sensors = Sensors(
            kernel,
            gps_errors={"y": GaussianNoise(std=0.5, seed=42)},
            velocity_errors={"y": GaussianNoise(std=0.2, seed=43)},
            orientation_errors={
                "roll": GaussianNoise(std=math.radians(0.5), seed=46),
                "pitch": GaussianNoise(std=math.radians(0.5), seed=47),
                "yaw": GaussianNoise(std=math.radians(0.5), seed=48),
            },
            gyro_errors={
                "x": GaussianNoise(std=math.radians(1.0), seed=49),
                "y": GaussianNoise(std=math.radians(1.0), seed=50),
                "z": GaussianNoise(std=math.radians(1.0), seed=51),
            },
        )
    else:
        sensors = Sensors(kernel)

    # Actuators
    alignment_errors = None
    if disturbed:
        # Thrust misalignment: constant bias + noise
        alignment_errors = {
            "yaw": CompositeError([
                Bias(offset=math.radians(0.5)),  # 0.5 deg bias
                GaussianNoise(std=math.radians(0.2), seed=44),
            ]),
            "pitch": CompositeError([
                Bias(offset=math.radians(-0.3)),
                GaussianNoise(std=math.radians(0.2), seed=45),
            ]),
        }

    throttle = ThrottleActuator(rocket, alignment_errors=alignment_errors)
    gimbal = GimbalActuator(
        rocket,
        max_outer_angle=math.radians(15.0),
        max_inner_angle=math.radians(15.0),
        # Fast servos for testing - no rate limit or lag
        # max_rate=math.radians(60.0),  # 60 deg/s servo rate
        # tau=0.05,  # 50ms servo lag
    )
    actuators = Actuators(throttle=throttle, gimbal=gimbal)

    # Controller with TVC
    # With gravity feedforward, feedback gains handle transient errors only
    controller = HoverController(
        throttle=throttle,
        gimbal=gimbal,
        config=HoverControllerConfig(
            target_alt=10.0,
            weight=weight,
            max_thrust=max_thrust,
            gravity=gravity,
            alt_Kp=100.0,
            alt_Kd=200.0,
            att_Kp=500.0,   # Feedback for transient errors
            att_Kd=200.0,   # Angular velocity damping
            max_gimbal_angle=math.radians(15.0),
            thrust_moment_arm=com_offset,  # CoM to thrust point
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
    noisy = "--noisy" in sys.argv
    disturbed = "--disturbed" in sys.argv
    tilted = "--tilted" in sys.argv

    # Determine output filename and initial tilt
    suffix = "clean"
    initial_tilt = 0.0
    if tilted:
        suffix = "tilted"
        initial_tilt = 15.0  # 15 degree initial tilt
    elif disturbed:
        suffix = "disturbed"
    elif noisy:
        suffix = "noisy"

    scenario = make_scenario(noisy=noisy, disturbed=disturbed, initial_tilt_deg=initial_tilt)

    output_dir = pathlib.Path("/tmp/simulacra")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"hover_tvc_{suffix}.ndjson"

    with open(output_path, "w") as f:
        scenario.run(duration=10.0, output=f)

    print(f"Wrote {output_path}", file=sys.stderr)
