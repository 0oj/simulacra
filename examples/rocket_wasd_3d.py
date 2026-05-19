"""
Basic 3D rocket demo with nested gimbals controlled by WASD (text mode).

- Uses the real `Rocket` + `Kernel` + `GimbalActuator`.
- W/S change inner pitch (+Y axis of inner gimbal).
- A/D change outer yaw (+Z axis in body frame).
- Thrust is constant; NDJSON output is written to stdout.

Usage (from backend directory):

    # Run and save output to file:
    uv run -m examples.rocket_wasd_3d > output.ndjson

    # Then visualize in another terminal:
    uv run -m examples.plot_wasd_3d output.ndjson

    # Or pipe directly (if controller doesn't need interactive input):
    uv run -m examples.rocket_wasd_3d | uv run -m examples.plot_wasd_3d

Then follow the prompt: type w/a/s/d (optionally multiple chars, e.g. \"wd\"),
or just press ENTER to advance without changing gimbals, or q to quit.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# <INSERT SRC_DIR>
#   We need to add the src directory to
#   the Python path so we can import simulacra.
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(script_dir)
src_dir = os.path.join(backend_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
# </INSERT SRC_DIR>

from simulacra.noumena import (  # type: ignore  # noqa: E402
    Rocket,
    RocketProps,
    RocketState,
    Environment,
    Kernel,
    get_upright_orientation,
    Scenario,
)
from simulacra.phenomena.actuators import (  # type: ignore  # noqa: E402
    GimbalActuator,
)
from simulacra.physics import RocketVec3, WorldVec3  # type: ignore  # noqa: E402
from simulacra.onboard.wasd_controller import WASDController  # type: ignore  # noqa: E402
from simulacra.phenomena.sensors import Sensors  # type: ignore  # noqa: E402

rocket = Rocket(
    props=RocketProps(
        length=6.0,
        mass=120.0,
        center_of_mass_offset=RocketVec3(0.0, 0.0, 0.0),
        inertia_diag=(50.0, 50.0, 10.0),
        thrust_application_offset=RocketVec3(0.0, 0.0, 0.0),
    ),
    state=RocketState(
        position=WorldVec3(0.0, 0.0, 0.0),
        velocity=WorldVec3(0.0, 0.0, 0.0),
        orientation=get_upright_orientation(),
        angular_velocity_body=RocketVec3(0.0, 0.0, 0.0),
        thrust_body=RocketVec3(8000.0, 0.0, 0.0),  # Constant thrust (N) along body +X
    ),
)
# environment = Environment(gravity=WorldVec3(0.0, -9.81, 0.0))
environment = Environment(gravity=WorldVec3(0.0, 0.0, 0.0))
kernel = Kernel(rocket=rocket, environment=environment)

controller = WASDController(
    gimbal_actuator=GimbalActuator(
        rocket=rocket,
        max_outer_angle=np.deg2rad(15.0),
        max_inner_angle=np.deg2rad(15.0),
    )
).with_sensors(Sensors(kernel=kernel))

scenario = Scenario(
    kernel=kernel,
    controller=controller,
    dt=0.01,
)

scenario.run(duration=10.0)
