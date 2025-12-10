"""
Basic 3D rocket demo with nested gimbals controlled by WASD (text mode).

- Uses the real `Rocket3D` + `Kernel3D` + `GimbalActuator3D`.
- W/S change inner pitch (+Y axis of inner gimbal).
- A/D change outer yaw (+Z axis in body frame).
- Thrust is constant; we just print state each step.

Usage (from backend directory):

    uv run -m examples.rocket_wasd_3d

Then follow the prompt: type w/a/s/d (optionally multiple chars, e.g. \"wd\"),
or just press ENTER to advance without changing gimbals, or q to quit.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

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
    Rocket3D,
    RocketProps3D,
    RocketState3D,
    Environment3D,
    Kernel3D,
    get_upright_orientation,
    Scenario3D,
)
from simulacra.phenomena.actuators import (  # type: ignore  # noqa: E402
    GimbalActuator3D,
)
from simulacra.physics import RocketVec3, WorldVec3  # type: ignore  # noqa: E402
from simulacra.onboard.wasd_controller import WASDController  # type: ignore  # noqa: E402
from simulacra.phenomena.sensors import Sensors3D  # type: ignore  # noqa: E402

rocket = Rocket3D(
    props=RocketProps3D(
        length=6.0,
        mass=120.0,
        center_of_mass_offset=RocketVec3(0.0, 0.0, 0.0),
        inertia_diag=(50.0, 50.0, 10.0),
        thrust_application_offset=RocketVec3(0.0, 0.0, 0.0),
    ),
    state=RocketState3D(
        position=WorldVec3(0.0, 0.0, 0.0),
        velocity=WorldVec3(0.0, 0.0, 0.0),
        orientation=get_upright_orientation(),
        angular_velocity_body=RocketVec3(0.0, 0.0, 0.0),
        thrust_magnitude=200.0,
        thrust_direction=RocketVec3(1.0, 0.0, 0.0),
    ),
)
environment = Environment3D(gravity=WorldVec3(0.0, -9.81, 0.0))
kernel = Kernel3D(rocket=rocket, environment=environment)

controller = WASDController(
    gimbal_actuator=GimbalActuator3D(
        rocket=rocket,
        max_outer_angle=np.deg2rad(15.0),
        max_inner_angle=np.deg2rad(15.0),
    )
).with_sensors(Sensors3D(kernel=kernel))

scenario = Scenario3D(
    kernel=kernel,
    controller=controller,
    dt=0.01,
)

scenario.run(duration=10.0)
