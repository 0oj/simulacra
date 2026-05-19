"""
The rocket as god sees it. True position, true velocity, true orientation.
The rocket itself never sees this; it only sees what the sensors reveal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from ..physics import (
    Orientation,
    RocketVec3,
    WorldVec3,
)

from ..noumena.mass_model import MassModel


@dataclass(frozen=True)
class RocketProps:
    """
    True physical properties of the rocket, as god sees them.

    For now we use a simple diagonal inertia tensor specified by (Ix, Iy, Iz)
    in the rocket *body* frame. ie. Assuming the rocket is symmetric about the X, Y, and Z axes.
    """

    length: float  # meters, length of the rocket
    mass: float  # kilograms, mass of the rocket
    center_of_mass_offset: RocketVec3  # meters, offset from the base of the rocket
    # Diagonal inertia components about the center of mass, body frame.
    # Ix: about X (roll), Iy: about Y (pitch), Iz: about Z (yaw)
    inertia_diag: Tuple[float, float, float]  # kg*m^2
    thrust_application_offset: RocketVec3  # meters, point where the thrust is applied
    drag_coefficient: float = 0.5      # Cd, dimensionless (~0.4-0.5 for slender body, subsonic)
    reference_area: float = 0.031416   # m², frontal area (π × 0.1² for ~20cm diameter)


@dataclass
class RocketState:
    """
    True rocket state, as god sees it.

    Position and velocity live in the world frame.
    Orientation maps body -> world and is stored as a quaternion.
    Angular velocity is body-frame [wx, wy, wz] (rad/s).
    """

    position: WorldVec3  # meters, world frame
    velocity: WorldVec3  # m/s, world frame
    orientation: Orientation  # body -> world
    angular_velocity_body: RocketVec3  # rad/s, body frame [wx, wy, wz]
    thrust_body: RocketVec3  # Newtons, body-frame thrust vector (direction × magnitude)


def _default_mass_model():
    return MassModel.constant(1.0, RocketVec3(0.0, 0.0, 0.0))


@dataclass
class Rocket:
    props: RocketProps
    state: RocketState
    mass_model: "MassModel" = field(default_factory=_default_mass_model)


def get_upright_orientation() -> Orientation:
    """
    Default 'upright on the pad' orientation.

    With our conventions:
    - Body +X is the rocket's nose.
    - World +Y is 'up' (opposite gravity).

    Identity orientation maps body +X -> world +X (rocket lying sideways).
    Upright orientation is a +90 degree rotation about +Z so that:
        body +X  ->  world +Y
    """
    angle = float(np.pi / 2.0)
    return Orientation.from_axis_angle([0.0, 0.0, 1.0], angle)
