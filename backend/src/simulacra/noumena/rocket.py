"""
The rocket as god sees it. True position, true velocity, true orientation.
The rocket itself never sees this; it only sees what the sensors reveal.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..physics import (
    Orientation,
    RocketVec2,
    RocketVec3,
    WorldVec2,
    WorldVec3,
)


# ============================================================================
# 2D rocket (legacy)
# ============================================================================


@dataclass(frozen=True)
class RocketProps2D:
    length: float  # meters, length of the rocket
    mass: float  # kilograms, mass of the rocket
    center_of_mass_offset: RocketVec2  # meters, offset from the base of the rocket
    moment_of_inertia: float  # kg*m^2, moment of inertia about the center of mass
    thrust_application_offset: RocketVec2  # meters, point where the thrust is applied


@dataclass
class RocketState2D:
    position: WorldVec2  # meters, position of the rocket in the world frame
    velocity: WorldVec2  # meters/second, velocity of the rocket in the world frame
    orientation: float  # radians, orientation of the rocket in the world frame
    angular_velocity: (
        float  # radians/second, angular velocity of the rocket in the world frame
    )
    gimbal_angle: float  # radians, angle of the gimbal in the rocket frame
    thrust_magnitude: float  # Newtons, magnitude of the thrust

    @property
    def thrust_vector(self) -> RocketVec2:
        return RocketVec2(
            self.thrust_magnitude * np.cos(self.gimbal_angle),
            self.thrust_magnitude * np.sin(self.gimbal_angle),
        )


@dataclass
class Rocket2D:
    props: RocketProps2D
    state: RocketState2D


# ============================================================================
# 3D rocket (work in progress)
# ============================================================================


@dataclass(frozen=True)
class RocketProps3D:
    """
    True physical properties of the rocket in 3D, as god sees them.

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


@dataclass
class RocketState3D:
    """
    True rocket state in 3D, as god sees it.

    Position and velocity live in the world frame.
    Orientation maps body -> world and is stored as a quaternion.
    Angular velocity is body-frame [wx, wy, wz] (rad/s).
    """

    position: WorldVec3  # meters, world frame
    velocity: WorldVec3  # m/s, world frame
    orientation: Orientation  # body -> world
    angular_velocity_body: RocketVec3  # rad/s, body frame [wx, wy, wz]
    thrust_magnitude: float  # Newtons
    thrust_direction: RocketVec3  # Newtons, body-frame thrust direction


@dataclass
class Rocket3D:
    props: RocketProps3D
    state: RocketState3D


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
