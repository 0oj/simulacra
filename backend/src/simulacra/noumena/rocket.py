"""
The rocket as god sees it. True position, true velocity, true orientation.
The rocket itself never sees this; it only sees what the sensors reveal.
"""
from dataclasses import dataclass

import numpy as np

from ..physics import RocketVec2, WorldVec2


@dataclass(frozen=True)
class RocketProperties:
    length: float  # meters, length of the rocket
    mass: float  # kilograms, mass of the rocket
    center_of_mass_offset: RocketVec2  # meters, offset from the base of the rocket
    moment_of_inertia: float  # kg*m^2, moment of inertia about the center of mass
    thrust_application_offset: RocketVec2  # meters, point where the thrust is applied


@dataclass
class RocketState:
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
class Rocket:
    properties: RocketProperties
    state: RocketState
