"""
The mathematical foundations. Coordinate frames and vectors,
separated by type so the rocket cannot confuse its own frame with the world's.
"""

from .vec3 import RocketVec3, WorldVec3
from .quat import Orientation
from .inertia import total_mass, center_of_mass, inertia_diag

__all__ = [
    "RocketVec3",
    "WorldVec3",
    "Orientation",
    "total_mass",
    "center_of_mass",
    "inertia_diag",
]
