"""
The world as god defines it. Gravity, wind, air density.
These are the laws that govern the rocket's motion.
"""

from dataclasses import dataclass, field

from ..physics import WorldVec3


@dataclass(frozen=True)
class Environment:
    """
    Environment for the rocket.

    - World +Y is 'up' (opposite gravity).
    - Gravity points along -Y.
    """

    gravity: WorldVec3 = field(
        default_factory=lambda: WorldVec3(0.0, -9.81, 0.0)
    )  # m/s^2, downwards
    wind: WorldVec3 = field(
        default_factory=lambda: WorldVec3(0.0, 0.0, 0.0)
    )  # m/s, wind velocity
    air_density: float = 1.225  # kg/m^3, air density (standard sea level)
