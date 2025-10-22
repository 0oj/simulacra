from typing import Tuple
from dataclasses import dataclass, field
import math

from .physics import (
    Vector,
    Gravity,
    WindPattern,
    SteadyWind,
    GustyWind,
    TurbulentWind,
    ChaoticWind,
)


@dataclass
class Frame:
    """Represents a coordinate frame with position and orientation."""

    position: Vector
    orientation: float  # Angle in radians (0 = aligned with world X axis)

    def __init__(self, position: Vector = None, orientation: float = 0.0):
        self.position = position or Vector(x=0, y=0, z=0)
        self.orientation = orientation

    @property
    def rotation_matrix(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Get 2D rotation matrix for this frame."""
        cos_theta = math.cos(self.orientation)
        sin_theta = math.sin(self.orientation)
        return ((cos_theta, -sin_theta), (sin_theta, cos_theta))

    def to_world(self, local_vector: Vector) -> Vector:
        """Transform a vector from this frame to world frame."""
        if local_vector.z != 0:
            # For 3D vectors, only rotate X and Y components
            cos_theta = math.cos(self.orientation)
            sin_theta = math.sin(self.orientation)

            world_x = local_vector.x * cos_theta - local_vector.y * sin_theta
            world_y = local_vector.x * sin_theta + local_vector.y * cos_theta

            return Vector(
                x=world_x + self.position.x,
                y=world_y + self.position.y,
                z=local_vector.z + self.position.z,
            )
        else:
            # 2D case
            cos_theta = math.cos(self.orientation)
            sin_theta = math.sin(self.orientation)

            world_x = local_vector.x * cos_theta - local_vector.y * sin_theta
            world_y = local_vector.x * sin_theta + local_vector.y * cos_theta

            return Vector(
                x=world_x + self.position.x,
                y=world_y + self.position.y,
                z=local_vector.z + self.position.z,
            )

    def to_local(self, world_vector: Vector) -> Vector:
        """Transform a vector from world frame to this frame."""
        # Translate to local origin
        local_x = world_vector.x - self.position.x
        local_y = world_vector.y - self.position.y
        local_z = world_vector.z - self.position.z

        if local_z != 0:
            # For 3D vectors, only rotate X and Y components
            cos_theta = math.cos(-self.orientation)  # Inverse rotation
            sin_theta = math.sin(-self.orientation)

            frame_x = local_x * cos_theta - local_y * sin_theta
            frame_y = local_x * sin_theta + local_y * cos_theta

            return Vector(x=frame_x, y=frame_y, z=local_z)
        else:
            # 2D case
            cos_theta = math.cos(-self.orientation)  # Inverse rotation
            sin_theta = math.sin(-self.orientation)

            frame_x = local_x * cos_theta - local_y * sin_theta
            frame_y = local_x * sin_theta + local_y * cos_theta

            return Vector(x=frame_x, y=frame_y, z=local_z)

    @property
    def x_axis(self) -> Vector:
        """Get the X axis of this frame in world coordinates."""
        return Vector(x=math.cos(self.orientation), y=math.sin(self.orientation), z=0.0)

    @property
    def y_axis(self) -> Vector:
        """Get the Y axis of this frame in world coordinates."""
        return Vector(
            x=-math.sin(self.orientation), y=math.cos(self.orientation), z=0.0
        )


@dataclass
class World:
    current_time: float = 0.0
    gravity: Vector = field(default_factory=lambda: Gravity)
    wind_pattern: WindPattern = field(
        default_factory=lambda: SteadyWind(Vector(x=0, y=0, z=0))
    )
    bounds: Tuple[Vector, Vector] = (Vector(x=0, y=0, z=0), Vector(x=100, y=100, z=100))

    @property
    def current_wind(self) -> Vector:
        """Get current wind vector from the wind pattern."""
        return self.wind_pattern.get_wind(self.current_time)

    def step(self, dt: float) -> None:
        self.current_time += dt


# Preset worlds with different wind patterns
CalmWorld = World()

NormalWorld = World(wind_pattern=SteadyWind(Vector(x=0, y=0, z=0)))

WindyWorld = World(wind_pattern=SteadyWind(Vector(x=1, y=0, z=0)))

GustyWorld = World(
    wind_pattern=GustyWind(
        base_wind=Vector(x=0.2, y=0, z=0), gust_strength=2.0, gust_frequency=0.02
    )
)

TurbulentWorld = World(
    wind_pattern=TurbulentWind(
        base_wind=Vector(x=0, y=0, z=0), intensity=1.5, frequency=0.3
    )
)

ChaoticWorld = World(
    wind_pattern=ChaoticWind(base_wind=Vector(x=0, y=0, z=0), intensity=2.0)
)
