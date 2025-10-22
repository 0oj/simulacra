import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Vector:
    x: float
    y: float
    z: float = 0.0  # Default to 0 for 2D compatibility

    @property
    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    @property
    def direction(self) -> float:
        """2D direction (for backward compatibility). Returns angle in XY plane."""
        return math.atan2(self.y, self.x)

    @property
    def elevation(self) -> float:
        """3D elevation angle from XY plane to vector."""
        if self.magnitude == 0:
            return 0.0
        return math.asin(self.z / self.magnitude)

    @property
    def azimuth(self) -> float:
        """3D azimuth angle in XY plane."""
        return math.atan2(self.y, self.x)

    def __add__(self, other: "Vector") -> "Vector":
        return Vector(x=self.x + other.x, y=self.y + other.y, z=self.z + other.z)

    def __mul__(self, other: float) -> "Vector":
        if not isinstance(other, float):
            raise ValueError("Cannot multiply by non-float")
        return Vector(x=self.x * other, y=self.y * other, z=self.z * other)

    def __truediv__(self, other: float) -> "Vector":
        if not isinstance(other, float):
            raise ValueError(f"Cannot divide by non-float: {type(other)}")
        if other == 0:
            raise ValueError("Cannot divide by zero")
        return Vector(x=self.x / other, y=self.y / other, z=self.z / other)

    def __neg__(self) -> "Vector":
        return Vector(x=-self.x, y=-self.y, z=-self.z)

    def __abs__(self) -> float:
        return self.magnitude

    def cross(self, other: "Vector") -> "Vector":
        """Cross product for 3D vectors."""
        return Vector(
            x=self.y * other.z - self.z * other.y,
            y=self.z * other.x - self.x * other.z,
            z=self.x * other.y - self.y * other.x,
        )

    def dot(self, other: "Vector") -> float:
        """Dot product for 3D vectors."""
        return self.x * other.x + self.y * other.y + self.z * other.z


class WindPattern(ABC):
    """Abstract base class for wind patterns."""

    @abstractmethod
    def get_wind(self, time: float) -> Vector:
        """Get wind vector at given time."""
        pass


class SteadyWind(WindPattern):
    """Constant wind pattern."""

    def __init__(self, wind_vector: Vector):
        self.wind_vector = wind_vector

    def get_wind(self, time: float) -> Vector:
        return self.wind_vector


class GustyWind(WindPattern):
    """Wind with periodic random gusts."""

    def __init__(
        self,
        base_wind: Vector,
        gust_strength: float = 2.0,
        gust_frequency: float = 0.5,
    ):
        self.base_wind = base_wind
        self.gust_strength = gust_strength
        self.gust_frequency = gust_frequency

    def get_wind(self, time: float) -> Vector:
        # Create periodic gusts using sine wave + random component
        gust_phase = time * self.gust_frequency * 2 * math.pi
        gust_magnitude = self.gust_strength * (0.5 + 0.5 * math.sin(gust_phase))

        # Add random direction component in 3D
        gust_azimuth = random.uniform(0, 2 * math.pi)
        gust_elevation = random.uniform(
            -math.pi / 4, math.pi / 4
        )  # Limit vertical component

        gust_x = gust_magnitude * math.cos(gust_elevation) * math.cos(gust_azimuth)
        gust_y = gust_magnitude * math.cos(gust_elevation) * math.sin(gust_azimuth)
        gust_z = gust_magnitude * math.sin(gust_elevation)

        return self.base_wind + Vector(gust_x, gust_y, gust_z)


class TurbulentWind(WindPattern):
    """Smooth turbulent wind using Perlin-like noise."""

    def __init__(
        self, base_wind: Vector, intensity: float = 1.0, frequency: float = 0.3
    ):
        self.base_wind = base_wind
        self.intensity = intensity
        self.frequency = frequency

    def _smooth_noise(self, t: float) -> float:
        """Simple 1D Perlin-like noise for smooth turbulence."""
        # Use multiple octaves for realistic turbulence
        noise = 0.0
        amplitude = 1.0
        freq = self.frequency

        for _ in range(3):  # 3 octaves
            noise += amplitude * math.sin(t * freq + random.uniform(0, 2 * math.pi))
            amplitude *= 0.5
            freq *= 2.0

        return noise * self.intensity

    def get_wind(self, time: float) -> Vector:
        # Generate smooth turbulence in x, y, and z
        turb_x = self._smooth_noise(time)
        turb_y = self._smooth_noise(time + 1000)  # Offset for different pattern
        turb_z = self._smooth_noise(time + 2000)  # Offset for different pattern

        return self.base_wind + Vector(turb_x, turb_y, turb_z)


class ChaoticWind(WindPattern):
    """Combines gusts, direction changes, and turbulence."""

    def __init__(self, base_wind: Vector, intensity: float = 2.0):
        self.base_wind = base_wind
        self.intensity = intensity
        self.turbulent = TurbulentWind(base_wind, intensity * 0.5, 0.2)
        self.gusty = GustyWind(Vector(0, 0), intensity * 1.5, 0.8)

    def get_wind(self, time: float) -> Vector:
        # Combine turbulent and gusty patterns
        turb_wind = self.turbulent.get_wind(time)
        gust_wind = self.gusty.get_wind(time)

        # Add occasional direction changes
        if random.random() < 0.01:  # 1% chance of direction change
            direction_change = Vector(
                random.uniform(-self.intensity, self.intensity),
                random.uniform(-self.intensity, self.intensity),
                random.uniform(
                    -self.intensity * 0.5, self.intensity * 0.5
                ),  # Less vertical change
            )
            return turb_wind + gust_wind + direction_change

        return turb_wind + gust_wind


Gravity = Vector(x=0, y=-9.81, z=0)
