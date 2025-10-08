import math
from dataclasses import dataclass


@dataclass
class Vector:
    x: float
    y: float

    @property
    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2)

    @property
    def direction(self) -> float:
        return math.atan2(self.y, self.x)

    def __add__(self, other: "Vector") -> "Vector":
        return Vector(x=self.x + other.x, y=self.y + other.y)

    def __mul__(self, other: float) -> "Vector":
        if not isinstance(other, float):
            raise ValueError("Cannot multiply by non-float")
        return Vector(x=self.x * other, y=self.y * other)

    def __truediv__(self, other: float) -> "Vector":
        if not isinstance(other, float):
            raise ValueError(f"Cannot divide by non-float: {type(other)}")
        if other == 0:
            raise ValueError("Cannot divide by zero")
        return Vector(x=self.x / other, y=self.y / other)

    def __neg__(self) -> "Vector":
        return Vector(x=-self.x, y=-self.y)

    def __abs__(self) -> float:
        return self.magnitude


Gravity = Vector(x=0, y=-9.81)
