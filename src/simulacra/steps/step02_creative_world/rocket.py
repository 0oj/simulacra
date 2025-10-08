from dataclasses import dataclass, field
from random import random
import time

from .physics import Vector


@dataclass
class Rocket:
    mass: float

    position: Vector
    velocity: Vector

    drag_factor: float = 0.1  # More explicit: acceleration per velocity unit

    def apply(self, force: Vector) -> None:
        self.velocity += force / self.mass

    @property
    def drag(self) -> Vector:
        return -self.velocity * self.drag_factor * self.mass

    @property
    def thrust(self) -> Vector:
        if time.time() % 2 < 1:
            return Vector(x=0, y=7.81)
        return Vector(x=(random() - 0.5) * 4, y=11.81)

    def step(self, dt: float) -> None:
        self.position += self.velocity * dt
        self.apply(self.drag)
        self.apply(self.thrust)
