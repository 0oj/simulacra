from typing import Tuple
from dataclasses import dataclass, field

from .physics import Vector, Gravity


@dataclass
class World:
    current_time: float = 0.0
    gravity: Vector = field(default_factory=lambda: Gravity)
    wind: Vector = field(default_factory=lambda: Vector(x=0, y=0))
    bounds: Tuple[Vector, Vector] = (Vector(x=0, y=0), Vector(x=100, y=100))

    def step(self, dt: float) -> None:
        self.current_time += dt


NormalWorld = World()

WindyWorld = World(
    wind=Vector(x=1, y=0),
)
