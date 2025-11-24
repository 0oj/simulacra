"""
The mathematical foundations. Coordinate frames and vectors,
separated by type so the rocket cannot confuse its own frame with the world's.
"""
from .vec2 import RocketVec2, WorldVec2

__all__ = ["RocketVec2", "WorldVec2"]
