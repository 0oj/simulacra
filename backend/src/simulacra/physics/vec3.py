"""
Type-safe 3D vector classes with frame separation.

These mirror Vec2/RocketVec2/WorldVec2, but for 3D.
We use them for positions, velocities, forces, etc. in both
world and rocket (body) frames.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Never, Self, overload

import numpy as np

from .quat import Orientation

try:
    from pydantic_core import core_schema
    from pydantic import GetCoreSchemaHandler

    PYDANTIC_AVAILABLE = True
except ImportError:  # pragma: no cover - pydantic is optional
    PYDANTIC_AVAILABLE = False


class Vec3(np.ndarray):
    """
    Base class for 3D vectors with type preservation.

    Subclasses (RocketVec3, WorldVec3) represent vectors in different
    coordinate frames and cannot be mixed without explicit conversion.
    """

    def __new__(cls, x: float, y: float, z: float):
        obj = np.array([x, y, z], dtype=float).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    # --- Frame-safety + type preservation decorators -----------------

    @staticmethod
    def check_frame_compatibility(func):
        """
        Decorator: disallow mixing different Vec3 subclasses
        (e.g. RocketVec3 + WorldVec3) directly.
        """

        @wraps(func)
        def wrapper(self, other):
            if isinstance(other, Vec3) and type(self) is not type(other):
                self_is_sub = type(self) is not Vec3
                other_is_sub = type(other) is not Vec3
                if self_is_sub and other_is_sub:
                    raise TypeError(
                        "Cannot mix RocketVec3 and WorldVec3 directly. "
                        "Convert to the same frame first."
                    )
            return func(self, other)

        return wrapper

    @staticmethod
    def preserve_type(func):
        """
        Decorator: operations like scalar multiply/divide preserve subclass.
        """

        @wraps(func)
        def wrapper(self, other):
            result = func(self, other)
            return type(self)(float(result[0]), float(result[1]), float(result[2]))

        return wrapper

    # --- Arithmetic --------------------------------------------------

    @overload
    def __add__(self: "RocketVec3", other: "RocketVec3") -> "RocketVec3": ...
    @overload
    def __add__(self: "WorldVec3", other: "WorldVec3") -> "WorldVec3": ...
    @overload
    def __add__(self: "RocketVec3", other: "WorldVec3") -> Never: ...
    @overload
    def __add__(self: "WorldVec3", other: "RocketVec3") -> Never: ...

    @check_frame_compatibility
    @preserve_type
    def __add__(self, other) -> Self:
        return super().__add__(other)

    @overload
    def __sub__(self: "RocketVec3", other: "RocketVec3") -> "RocketVec3": ...
    @overload
    def __sub__(self: "WorldVec3", other: "WorldVec3") -> "WorldVec3": ...
    @overload
    def __sub__(self: "RocketVec3", other: "WorldVec3") -> Never: ...
    @overload
    def __sub__(self: "WorldVec3", other: "RocketVec3") -> Never: ...

    @check_frame_compatibility
    @preserve_type
    def __sub__(self, other) -> Self:
        return super().__sub__(other)

    @preserve_type
    def __mul__(self, other) -> Self:
        # vector * scalar
        return super().__mul__(other)

    @preserve_type
    def __rmul__(self, other) -> Self:
        # scalar * vector
        return super().__rmul__(other)

    @preserve_type
    def __truediv__(self, other) -> Self:
        # vector / scalar
        return super().__truediv__(other)

    # --- Convenience properties --------------------------------------

    @property
    def x(self) -> float:
        return float(self[0])

    @x.setter
    def x(self, value: float):
        self[0] = value

    @property
    def y(self) -> float:
        return float(self[1])

    @y.setter
    def y(self, value: float):
        self[1] = value

    @property
    def z(self) -> float:
        return float(self[2])

    @z.setter
    def z(self, value: float):
        self[2] = value

    @property
    def norm(self) -> float:
        return float(np.linalg.norm(self))

    @property
    def direction(self) -> Self:
        """Unit vector in the same direction; zero vector if length == 0."""
        n = self.norm
        if n == 0.0:
            return type(self)(0.0, 0.0, 0.0)
        return type(self)(self[0] / n, self[1] / n, self[2] / n)

    def dot(self, other: "Vec3") -> float:
        return float(np.dot(self, other))

    def cross(self, other: "Vec3") -> "Vec3":
        res = np.cross(self, other)
        # cross-product can mix frames; return plain Vec3
        return Vec3(float(res[0]), float(res[1]), float(res[2]))

    # --- Optional: pydantic v2 schema --------------------------------

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: "GetCoreSchemaHandler",
    ) -> "core_schema.CoreSchema":
        if not PYDANTIC_AVAILABLE:
            return handler(source_type)

        def serialize_vec3(value: "Vec3") -> dict:
            return {"x": float(value.x), "y": float(value.y), "z": float(value.z)}

        def validate_vec3(value: Any) -> "Vec3":
            if isinstance(value, cls):
                return value
            if isinstance(value, dict):
                return cls(value["x"], value["y"], value["z"])
            if isinstance(value, (list, tuple)) and len(value) == 3:
                return cls(float(value[0]), float(value[1]), float(value[2]))
            raise ValueError(f"Cannot convert {type(value)} to {cls.__name__}")

        return core_schema.no_info_plain_validator_function(
            validate_vec3,
            serialization=core_schema.plain_serializer_function_ser_schema(
                serialize_vec3,
            ),
        )


class RocketVec3(Vec3):
    """Vector in rocket *body* frame (RGB axes attached to the rocket)."""

    def to_world(self, orientation: Orientation) -> "WorldVec3":
        """
        Convert this body-frame vector into the world frame
        using the given orientation (body -> world).
        """
        R = orientation.rotation_matrix()  # 3x3
        res = R @ np.asarray(self, dtype=float)
        return WorldVec3(res[0], res[1], res[2])


class WorldVec3(Vec3):
    """Vector in world frame (global RGB axes)."""

    pass
