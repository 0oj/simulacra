"""
Type-safe 2D vector classes with frame separation.

This module provides Vec2, RocketVec2, and WorldVec2 classes that:
- Preserve type information through arithmetic operations
- Prevent mixing vectors from different coordinate frames
- Provide methods for rotation and frame conversion
"""

import numpy as np
from functools import wraps
from typing import Self, overload, Never, Any

try:
    from pydantic_core import core_schema
    from pydantic import GetCoreSchemaHandler
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False


class Vec2(np.ndarray):
    """
    Base class for 2D vectors with type preservation.
    
    Subclasses (RocketVec2, WorldVec2) represent vectors in different
    coordinate frames and cannot be mixed without explicit conversion.
    """
    
    def __new__(cls, x: float, y: float):
        obj = np.array([x, y], dtype=float).view(cls)
        return obj

    def __array_finalize__(self, obj):
        """Called after array creation to preserve subclass type."""
        if obj is None:
            return

    @staticmethod
    def check_frame_compatibility(func):
        """
        Decorator that checks frame compatibility for binary operations.
        Raises TypeError if both operands are different Vec2 subclasses (not base Vec2).
        """
        @wraps(func)
        def wrapper(self, other):
            if isinstance(other, Vec2) and type(self) is not type(other):
                self_is_subclass = type(self) is not Vec2
                other_is_subclass = type(other) is not Vec2
                if self_is_subclass and other_is_subclass:
                    raise TypeError(
                        "Cannot mix RocketVec2 and WorldVec2. "
                        "Convert to the same frame first using rotate() or explicit conversion."
                    )
            return func(self, other)
        return wrapper

    @staticmethod
    def preserve_type(func):
        """
        Decorator that preserves the Vec2 subclass type through operations (eg. scalar multiplication)
        """
        @wraps(func)
        def wrapper(self, other):
            result = func(self, other)
            return type(self)(result[0], result[1])
        return wrapper

    # Overloads for __add__ - only frame-compatible combinations allowed
    @overload
    def __add__(self: 'RocketVec2', other: 'RocketVec2') -> 'RocketVec2': ...
    
    @overload
    def __add__(self: 'WorldVec2', other: 'WorldVec2') -> 'WorldVec2': ...
    
    # Explicitly reject invalid frame mixing combinations
    @overload
    def __add__(self: 'RocketVec2', other: 'WorldVec2') -> Never: ...
    
    @overload
    def __add__(self: 'WorldVec2', other: 'RocketVec2') -> Never: ...
    
    @check_frame_compatibility
    @preserve_type
    def __add__(self, other) -> Self:
        """Override addition to preserve type and check frame compatibility."""
        return super().__add__(other)

    # Overloads for __sub__ - only frame-compatible combinations allowed
    @overload
    def __sub__(self: 'RocketVec2', other: 'RocketVec2') -> 'RocketVec2': ...
    
    @overload
    def __sub__(self: 'WorldVec2', other: 'WorldVec2') -> 'WorldVec2': ...
    
    # Explicitly reject invalid frame mixing combinations
    @overload
    def __sub__(self: 'RocketVec2', other: 'WorldVec2') -> Never: ...
    
    @overload
    def __sub__(self: 'WorldVec2', other: 'RocketVec2') -> Never: ...
    
    @check_frame_compatibility
    @preserve_type
    def __sub__(self, other) -> Self:
        """Override subtraction to preserve type and check frame compatibility."""
        return super().__sub__(other)

    @preserve_type
    def __mul__(self, other) -> Self:
        """Override multiplication (vector * scalar) to preserve type."""
        return super().__mul__(other)

    @preserve_type
    def __rmul__(self, other) -> Self:
        """Override right multiplication (scalar * vector) to preserve type."""
        return super().__rmul__(other)

    @preserve_type
    def __truediv__(self, other) -> Self:
        """Override division (vector / scalar) to preserve type."""
        return super().__truediv__(other)

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
    def direction(self) -> Self:
        """Returns normalized direction vector, preserving type."""
        norm = np.linalg.norm(self)
        if norm == 0:
            return type(self)(0.0, 0.0)
        return type(self)(self[0] / norm, self[1] / norm)

    def rotate(self, theta: float) -> Self:
        """
        Rotates this vector in 2D by angle theta (radians), counterclockwise.
        Preserves the input type (RocketVec2 or WorldVec2).
        
        Args:
            theta: float - angle in radians (CCW)
        Returns:
            Vec2 - rotated vector (same type as self)
        """
        c = np.cos(theta)
        s = np.sin(theta)
        rot = np.array([[c, -s],
                        [s,  c]])
        result = rot @ self
        return type(self)(result[0], result[1])

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """
        Pydantic v2 integration: serialize Vec2 as {"x": float, "y": float} dict.
        """
        if not PYDANTIC_AVAILABLE:
            return handler(source_type)
        
        def serialize_vec2(value: Vec2) -> dict:
            """Serialize Vec2 to dict."""
            return {"x": float(value.x), "y": float(value.y)}
        
        def validate_vec2(value: Any) -> Vec2:
            """Validate and convert dict/list to Vec2."""
            if isinstance(value, cls):
                return value
            if isinstance(value, dict):
                return cls(value["x"], value["y"])
            if isinstance(value, (list, tuple)) and len(value) == 2:
                return cls(float(value[0]), float(value[1]))
            raise ValueError(f"Cannot convert {type(value)} to {cls.__name__}")
        
        return core_schema.no_info_plain_validator_function(
            validate_vec2,
            serialization=core_schema.plain_serializer_function_ser_schema(
                serialize_vec2,
            ),
        )


class RocketVec2(Vec2):
    """Vector in rocket frame (tail-to-nose along X axis)."""
    
    def to_world(self, rocket_angle: float) -> 'WorldVec2':
        """
        Converts this rocket frame vector to world frame by rotating it.
        
        Args:
            rocket_angle: float - rocket orientation angle in radians (CCW from world X)
        
        Returns:
            WorldVec2 - vector in world frame
        """
        rotated = self.rotate(rocket_angle)
        return WorldVec2(rotated[0], rotated[1])


class WorldVec2(Vec2):
    """Vector in world frame (rightwards X, upwards Y)."""
    pass

