"""
Minimal quaternion-based orientation helper built on top of SciPy + NumPy.

Conventions
-----------
- We store quaternions as np.ndarray([w, x, y, z]) representing
  w*1 + x*i + y*j + z*k.
- i, j, k are aligned with the X (red), Y (green), Z (blue) axes.
- An Orientation always represents a rotation mapping body-frame
  vectors -> world-frame vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    from pydantic_core import core_schema
    from pydantic import GetCoreSchemaHandler

    PYDANTIC_AVAILABLE = True
except ImportError:  # pragma: no cover - pydantic is optional
    PYDANTIC_AVAILABLE = False


@dataclass
class Orientation:
    """
    Rocket orientation (body -> world), stored as a unit quaternion.

    This is a thin semantic wrapper around a quaternion stored as a
    NumPy array [w, x, y, z].

    - For math/state packing, use `as_array()`.
    - For rocket semantics, use the methods here (e.g. rotation_matrix).
    """

    q: np.ndarray  # shape (4,), [w, x, y, z]

    def __post_init__(self) -> None:
        """
        Ensure q is a 1D float array of length 4.
        """
        arr = np.asarray(self.q, dtype=float).reshape(4)
        self.q = arr

    # ---------- constructors ----------

    @staticmethod
    def identity() -> "Orientation":
        """No rotation: body axes aligned with world axes."""
        return Orientation(np.array([1.0, 0.0, 0.0, 0.0], dtype=float))

    @classmethod
    def from_wxyz(cls, w: float, x: float, y: float, z: float) -> "Orientation":
        """Build from scalar/vector parts (w, x, y, z) and normalize."""
        return cls(np.array([w, x, y, z], dtype=float)).normalized()

    @classmethod
    def from_array(cls, arr: Iterable[float]) -> "Orientation":
        """Build from array-like [w, x, y, z] and normalize."""
        w, x, y, z = map(float, arr)
        return cls(np.array([w, x, y, z], dtype=float)).normalized()

    @classmethod
    def from_axis_angle(cls, axis: Iterable[float], angle: float) -> "Orientation":
        """
        Build from an axis (X/red, Y/green, Z/blue) and rotation angle.

        Args:
            axis: Iterable[float] of length 3, rotation axis in body frame.
            angle: Rotation angle in radians, right-hand convention.
        """
        axis_arr = np.asarray(axis, dtype=float)
        n = float(np.linalg.norm(axis_arr))
        if n == 0.0:
            return cls.identity()
        axis_unit = axis_arr / n
        r = R.from_rotvec(angle * axis_unit)
        # SciPy returns [x, y, z, w]
        x, y, z, w = r.as_quat()
        return cls.from_wxyz(w, x, y, z)

    # ---------- representation helpers ----------

    def as_array(self) -> np.ndarray:
        """
        Return np.array([w, x, y, z]) for packing into a state vector.

        This is what you'd concatenate with position/velocity/etc. when
        building a big "math state" array.
        """
        # Return a copy to avoid accidental external mutation.
        return self.q.astype(float, copy=True)

    # ---------- invariants ----------

    def normalized(self) -> "Orientation":
        """
        Return a unit-norm Orientation with the same rotation.

        If the quaternion is zero-length (shouldn't normally happen),
        fall back to identity.
        """
        arr = self.as_array()
        n = float(np.linalg.norm(arr))
        if n == 0.0:
            return Orientation.identity()
        arr /= n
        return Orientation(arr)

    # ---------- SciPy rotation helper ----------

    def _as_scipy_rotation(self) -> R:
        """
        Internal: return a SciPy Rotation object for this orientation.

        SciPy uses [x, y, z, w] ordering for quaternions, so we convert.
        """
        w, x, y, z = self.as_array()
        return R.from_quat([x, y, z, w])

    # ---------- algebra & rotation matrix ----------

    def conjugate(self) -> "Orientation":
        """Conjugate; for unit quaternions this is the inverse rotation."""
        w, x, y, z = self.as_array()
        return Orientation.from_wxyz(w, -x, -y, -z)

    def rotation_matrix(self) -> np.ndarray:
        """
        3x3 rotation matrix R mapping body vectors -> world vectors.

        For any body-frame vector v_B (length-3 array), world-frame is:
            v_W = R @ v_B
        """
        return self._as_scipy_rotation().as_matrix()

    # ---------- Optional: pydantic v2 schema ---------------------------

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: "GetCoreSchemaHandler",
    ) -> "core_schema.CoreSchema":
        """
        Pydantic v2 integration: serialize Orientation as
        {"w": float, "x": float, "y": float, "z": float}.
        """
        if not PYDANTIC_AVAILABLE:
            return handler(source_type)

        def serialize_orientation(value: "Orientation") -> dict:
            w, x, y, z = value.as_array()
            return {
                "w": float(w),
                "x": float(x),
                "y": float(y),
                "z": float(z),
            }

        def validate_orientation(value: Any) -> "Orientation":
            if isinstance(value, cls):
                return value
            if isinstance(value, dict):
                return cls.from_wxyz(
                    float(value["w"]),
                    float(value["x"]),
                    float(value["y"]),
                    float(value["z"]),
                )
            if isinstance(value, (list, tuple)) and len(value) == 4:
                return cls.from_array(value)
            raise ValueError(f"Cannot convert {type(value)} to {cls.__name__}")

        return core_schema.no_info_plain_validator_function(
            validate_orientation,
            serialization=core_schema.plain_serializer_function_ser_schema(
                serialize_orientation,
            ),
        )
