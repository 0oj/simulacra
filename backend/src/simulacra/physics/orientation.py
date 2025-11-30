"""
Orientation helper built on top of numpy-quaternion.

Conventions:
- Underlying type is `np.quaternion(w, x, y, z)` representing
  w*1 + x*i + y*j + z*k.
- i, j, k are aligned with the X (red), Y (green), Z (blue) axes.
- Orientation represents a rotation mapping body-frame vectors -> world-frame vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .vec3 import RocketVec3, WorldVec3


ArrayLike3 = Tuple[float, float, float] | np.ndarray


@dataclass
class Orientation:
    """
    Rocket orientation (body -> world), stored as a unit quaternion.

    This is a *thin* semantic wrapper around `np.quaternion`, so the
    underlying math stays visible and we can still use numpy-quaternion
    directly when needed.
    """

    q: np.quaternion

    # ---------- constructors ----------

    @staticmethod
    def identity() -> "Orientation":
        """No rotation: body axes aligned with world axes."""
        return Orientation(np.quaternion(1.0, 0.0, 0.0, 0.0))

    @classmethod
    def from_wxyz(cls, w: float, x: float, y: float, z: float) -> "Orientation":
        """Build from scalar/vector parts explicitly."""
        return cls(np.quaternion(w, x, y, z)).normalized()

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Orientation":
        """Build from np.array([w, x, y, z])."""
        w, x, y, z = map(float, arr)
        return cls(np.quaternion(w, x, y, z)).normalized()

    def as_array(self) -> np.ndarray:
        """Return np.array([w, x, y, z]) for logging / math."""
        return np.array([self.q.w, self.q.x, self.q.y, self.q.z], dtype=float)

    # ---------- normalization ----------

    def normalized(self) -> "Orientation":
        """Return a unit-norm Orientation with the same rotation."""
        arr = self.as_array()
        n = np.linalg.norm(arr)
        if n == 0.0:
            return Orientation.identity()
        arr /= n
        return Orientation.from_array(arr)

    # ---------- basic algebra ----------

    def conjugate(self) -> "Orientation":
        """Conjugate (inverse rotation for unit quaternions)."""
        q = self.q
        return Orientation(np.quaternion(q.w, -q.x, -q.y, -q.z))

    def __mul__(self, other: "Orientation") -> "Orientation":
        """
        Compose rotations: first self, then other.

        Interpreting vectors as v' = q * v * q^*, this corresponds to
        applying self, then other.
        """
        return Orientation(self.q * other.q).normalized()

    # ---------- rotations ----------

    def rotate_body_to_world(self, v_body: ArrayLike3 | RocketVec3) -> WorldVec3:
        """
        Rotate a 3D vector from body frame to world frame.

        v_body is expressed in the rocket's body RGB axes.
        """
        vx, vy, vz = map(float, np.asarray(v_body))
        v_quat = np.quaternion(0.0, vx, vy, vz)

        qn = self.normalized().q
        q_conj = np.quaternion(qn.w, -qn.x, -qn.y, -qn.z)

        v_world_quat = qn * v_quat * q_conj
        return WorldVec3(v_world_quat.x, v_world_quat.y, v_world_quat.z)

    def rotation_matrix(self) -> np.ndarray:
        """
        3x3 rotation matrix R mapping body vectors -> world vectors.

        For any body-frame vector v_B, the world-frame vector is:
            v_W = R @ v_B
        """
        w, x, y, z = self.normalized().as_array()

        xx, yy, zz = x * x, y * y, z * z
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z

        return np.array(
            [
                [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
                [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
                [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
            ],
            dtype=float,
        )