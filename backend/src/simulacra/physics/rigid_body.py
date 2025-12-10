"""
Rigid-body dynamics helpers.

This module holds physics assumptions about treating the rocket as a
rigid body with a body-frame angular velocity, separate from the raw
quaternion math in `quat.py`.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation as R

from .quat import Orientation


def integrate_orientation_body_omega(
    orientation: Orientation,
    omega_body: Iterable[float],
    dt: float,
) -> Orientation:
    """
    Advance orientation forward by dt given body-frame angular velocity.

    Args:
        orientation: Current body->world orientation.
        omega_body: Iterable[float] [wx, wy, wz] in rad/s, body frame.
        dt: Time step in seconds.

    Uses a simple first-order update based on composition of a small
    body-frame rotation:

        R_new = R_current * R_delta

    where R_delta = exp([omega]× * dt).
    """
    omega = np.asarray(omega_body, dtype=float).reshape(3)

    r_delta = R.from_rotvec(omega * float(dt))
    r_current = orientation._as_scipy_rotation()
    r_new = r_current * r_delta

    x, y, z, w = r_new.as_quat()
    return Orientation.from_wxyz(w, x, y, z)


