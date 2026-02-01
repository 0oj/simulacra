"""
The rocket's effectors. Commands from the controller become physical changes,
but only if god allows it (saturation, rate limits, failures).
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..noumena.rocket import Rocket
from ..physics import RocketVec3


@dataclass
class ThrottleCommand:
    """Desired thrust magnitude in Newtons."""

    thrust_N: float


class ThrottleActuator:
    """
    Throttle actuator controlling thrust magnitude.

    Preserves the current thrust direction while changing magnitude.
    If the engine was off (zero thrust), defaults to body +X (straight ahead).
    """

    def __init__(self, rocket: Rocket):
        self._rocket = rocket

    def apply(self, command: ThrottleCommand) -> None:
        T = command.thrust_N
        current = self._rocket.state.thrust_body
        if current.norm == 0.0:
            self._rocket.state.thrust_body = RocketVec3(T, 0.0, 0.0)
        else:
            self._rocket.state.thrust_body = current.direction * T


@dataclass
class GimbalCommand:
    """
    Desired outer/inner gimbal angles for a 2-DOF nozzle.

    - outer_angle: rotation about the rocket body +Z axis (yaw)
    - inner_angle: rotation about the inner gimbal's own +Y axis
    """

    desired_outer_angle: float  # radians
    desired_inner_angle: float  # radians


class GimbalActuator:
    """
    2-DOF nested gimbal actuator driving the rocket's thrust vector.

    Frames (conceptually):
    - Body frame B: attached to the rocket, +X along the nose.
    - Outer gimbal frame G1: rotates about body +Z by outer_angle.
    - Inner gimbal frame G2: rotates about its own +Y axis by inner_angle.

    The nozzle points along +X in the inner frame. The resulting nozzle
    direction in the body frame is:

        v_body = R_outer * R_inner * [1, 0, 0]

    where:
    - R_outer: rotation about body +Z
    - R_inner: rotation about +Y in the *outer* frame (nested)
    """

    def __init__(
        self,
        rocket: Rocket,
        max_outer_angle: float,
        max_inner_angle: float,
    ):
        self._rocket = rocket
        self.max_outer_angle = float(max_outer_angle)
        self.max_inner_angle = float(max_inner_angle)

        self.outer_angle: float = 0.0
        self.inner_angle: float = 0.0

    def apply(self, command: GimbalCommand) -> None:
        """
        Update internal gimbal angles and write thrust_vector_body on the rocket.
        """
        # TODO: Add rate limiting and failures.
        self.outer_angle = float(
            np.clip(
                command.desired_outer_angle,
                -self.max_outer_angle,
                self.max_outer_angle,
            )
        )
        self.inner_angle = float(
            np.clip(
                command.desired_inner_angle,
                -self.max_inner_angle,
                self.max_inner_angle,
            )
        )

        # Compute unit nozzle direction in body frame using nested frames.
        dir_body = _nozzle_direction_body(self.outer_angle, self.inner_angle)

        # Preserve current thrust magnitude, redirect it.
        T = float(self._rocket.state.thrust_body.norm)
        self._rocket.state.thrust_body = RocketVec3(
            T * dir_body[0],
            T * dir_body[1],
            T * dir_body[2],
        )


def _nozzle_direction_body(outer_angle: float, inner_angle: float) -> np.ndarray:
    """
    Compute unit nozzle direction in the *body* frame from nested gimbal angles.

    - outer_angle: rotation about body +Z (yaw)
    - inner_angle: rotation about +Y in the *outer* gimbal frame

    We reuse the same frame-transition idea as for world/body rotations:
    1. Start with the body frame basis.
    2. Apply outer rotation Rz(outer_angle) to get the outer gimbal frame.
    3. Express the inner gimbal +Y axis in the body frame.
    4. Apply rotation about that axis by inner_angle.
    5. Rotate the nominal nozzle vector [1, 0, 0] in the inner frame.
    """
    # Outer rotation about body +Z
    r_outer = R.from_rotvec(outer_angle * np.array([0.0, 0.0, 1.0]))

    # Inner rotation axis: +Y in the outer gimbal frame, expressed in body frame.
    axis_inner_body = r_outer.apply([0.0, 1.0, 0.0])
    axis_inner_body = np.asarray(axis_inner_body, dtype=float)

    if np.linalg.norm(axis_inner_body) == 0.0:
        # Degenerate, shouldn't happen, fall back to body +Y.
        axis_inner_body = np.array([0.0, 1.0, 0.0], dtype=float)

    r_inner = R.from_rotvec(inner_angle * axis_inner_body)

    # Total rotation from inner nozzle frame to body frame.
    r_total = r_inner * r_outer

    # Nozzle points along +X in inner frame.
    v_inner = np.array([1.0, 0.0, 0.0], dtype=float)
    v_body = r_total.apply(v_inner)
    # Ensure unit length (protect against numerical drift).
    v_body = np.asarray(v_body, dtype=float)
    n = float(np.linalg.norm(v_body))
    if n == 0.0:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return v_body / n
