"""
The rocket's effectors. Commands from the controller become physical changes,
but only if god allows it (saturation, rate limits, failures).
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..noumena import Kernel2D
from ..noumena.rocket import Rocket3D
from ..physics import RocketVec3


# ============================================================================
# 2D actuators (legacy)
# ============================================================================


@dataclass
class GimbalCommand2D:
    desired_gimbal_angle: float  # radians, desired gimbal angle


class GimbalActuator:
    max_gimbal_angle: float  # radians, maximum gimbal angle

    def __init__(self, kernel: Kernel2D, max_gimbal_angle: float):
        self.max_gimbal_angle = max_gimbal_angle
        self._kernel = kernel

    def apply(self, command: GimbalCommand2D) -> None:
        ...  # TODO: implement rate limiting, saturation, etc.
        self._kernel.rocket.state.gimbal_angle = np.clip(
            command.desired_gimbal_angle,
            -self.max_gimbal_angle,
            self.max_gimbal_angle,
        )


@dataclass
class ThrustCommand2D:
    desired_thrust_magnitude: float  # Newtons, desired thrust magnitude


class ThrustActuator:
    max_thrust_magnitude: float  # Newtons, maximum thrust magnitude

    def __init__(self, kernel: Kernel2D, max_thrust_magnitude: float):
        self.max_thrust_magnitude = max_thrust_magnitude
        self._kernel = kernel

    def apply(self, command: ThrustCommand2D) -> None:
        self._kernel.rocket.state.thrust_magnitude = np.clip(
            command.desired_thrust_magnitude,
            0.0,
            self.max_thrust_magnitude,
        )


# ============================================================================
# 3D actuators (work in progress)
# ============================================================================


@dataclass
class GimbalCommand3D:
    """
    Desired outer/inner gimbal angles for a 2-DOF nozzle.

    - outer_angle: rotation about the rocket body +Z axis (yaw)
    - inner_angle: rotation about the inner gimbal's own +Y axis
    """

    desired_outer_angle: float  # radians
    desired_inner_angle: float  # radians


class GimbalActuator3D:
    """
    2-DOF nested gimbal actuator driving the 3D rocket's thrust vector.

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
        rocket: Rocket3D,
        max_outer_angle: float,
        max_inner_angle: float,
    ):
        self._rocket = rocket
        self.max_outer_angle = float(max_outer_angle)
        self.max_inner_angle = float(max_inner_angle)

        self.outer_angle: float = 0.0
        self.inner_angle: float = 0.0

    def apply(self, command: GimbalCommand3D) -> None:
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

        # Scale by current thrust magnitude from noumena state.
        T = float(self._rocket.state.thrust_magnitude)
        self._rocket.state.thrust_direction = RocketVec3(
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
