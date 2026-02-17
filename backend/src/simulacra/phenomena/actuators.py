"""
The rocket's effectors. Commands from the controller become physical changes,
but only if god allows it (saturation, rate limits, failures).

Actuator dynamics:
- Commands are latched via apply()
- Physical state evolves via step(dt) with rate limits and lag
- step() must be called each simulation tick to update the rocket state
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..noumena.rocket import Rocket
from ..physics import RocketVec3
from .errors import SensorError, IdentityError


@dataclass
class ThrottleCommand:
    """Desired thrust magnitude in Newtons."""

    thrust_N: float


class ThrottleActuator:
    """
    Throttle actuator controlling thrust magnitude.

    Dynamics:
    - Rate limiting: thrust changes at most max_thrust_rate N/s
    - Deadband: thrust below min_thrust snaps to 0 (engine cutoff)

    Imperfections:
    - alignment_errors: dict with "yaw" and/or "pitch" SensorErrors that perturb
      the actual thrust direction from commanded. Models manufacturing tolerances,
      combustion instabilities, nozzle asymmetries.

    Preserves the current thrust direction while changing magnitude.
    If the engine was off (zero thrust), defaults to body +X (straight ahead).
    """

    def __init__(
        self,
        rocket: Rocket,
        max_thrust_rate: float = float("inf"),  # N/s, infinite = instantaneous
        min_thrust: float = 0.0,  # N, deadband threshold
        alignment_errors: Optional[dict[str, SensorError]] = None,  # yaw/pitch errors
    ):
        self._rocket = rocket
        self.max_thrust_rate = float(max_thrust_rate)
        self.min_thrust = float(min_thrust)

        # Alignment errors (thrust direction perturbation)
        self._alignment_errors = alignment_errors or {}
        self._time: float = 0.0

        # Internal state
        self._commanded_thrust: float = 0.0
        self._actual_thrust: float = self._rocket.state.thrust_body.norm
        # Nominal direction (set by gimbal or default to +X)
        self._nominal_direction: np.ndarray = np.array([1.0, 0.0, 0.0])

    def set_time(self, t: float) -> None:
        """Set current simulation time (for time-dependent errors)."""
        self._time = t

    def apply(self, command: ThrottleCommand) -> None:
        """
        Set the commanded thrust. Actual thrust will evolve in step().
        """
        self._commanded_thrust = max(0.0, command.thrust_N)

    def step(self, dt: float) -> None:
        """
        Advance actuator dynamics by dt seconds.

        Updates actual thrust toward commanded, respecting rate limit.
        Then applies to the rocket state.
        """
        # Rate limit
        max_delta = self.max_thrust_rate * dt
        delta = self._commanded_thrust - self._actual_thrust
        if abs(delta) > max_delta:
            delta = np.sign(delta) * max_delta

        self._actual_thrust += delta

        # Deadband: snap to zero if below threshold
        if self._actual_thrust < self.min_thrust:
            self._actual_thrust = 0.0

        # Apply to rocket
        self._update_rocket_thrust()

    def set_nominal_direction(self, direction: np.ndarray) -> None:
        """Set the nominal thrust direction (called by gimbal actuator)."""
        n = np.linalg.norm(direction)
        if n > 0:
            self._nominal_direction = direction / n

    def _update_rocket_thrust(self) -> None:
        """Write current actual thrust to rocket state."""
        # Start from nominal direction and apply alignment errors.
        # Nominal is set by gimbal (if used) or defaults to body +X.
        direction = self._nominal_direction.copy()

        # Apply alignment errors (thrust misalignment from nominal)
        direction = self._apply_alignment_errors(direction)

        self._rocket.state.thrust_body = RocketVec3(
            self._actual_thrust * direction[0],
            self._actual_thrust * direction[1],
            self._actual_thrust * direction[2],
        )

    def _apply_alignment_errors(self, direction: np.ndarray) -> np.ndarray:
        """
        Apply alignment errors to perturb thrust direction.

        Errors are small-angle rotations about yaw (body +Z) and pitch (body +Y).
        """
        if not self._alignment_errors:
            return direction

        # Get error angles (radians) - error.apply(0, t) gives the perturbation
        yaw_error = self._alignment_errors.get("yaw", IdentityError())
        pitch_error = self._alignment_errors.get("pitch", IdentityError())

        d_yaw = yaw_error.apply(0.0, self._time)  # rotation about +Z
        d_pitch = pitch_error.apply(0.0, self._time)  # rotation about +Y

        # Apply rotations
        if abs(d_yaw) > 1e-12:
            r_yaw = R.from_rotvec(d_yaw * np.array([0.0, 0.0, 1.0]))
            direction = r_yaw.apply(direction)

        if abs(d_pitch) > 1e-12:
            r_pitch = R.from_rotvec(d_pitch * np.array([0.0, 1.0, 0.0]))
            direction = r_pitch.apply(direction)

        # Normalize (protect against drift)
        n = np.linalg.norm(direction)
        if n > 0:
            direction = direction / n

        return direction


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

    Dynamics:
    - Rate limiting: angles change at most max_rate rad/s
    - First-order lag: actual angle approaches commanded with time constant tau

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
        max_rate: float = float("inf"),  # rad/s, infinite = instantaneous
        tau: float = 0.0,  # seconds, 0 = no lag
    ):
        self._rocket = rocket
        self.max_outer_angle = float(max_outer_angle)
        self.max_inner_angle = float(max_inner_angle)
        self.max_rate = float(max_rate)
        self.tau = float(tau)

        # Commanded angles (set by apply)
        self._commanded_outer: float = 0.0
        self._commanded_inner: float = 0.0

        # Actual angles (evolve in step)
        self.outer_angle: float = 0.0
        self.inner_angle: float = 0.0

    def apply(self, command: GimbalCommand) -> None:
        """
        Set commanded gimbal angles (with saturation).

        Actual angles will evolve toward commanded in step().
        """
        self._commanded_outer = float(
            np.clip(
                command.desired_outer_angle,
                -self.max_outer_angle,
                self.max_outer_angle,
            )
        )
        self._commanded_inner = float(
            np.clip(
                command.desired_inner_angle,
                -self.max_inner_angle,
                self.max_inner_angle,
            )
        )

    def step(self, dt: float) -> None:
        """
        Advance actuator dynamics by dt seconds.

        Updates actual angles toward commanded, respecting rate limit and lag.
        Then updates the rocket thrust vector.
        """
        self.outer_angle = self._advance_angle(
            self.outer_angle, self._commanded_outer, dt
        )
        self.inner_angle = self._advance_angle(
            self.inner_angle, self._commanded_inner, dt
        )

        # Update rocket thrust direction
        self._update_rocket_thrust()

    def _advance_angle(self, actual: float, commanded: float, dt: float) -> float:
        """
        Advance a single angle toward its commanded value.

        Applies first-order lag if tau > 0, then rate limiting.
        """
        if self.tau > 0:
            # First-order lag: actual' = (commanded - actual) / tau
            # Discrete: actual += dt * (commanded - actual) / tau
            alpha = min(1.0, dt / self.tau)
            target = actual + alpha * (commanded - actual)
        else:
            target = commanded

        # Rate limit
        max_delta = self.max_rate * dt
        delta = target - actual
        if abs(delta) > max_delta:
            delta = np.sign(delta) * max_delta

        return actual + delta

    def _update_rocket_thrust(self) -> None:
        """Compute and write thrust direction to rocket state."""
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


class Actuators:
    """
    Bundle of actuators for centralized management.

    Provides:
    - Unified step() for advancing all actuator dynamics
    - Properties for logging actuator state
    """

    def __init__(
        self,
        gimbal: GimbalActuator | None = None,
        throttle: ThrottleActuator | None = None,
    ):
        self._gimbal = gimbal
        self._throttle = throttle
        self._throttle_command: float | None = None

    def set_time(self, t: float) -> None:
        """Set current simulation time (for time-dependent errors)."""
        if self._throttle is not None:
            self._throttle.set_time(t)

    def step(self, dt: float) -> None:
        """
        Advance all actuator dynamics by dt seconds.

        Called by Scenario each simulation tick.
        Gimbal runs first to set direction, then throttle applies magnitude + errors.
        """
        if self._gimbal is not None:
            self._gimbal.step(dt)
            # Pass nominal direction from gimbal to throttle
            if self._throttle is not None:
                direction = _nozzle_direction_body(
                    self._gimbal.outer_angle,
                    self._gimbal.inner_angle,
                )
                self._throttle.set_nominal_direction(direction)

        if self._throttle is not None:
            self._throttle.step(dt)

    @property
    def gimbal_outer_angle(self) -> float | None:
        """Current outer gimbal angle (radians)."""
        if self._gimbal is None:
            return None
        return self._gimbal.outer_angle

    @property
    def gimbal_inner_angle(self) -> float | None:
        """Current inner gimbal angle (radians)."""
        if self._gimbal is None:
            return None
        return self._gimbal.inner_angle

    @property
    def throttle_command(self) -> float | None:
        """Last commanded throttle (Newtons)."""
        return self._throttle_command

    def set_throttle_command(self, thrust_N: float) -> None:
        """Record a throttle command for logging."""
        self._throttle_command = thrust_N
