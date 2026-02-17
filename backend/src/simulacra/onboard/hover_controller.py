"""
Full hover controller with altitude and attitude control.

Cascade architecture:
- Outer loop: PD altitude control → thrust magnitude
- Inner loop: Quaternion PD attitude control + gravity feedforward → gimbal angles

This is a baseline controller for comparison with RL approaches.
Based on:
- NASA quaternion-based control (NTRS 20120014565)
- Standard TVC implementations for model rockets

Key insight: A hovering rocket with tilt must deflect the gimbal to counter gravity
torque. This is a static requirement (feedforward), not just a dynamic one (feedback).
Without gravity compensation, the controller needs impossibly high gains.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..phenomena.actuators import (
    ThrottleActuator,
    ThrottleCommand,
    GimbalActuator,
    GimbalCommand,
)
from ..phenomena.sensors import Sensors
from .controller import Controller


@dataclass
class HoverControllerConfig:
    """Configuration for the hover controller."""

    # Target state
    target_alt: float = 10.0  # meters

    # Physical parameters
    weight: float = 981.0  # N (mass * g)
    max_thrust: float = 2000.0  # N
    gravity: float = 9.81  # m/s^2

    # Altitude control gains (outer loop)
    alt_Kp: float = 100.0  # N/m
    alt_Kd: float = 200.0  # N/(m/s)

    # Attitude control gains (inner loop)
    # These handle transient errors; gravity is handled by feedforward
    att_Kp: float = 500.0  # N·m/rad (proportional on quaternion error)
    att_Kd: float = 200.0  # N·m/(rad/s) (derivative on angular velocity)

    # Gimbal parameters
    max_gimbal_angle: float = math.radians(15.0)  # rad
    thrust_moment_arm: float = 2.0  # meters (distance from CoM to thrust point)


class HoverController(Controller):
    """
    Full 6-DOF hover controller with altitude and attitude stabilization.

    Architecture:
    1. Altitude PD → thrust magnitude command
    2. Attitude PD → desired torque → gimbal angles

    The attitude controller uses quaternion error, which is singularity-free
    and well-suited for small-angle control around vertical.
    """

    def __init__(
        self,
        throttle: ThrottleActuator,
        gimbal: GimbalActuator,
        config: Optional[HoverControllerConfig] = None,
    ):
        self.throttle = throttle
        self.gimbal = gimbal
        self.config = config or HoverControllerConfig()
        self.sensors: Optional[Sensors] = None

    def with_sensors(self, sensors: Sensors) -> "HoverController":
        self.sensors = sensors
        return self

    def run(self) -> None:
        if self.sensors is None:
            raise ValueError("Sensors not attached to controller")

        cfg = self.config

        # Read sensor data
        pos = self.sensors.GPS.read()
        vel = self.sensors.VelocitySensor.read()
        orientation = self.sensors.OrientationSensor.read()
        quat = tuple(orientation.as_array())  # (w, x, y, z), body→world
        omega_body = self.sensors.GyroSensor.read()  # angular velocity in body frame

        # === Altitude Control (outer loop) ===
        alt_error = cfg.target_alt - pos.y
        vy = vel.y

        # PD control for thrust magnitude
        # When tilted, we need more thrust to maintain altitude
        # Account for tilt by dividing by cos(tilt_angle)
        tilt_factor = self._compute_tilt_factor(quat)
        base_thrust = cfg.weight + cfg.alt_Kp * alt_error - cfg.alt_Kd * vy
        thrust = base_thrust / max(tilt_factor, 0.5)  # Limit correction
        thrust = max(0.0, min(thrust, cfg.max_thrust))

        self.throttle.apply(ThrottleCommand(thrust_N=thrust))

        # === Attitude Control (inner loop) ===
        # Desired orientation: upright (body +X aligned with world +Y)
        # This is a 90° rotation about +Z from identity
        quat_desired = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))  # w, x, y, z

        # Compute quaternion error
        quat_error = self._quaternion_error(quat, quat_desired)

        # The vector part of the error quaternion gives the rotation axis * sin(θ/2)
        # For small angles, this is approximately the rotation axis * θ/2
        # Scale by 2 to get approximate rotation angle
        error_vec = np.array([quat_error[1], quat_error[2], quat_error[3]]) * 2.0

        # Angular velocity damping (in body frame)
        omega = np.array([omega_body.x, omega_body.y, omega_body.z])

        # === Gravity Feedforward ===
        # Compute gravity torque in body frame. This is the torque that gravity
        # exerts about the thrust application point (assumed at base).
        # τ = r_CoM × F_gravity_body
        # where r_CoM is CoM position relative to thrust point in body frame.
        #
        # For our rocket: CoM is at (thrust_moment_arm, 0, 0) in body frame
        # (along body +X, which is the nose direction)
        #
        # Gravity in body frame = R^T * [0, -mg, 0]
        # where R is the rotation matrix from body to world
        gravity_world = np.array([0.0, -cfg.weight, 0.0])  # weight = m*g
        gravity_body = self._rotate_world_to_body(quat, gravity_world)

        # CoM offset in body frame (from thrust point to CoM)
        r_com = np.array([cfg.thrust_moment_arm, 0.0, 0.0])

        # Gravity torque about thrust point: τ = r × F
        tau_gravity = np.cross(r_com, gravity_body)

        # === PD Feedback + Feedforward ===
        # The gimbal must produce torque to:
        # 1. Cancel gravity torque (feedforward)
        # 2. Correct attitude error (proportional)
        # 3. Damp angular velocity (derivative)
        tau_feedback = cfg.att_Kp * error_vec - cfg.att_Kd * omega
        tau_desired = -tau_gravity + tau_feedback  # Counter gravity + feedback

        # Convert torque to gimbal angles
        # Torque = thrust × moment_arm × sin(gimbal_angle)
        # For small angles: gimbal_angle ≈ torque / (thrust × moment_arm)
        if thrust > 100.0:  # Only control attitude when there's significant thrust
            scale = 1.0 / (thrust * cfg.thrust_moment_arm)

            # Gimbal angles in body frame:
            # - outer_angle (yaw): rotation about body +Z, creates torque about +Z
            # - inner_angle (pitch): rotation about +Y (after yaw), creates torque about +Y
            # Sign convention: positive gimbal deflection creates negative torque
            # (thrust vector deflects opposite to gimbal rotation)
            outer_angle = -tau_desired[2] * scale  # Yaw torque → outer gimbal
            inner_angle = -tau_desired[1] * scale  # Pitch torque → inner gimbal

            # Clamp to gimbal limits
            outer_angle = np.clip(outer_angle, -cfg.max_gimbal_angle, cfg.max_gimbal_angle)
            inner_angle = np.clip(inner_angle, -cfg.max_gimbal_angle, cfg.max_gimbal_angle)
        else:
            outer_angle = 0.0
            inner_angle = 0.0

        self.gimbal.apply(GimbalCommand(
            desired_outer_angle=float(outer_angle),
            desired_inner_angle=float(inner_angle),
        ))

    def _rotate_world_to_body(self, quat: tuple, vec_world: np.ndarray) -> np.ndarray:
        """
        Rotate a vector from world frame to body frame.

        The quaternion quat represents body→world, so we need the inverse
        (conjugate) for world→body.
        """
        w, x, y, z = quat
        # Quaternion conjugate (inverse for unit quaternion)
        qinv = (w, -x, -y, -z)

        # Apply rotation: v' = q^-1 * v * q
        # For quaternion q = (w, x, y, z) and vector v = (vx, vy, vz):
        vx, vy, vz = vec_world

        # First compute q^-1 * v (treating v as quaternion (0, vx, vy, vz))
        # Then multiply result by q
        # Using the standard quaternion rotation formula:
        # v' = v + 2*w*(q_vec × v) + 2*(q_vec × (q_vec × v))
        # where q_vec = (x, y, z)

        # For the inverse quaternion, q_vec becomes (-x, -y, -z)
        qv = np.array([-x, -y, -z])
        v = vec_world

        # v' = v + 2*w*(qv × v) + 2*(qv × (qv × v))
        cross1 = np.cross(qv, v)
        cross2 = np.cross(qv, cross1)
        return v + 2.0 * w * cross1 + 2.0 * cross2

    def _compute_tilt_factor(self, quat: tuple) -> float:
        """
        Compute cos(tilt_angle) - how vertical the rocket is.

        Returns 1.0 when upright, 0.0 when horizontal.
        """
        w, x, y, z = quat
        # Body +X (nose) rotated to world frame
        # nose_world.y tells us how vertical we are
        nose_y = 2.0 * (x * y + w * z)
        return max(0.0, nose_y)  # Clamp to non-negative

    def _quaternion_error(
        self,
        quat_current: tuple,
        quat_desired: tuple,
    ) -> tuple:
        """
        Compute quaternion error: q_error = q_desired^-1 * q_current

        The error quaternion represents the rotation from current to desired.
        For small errors, the vector part is approximately half the rotation angle.

        Returns (w, x, y, z) error quaternion.
        """
        # Quaternion inverse of desired (conjugate, since unit quaternion)
        qd_inv = (quat_desired[0], -quat_desired[1], -quat_desired[2], -quat_desired[3])

        # Quaternion multiplication: qd_inv * qc
        qc = quat_current
        w = qd_inv[0] * qc[0] - qd_inv[1] * qc[1] - qd_inv[2] * qc[2] - qd_inv[3] * qc[3]
        x = qd_inv[0] * qc[1] + qd_inv[1] * qc[0] + qd_inv[2] * qc[3] - qd_inv[3] * qc[2]
        y = qd_inv[0] * qc[2] - qd_inv[1] * qc[3] + qd_inv[2] * qc[0] + qd_inv[3] * qc[1]
        z = qd_inv[0] * qc[3] + qd_inv[1] * qc[2] - qd_inv[2] * qc[1] + qd_inv[3] * qc[0]

        # Ensure shortest path (w should be positive)
        if w < 0:
            w, x, y, z = -w, -x, -y, -z

        return (w, x, y, z)
