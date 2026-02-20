"""
Landing controller for VTVL rocket.

Uses proportional guidance to descend and land at a target position.
Not a true gravity turn (suicide burn), but a more forgiving approach
that tracks a velocity profile based on altitude.

Architecture:
1. Guidance: compute desired velocity vector based on position
2. Attitude: tilt rocket to align thrust with desired acceleration
3. Throttle: control thrust magnitude to track velocity profile

The key insight: to go somewhere, you need to be moving toward it,
but slowing down as you approach. The velocity profile encodes this.
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
class LandingControllerConfig:
    """Configuration for landing controller."""

    # Target landing position
    target_x: float = 0.0
    target_z: float = 0.0

    # Physical parameters
    weight: float = 981.0  # N (mass * g)
    max_thrust: float = 2000.0  # N
    gravity: float = 9.81  # m/s^2

    # Guidance parameters
    # Desired descent rate = k_descent * sqrt(altitude)
    # This gives a natural deceleration profile
    k_descent: float = 2.0  # tuning parameter

    # Horizontal guidance gain
    # Desired horizontal velocity = k_horizontal * displacement
    # Clamped to max_horizontal_velocity
    k_horizontal: float = 0.3
    max_horizontal_velocity: float = 10.0  # m/s

    # Velocity tracking gains
    # Desired acceleration = Kp * (v_desired - v_actual)
    Kp_vertical: float = 2.0
    Kp_horizontal: float = 1.5

    # Attitude control gains (same as HoverController)
    att_Kp: float = 500.0  # N·m/rad
    att_Kd: float = 200.0  # N·m/(rad/s)

    # Gimbal parameters
    max_gimbal_angle: float = math.radians(15.0)
    thrust_moment_arm: float = 2.0  # meters

    # Terminal phase (close to ground)
    terminal_altitude: float = 5.0  # meters
    terminal_descent_rate: float = 1.0  # m/s (slow final approach)


class LandingController(Controller):
    """
    Proportional guidance landing controller.

    The controller computes a desired velocity based on position,
    then computes the thrust vector needed to achieve that velocity.
    Attitude control points the rocket in the thrust direction.

    This is NOT a gravity turn / suicide burn. It's a continuous
    guidance approach that's more forgiving of errors but less
    fuel-efficient.
    """

    def __init__(
        self,
        throttle: ThrottleActuator,
        gimbal: GimbalActuator,
        config: Optional[LandingControllerConfig] = None,
    ):
        self.throttle = throttle
        self.gimbal = gimbal
        self.config = config or LandingControllerConfig()
        self.sensors: Optional[Sensors] = None

    def with_sensors(self, sensors: Sensors) -> "LandingController":
        self.sensors = sensors
        return self

    def run(self) -> None:
        if self.sensors is None:
            raise ValueError("Sensors not attached to controller")

        cfg = self.config

        # Read sensors
        gps = self.sensors.GPS.read()  # Horizontal position
        alt = self.sensors.Barometer.read()  # Altitude
        vel = self.sensors.VelocitySensor.read()
        orientation = self.sensors.OrientationSensor.read()
        omega_body = self.sensors.GyroSensor.read()

        # Current state
        x, z = gps.x, gps.z
        y = alt
        vx, vy, vz = vel.x, vel.y, vel.z
        altitude = max(0.1, y)  # clamp to avoid issues near ground
        vel_mag = math.sqrt(vx**2 + vy**2 + vz**2)
        omega_mag = math.sqrt(omega_body.x**2 + omega_body.y**2 + omega_body.z**2)

        # === LANDING DETECTION ===
        # When very close to ground with low velocity, cut thrust and let ground handle it
        if y < 0.3 and vel_mag < 0.5:
            self.throttle.apply(ThrottleCommand(thrust_N=0.0))
            self.gimbal.apply(GimbalCommand(
                desired_outer_angle=0.0,
                desired_inner_angle=0.0,
            ))
            return

        # === GUIDANCE: Compute desired velocity ===

        # Horizontal: move toward target, slow down as we approach
        dx = cfg.target_x - x  # displacement to target
        dz = cfg.target_z - z
        dist_horizontal = math.sqrt(dx**2 + dz**2)

        # Desired horizontal velocity: proportional to displacement
        # But also limited based on altitude - can't steer aggressively near ground
        altitude_factor = min(1.0, altitude / 50.0)  # full authority above 50m
        effective_k = cfg.k_horizontal * altitude_factor

        vx_desired = effective_k * dx
        vz_desired = effective_k * dz

        # Clamp horizontal velocity
        v_horiz_desired_mag = math.sqrt(vx_desired**2 + vz_desired**2)
        max_horiz = cfg.max_horizontal_velocity * altitude_factor
        if v_horiz_desired_mag > max_horiz:
            scale = max_horiz / v_horiz_desired_mag
            vx_desired *= scale
            vz_desired *= scale

        # Vertical: descent rate based on altitude
        # Higher altitude = faster descent allowed
        # v_descent = -k * sqrt(h) gives natural deceleration

        if altitude > cfg.terminal_altitude:
            # Normal descent phase
            vy_desired = -cfg.k_descent * math.sqrt(altitude)
        else:
            # Terminal phase: constant slow descent
            vy_desired = -cfg.terminal_descent_rate

        # Very close to ground: even slower
        if altitude < 1.0:
            vy_desired = max(vy_desired, -0.5)

        # === CONTROL: Compute desired acceleration ===

        # Velocity error
        vx_error = vx_desired - vx
        vy_error = vy_desired - vy
        vz_error = vz_desired - vz

        # Desired acceleration (in world frame) - this is what we want the rocket to do
        ax_desired = cfg.Kp_horizontal * vx_error
        ay_desired = cfg.Kp_vertical * vy_error  # Note: no gravity here yet
        az_desired = cfg.Kp_horizontal * vz_error

        # === THRUST VECTOR: Convert acceleration to thrust ===

        # The thrust must produce the desired acceleration PLUS cancel gravity
        # T_world = m * (a_desired + [0, g, 0])
        mass = cfg.weight / cfg.gravity

        # Thrust vector in world frame
        Tx = mass * ax_desired
        Ty = mass * (ay_desired + cfg.gravity)  # Add gravity compensation here
        Tz = mass * az_desired

        # Thrust magnitude
        thrust_desired = math.sqrt(Tx**2 + Ty**2 + Tz**2)

        # If Ty < 0, we want to accelerate down faster than gravity - just cut thrust
        if Ty < 0:
            thrust_desired = 0.0
            Tx, Ty, Tz = 0.0, 1.0, 0.0  # Default direction (up)

        thrust_desired = max(0, min(thrust_desired, cfg.max_thrust))

        # Desired thrust direction (unit vector in world frame)
        thrust_mag_raw = math.sqrt(Tx**2 + Ty**2 + Tz**2)
        if thrust_mag_raw > 0.01:
            thrust_dir_world = np.array([
                Tx / thrust_mag_raw,
                Ty / thrust_mag_raw,
                Tz / thrust_mag_raw,
            ])
        else:
            thrust_dir_world = np.array([0.0, 1.0, 0.0])  # point up

        # Safety: limit how far from vertical we can tilt
        # This prevents flip-overs from aggressive horizontal corrections
        # Tighter at low altitude where stability matters more
        base_max_tilt = math.radians(20.0)  # 20 degrees max at high altitude
        if altitude < 20.0:
            # Reduce max tilt as we approach ground
            max_tilt_angle = base_max_tilt * (altitude / 20.0)
            max_tilt_angle = max(math.radians(5.0), max_tilt_angle)  # minimum 5 deg
        else:
            max_tilt_angle = base_max_tilt

        # Compute current tilt of thrust direction
        vertical = np.array([0.0, 1.0, 0.0])
        cos_tilt = np.dot(thrust_dir_world, vertical)

        if cos_tilt < math.cos(max_tilt_angle):
            # Tilting too far - project onto the cone of allowed directions
            # Decompose into vertical and horizontal components
            horiz = thrust_dir_world - cos_tilt * vertical
            horiz_mag = np.linalg.norm(horiz)

            if horiz_mag > 0.001:
                horiz_unit = horiz / horiz_mag
                # Reconstruct at max allowed tilt
                new_cos = math.cos(max_tilt_angle)
                new_sin = math.sin(max_tilt_angle)
                thrust_dir_world = new_cos * vertical + new_sin * horiz_unit

        # === ATTITUDE CONTROL ===

        # We want body +X (nose/thrust direction) to align with thrust_dir_world
        # Current nose direction in world frame
        quat = tuple(orientation.as_array())  # (w, x, y, z)
        nose_world = self._rotate_body_to_world(quat, np.array([1.0, 0.0, 0.0]))

        # Rotation error: cross product gives axis, dot product gives angle
        # Small angle approximation: error ≈ nose × desired
        error_world = np.cross(nose_world, thrust_dir_world)

        # Convert error to body frame for gimbal control
        error_body = self._rotate_world_to_body(quat, error_world)

        # Angular velocity damping
        omega = np.array([omega_body.x, omega_body.y, omega_body.z])

        # Desired torque (in body frame)
        # Scale error by thrust (more thrust = more control authority)
        tau_feedback = cfg.att_Kp * error_body - cfg.att_Kd * omega

        # Convert torque to gimbal angles
        if thrust_desired > 100.0:
            scale = 1.0 / (thrust_desired * cfg.thrust_moment_arm)

            # Gimbal angles
            outer_angle = -tau_feedback[2] * scale  # yaw torque
            inner_angle = -tau_feedback[1] * scale  # pitch torque

            # Clamp
            outer_angle = np.clip(outer_angle, -cfg.max_gimbal_angle, cfg.max_gimbal_angle)
            inner_angle = np.clip(inner_angle, -cfg.max_gimbal_angle, cfg.max_gimbal_angle)
        else:
            outer_angle = 0.0
            inner_angle = 0.0

        # === APPLY COMMANDS ===

        self.throttle.apply(ThrottleCommand(thrust_N=thrust_desired))
        self.gimbal.apply(GimbalCommand(
            desired_outer_angle=float(outer_angle),
            desired_inner_angle=float(inner_angle),
        ))

    def _rotate_body_to_world(self, quat: tuple, vec_body: np.ndarray) -> np.ndarray:
        """Rotate vector from body frame to world frame."""
        w, x, y, z = quat
        qv = np.array([x, y, z])
        v = vec_body
        # v' = v + 2*w*(qv × v) + 2*(qv × (qv × v))
        cross1 = np.cross(qv, v)
        cross2 = np.cross(qv, cross1)
        return v + 2.0 * w * cross1 + 2.0 * cross2

    def _rotate_world_to_body(self, quat: tuple, vec_world: np.ndarray) -> np.ndarray:
        """Rotate vector from world frame to body frame."""
        w, x, y, z = quat
        # Conjugate for inverse rotation
        qv = np.array([-x, -y, -z])
        v = vec_world
        cross1 = np.cross(qv, v)
        cross2 = np.cross(qv, cross1)
        return v + 2.0 * w * cross1 + 2.0 * cross2
