"""
Hopper controller: cascaded PID trajectory tracker for VTVL rocket.

Three nested loops:
  Outer  (position P, 50 Hz):  pos_error → desired velocity
  Middle (velocity PID, 50 Hz): vel_error → desired accel → thrust vector
  Inner  (attitude PD, 50 Hz):  attitude error → gimbal commands

The controller follows a list of Waypoints, holding at each for a
specified duration before advancing to the next. A deceleration profile
ensures smooth arrivals.

Designed for the hopper test flight: takeoff, hover at various altitudes,
lateral movements, a circular orbit, and landing.
"""

import math
from dataclasses import dataclass, field
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


# ──────────────────────────────────────────────────────────────
# Waypoint definition
# ──────────────────────────────────────────────────────────────

@dataclass
class Waypoint:
    """A target position with a hold duration."""
    position: np.ndarray          # [x, y, z] world frame, meters
    hold_time: float = 5.0        # seconds to hover once arrived
    acceptance_radius: float = 1.0  # meters — "arrived" threshold
    max_speed: float = 5.0        # m/s approach speed limit


# ──────────────────────────────────────────────────────────────
# Controller config
# ──────────────────────────────────────────────────────────────

@dataclass
class HopperConfig:
    """Gains and physical parameters for the hopper controller."""

    # Rocket physical parameters
    dry_mass: float = 60.0        # kg
    full_fuel_mass: float = 80.0  # kg
    max_thrust: float = 2000.0    # N
    gravity: float = 9.81         # m/s^2

    # Position P gains (outer loop)
    Kp_pos_xy: float = 0.8       # 1/s — position error → desired velocity
    Kp_pos_z: float = 1.0        # 1/s — vertical
    max_vel_xy: float = 10.0     # m/s
    max_vel_z_up: float = 4.0    # m/s
    max_vel_z_down: float = 3.0  # m/s

    # Velocity PID gains (middle loop)
    Kp_vel_xy: float = 1.5       # m/s^2 per m/s error
    Ki_vel_xy: float = 0.3       # integral
    Kd_vel_xy: float = 0.1       # derivative
    Kp_vel_z: float = 3.0
    Ki_vel_z: float = 1.0
    Kd_vel_z: float = 0.0

    # Anti-windup limits (m/s^2)
    integral_limit_xy: float = 2.0
    integral_limit_z: float = 3.0

    # Thrust decomposition
    max_tilt: float = math.radians(35.0)  # maximum allowed tilt

    # Deceleration profile
    a_decel: float = 3.0         # m/s^2 — braking deceleration for speed limiting

    # Attitude PD gains (inner loop) — same as landing controller
    att_Kp: float = 500.0        # N·m/rad
    att_Kd: float = 200.0        # N·m/(rad/s)
    max_gimbal_angle: float = math.radians(15.0)
    thrust_moment_arm: float = 2.0  # m

    # Landing
    landing_altitude: float = 0.3  # m — cut thrust below this
    landing_speed: float = 0.5     # m/s — cut thrust below this speed


# ──────────────────────────────────────────────────────────────
# Controller
# ──────────────────────────────────────────────────────────────

class HopperController(Controller):
    """
    Cascaded PID controller for waypoint-following flight.

    Architecture follows PX4's position controller: P on position,
    PID on velocity (with gravity feedforward), PD on attitude.
    """

    def __init__(
        self,
        throttle: ThrottleActuator,
        gimbal: GimbalActuator,
        waypoints: list[Waypoint],
        config: Optional[HopperConfig] = None,
    ):
        self.throttle = throttle
        self.gimbal = gimbal
        self.waypoints = waypoints
        self.config = config or HopperConfig()
        self.sensors: Optional[Sensors] = None

        # Waypoint state
        self._wp_idx = 0
        self._wp_state = "TRACKING"  # or "HOLDING"
        self._hold_timer = 0.0

        # PID state
        self._vel_integral = np.zeros(3)
        self._prev_vel_error = np.zeros(3)
        self._prev_time = None

        # Telemetry (readable from outside)
        self.current_target = np.zeros(3)
        self.phase_name = ""

    def with_sensors(self, sensors: Sensors) -> "HopperController":
        self.sensors = sensors
        return self

    @property
    def finished(self) -> bool:
        """True when all waypoints have been visited."""
        return self._wp_idx >= len(self.waypoints)

    def run(self) -> None:
        if self.sensors is None:
            raise ValueError("Sensors not attached to controller")
        if self.finished:
            # All waypoints done — cut thrust
            self.throttle.apply(ThrottleCommand(thrust_N=0.0))
            self.gimbal.apply(GimbalCommand(desired_outer_angle=0.0, desired_inner_angle=0.0))
            return

        cfg = self.config
        dt = 0.02  # controller runs at 50 Hz (same as sim dt)

        # ── Read sensors ──
        gps = self.sensors.GPS.read()
        alt = self.sensors.Barometer.read()
        vel = self.sensors.VelocitySensor.read()
        orientation = self.sensors.OrientationSensor.read()
        omega_body = self.sensors.GyroSensor.read()
        fuel_frac = self.sensors.FuelSensor.read()

        pos = np.array([gps.x, alt, gps.z])
        vel_vec = np.array([vel.x, vel.y, vel.z])
        quat = tuple(orientation.as_array())  # (w, x, y, z)
        omega = np.array([omega_body.x, omega_body.y, omega_body.z])

        mass = cfg.dry_mass + fuel_frac * cfg.full_fuel_mass

        # ── Waypoint management ──
        target = self._get_target(pos, dt)
        self.current_target = target.copy()

        # Check if _get_target advanced past last waypoint
        if self.finished:
            self.throttle.apply(ThrottleCommand(thrust_N=0.0))
            self.gimbal.apply(GimbalCommand(desired_outer_angle=0.0, desired_inner_angle=0.0))
            return

        # Landing detection: near ground, slow, at final waypoint
        speed = np.linalg.norm(vel_vec)
        is_last_wp = (self._wp_idx == len(self.waypoints) - 1)
        if is_last_wp and alt < cfg.landing_altitude and speed < cfg.landing_speed:
            self.throttle.apply(ThrottleCommand(thrust_N=0.0))
            self.gimbal.apply(GimbalCommand(desired_outer_angle=0.0, desired_inner_angle=0.0))
            self._wp_idx = len(self.waypoints)  # mark finished
            return

        # ── Outer loop: position P → desired velocity ──
        pos_error = target - pos
        dist = np.linalg.norm(pos_error)

        vel_desired = np.array([
            cfg.Kp_pos_xy * pos_error[0],
            cfg.Kp_pos_z * pos_error[1],
            cfg.Kp_pos_xy * pos_error[2],
        ])

        # Speed limit: braking distance profile
        v_limit_brake = math.sqrt(2.0 * cfg.a_decel * max(dist, 0.01))
        wp = self.waypoints[self._wp_idx]
        v_limit = min(wp.max_speed, v_limit_brake)

        # Clamp velocity components
        vel_desired[0] = np.clip(vel_desired[0], -cfg.max_vel_xy, cfg.max_vel_xy)
        vel_desired[2] = np.clip(vel_desired[2], -cfg.max_vel_xy, cfg.max_vel_xy)
        if vel_desired[1] > 0:
            vel_desired[1] = min(vel_desired[1], cfg.max_vel_z_up)
        else:
            vel_desired[1] = max(vel_desired[1], -cfg.max_vel_z_down)

        # Soft landing: tighten descent speed near ground for the last waypoint
        if is_last_wp and alt < 5.0:
            soft_max = max(0.3, 0.3 * alt)
            vel_desired[1] = max(vel_desired[1], -soft_max)

        # Overall speed limit
        speed_desired = np.linalg.norm(vel_desired)
        if speed_desired > v_limit:
            vel_desired *= v_limit / speed_desired

        # ── Middle loop: velocity PID → desired acceleration ──
        vel_error = vel_desired - vel_vec

        # Derivative (finite difference)
        vel_error_deriv = (vel_error - self._prev_vel_error) / dt
        self._prev_vel_error = vel_error.copy()

        # Integral with anti-windup
        self._vel_integral += vel_error * dt
        self._vel_integral[0] = np.clip(
            self._vel_integral[0], -cfg.integral_limit_xy, cfg.integral_limit_xy)
        self._vel_integral[2] = np.clip(
            self._vel_integral[2], -cfg.integral_limit_xy, cfg.integral_limit_xy)
        self._vel_integral[1] = np.clip(
            self._vel_integral[1], -cfg.integral_limit_z, cfg.integral_limit_z)

        # PID acceleration command (world frame)
        acc_cmd = np.array([
            cfg.Kp_vel_xy * vel_error[0] + cfg.Ki_vel_xy * self._vel_integral[0]
            + cfg.Kd_vel_xy * vel_error_deriv[0],
            cfg.Kp_vel_z * vel_error[1] + cfg.Ki_vel_z * self._vel_integral[1]
            + cfg.Kd_vel_z * vel_error_deriv[1],
            cfg.Kp_vel_xy * vel_error[2] + cfg.Ki_vel_xy * self._vel_integral[2]
            + cfg.Kd_vel_xy * vel_error_deriv[2],
        ])

        # Gravity feedforward
        acc_total = acc_cmd + np.array([0.0, cfg.gravity, 0.0])

        # ── Thrust decomposition (PX4 vertical-priority method) ──
        F = mass * acc_total
        Fx, Fy, Fz = F

        # Vertical priority: clamp Fy to [0, max_thrust]
        Fy = np.clip(Fy, 0.0, cfg.max_thrust)

        # Horizontal budget
        Fh_from_tilt = Fy * math.tan(cfg.max_tilt) if Fy > 1.0 else 0.0
        Fh_from_thrust = math.sqrt(max(0.0, cfg.max_thrust**2 - Fy**2))
        Fh_budget = min(Fh_from_tilt, Fh_from_thrust)

        Fh = math.sqrt(Fx**2 + Fz**2)
        if Fh > Fh_budget and Fh > 0.01:
            scale = Fh_budget / Fh
            Fx *= scale
            Fz *= scale

        thrust = math.sqrt(Fx**2 + Fy**2 + Fz**2)
        thrust = min(thrust, cfg.max_thrust)

        if thrust > 1.0:
            thrust_dir = np.array([Fx, Fy, Fz]) / math.sqrt(Fx**2 + Fy**2 + Fz**2)
        else:
            thrust_dir = np.array([0.0, 1.0, 0.0])

        # ── Inner loop: attitude PD → gimbal commands ──
        nose_world = _rotate_body_to_world(quat, np.array([1.0, 0.0, 0.0]))
        error_world = np.cross(nose_world, thrust_dir)
        error_body = _rotate_world_to_body(quat, error_world)

        tau = cfg.att_Kp * error_body - cfg.att_Kd * omega

        if thrust > 100.0:
            scale = 1.0 / (thrust * cfg.thrust_moment_arm)
            outer_angle = float(np.clip(-tau[2] * scale,
                                        -cfg.max_gimbal_angle, cfg.max_gimbal_angle))
            inner_angle = float(np.clip(-tau[1] * scale,
                                        -cfg.max_gimbal_angle, cfg.max_gimbal_angle))
        else:
            outer_angle = 0.0
            inner_angle = 0.0

        # ── Apply commands ──
        self.throttle.apply(ThrottleCommand(thrust_N=thrust))
        self.gimbal.apply(GimbalCommand(
            desired_outer_angle=outer_angle,
            desired_inner_angle=inner_angle,
        ))

    def _get_target(self, pos: np.ndarray, dt: float) -> np.ndarray:
        """Get current target position, advancing waypoints as needed."""
        wp = self.waypoints[self._wp_idx]

        if self._wp_state == "TRACKING":
            dist = np.linalg.norm(wp.position - pos)
            if dist < wp.acceptance_radius:
                self._wp_state = "HOLDING"
                self._hold_timer = 0.0
                self.phase_name = f"HOLD wp{self._wp_idx} ({wp.hold_time:.0f}s)"
            else:
                self.phase_name = f"TRACK wp{self._wp_idx} ({dist:.1f}m)"

        if self._wp_state == "HOLDING":
            self._hold_timer += dt
            if self._hold_timer >= wp.hold_time:
                self._wp_idx += 1
                self._wp_state = "TRACKING"
                if self._wp_idx < len(self.waypoints):
                    next_wp = self.waypoints[self._wp_idx]
                    self.phase_name = f"TRACK wp{self._wp_idx}"
                else:
                    self.phase_name = "DONE"

        if self._wp_idx < len(self.waypoints):
            return self.waypoints[self._wp_idx].position.copy()
        else:
            return wp.position.copy()  # stay at last waypoint


# ──────────────────────────────────────────────────────────────
# Rotation helpers (pure functions, match landing_controller.py)
# ──────────────────────────────────────────────────────────────

def _rotate_body_to_world(quat: tuple, vec_body: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    qv = np.array([x, y, z])
    c1 = np.cross(qv, vec_body)
    c2 = np.cross(qv, c1)
    return vec_body + 2.0 * w * c1 + 2.0 * c2


def _rotate_world_to_body(quat: tuple, vec_world: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    qv = np.array([-x, -y, -z])
    c1 = np.cross(qv, vec_world)
    c2 = np.cross(qv, c1)
    return vec_world + 2.0 * w * c1 + 2.0 * c2


# ──────────────────────────────────────────────────────────────
# Hopper flight plan builder
# ──────────────────────────────────────────────────────────────

def hopper_flight_plan(circle_points: int = 24) -> list[Waypoint]:
    """
    Build the standard hopper test flight waypoint sequence.

    Phases:
     1. Liftoff → 5m hover
     2. Climb → 10m hover
     3. Descend → 5m hover
     4. Climb → 20m hover
     5. Lateral +X 5m
     6. Return center
     7. Lateral -X 5m
     8. Return center
     9. Circle (5m radius at 20m alt)
    10. Return center
    11. Land
    """
    wps = []

    # Phase 1-4: vertical maneuvers
    wps.append(Waypoint(np.array([0., 5., 0.]), hold_time=5.0, max_speed=2.0,
                        acceptance_radius=0.5))
    wps.append(Waypoint(np.array([0., 10., 0.]), hold_time=5.0, max_speed=3.0,
                        acceptance_radius=0.5))
    wps.append(Waypoint(np.array([0., 5., 0.]), hold_time=5.0, max_speed=2.0,
                        acceptance_radius=0.5))
    wps.append(Waypoint(np.array([0., 20., 0.]), hold_time=5.0, max_speed=3.0,
                        acceptance_radius=1.0))

    # Phase 5-8: lateral maneuvers at 20m
    wps.append(Waypoint(np.array([5., 20., 0.]), hold_time=5.0, max_speed=3.0,
                        acceptance_radius=1.0))
    wps.append(Waypoint(np.array([0., 20., 0.]), hold_time=3.0, max_speed=3.0,
                        acceptance_radius=1.0))
    wps.append(Waypoint(np.array([-5., 20., 0.]), hold_time=5.0, max_speed=3.0,
                        acceptance_radius=1.0))
    wps.append(Waypoint(np.array([0., 20., 0.]), hold_time=3.0, max_speed=3.0,
                        acceptance_radius=1.0))

    # Phase 9: circle at 20m altitude, 5m radius
    radius = 5.0
    alt = 20.0
    for i in range(circle_points):
        angle = 2.0 * math.pi * i / circle_points
        x = radius * math.cos(angle)
        z = radius * math.sin(angle)
        wps.append(Waypoint(
            np.array([x, alt, z]),
            hold_time=0.5,  # brief hold at each point
            max_speed=3.0,
            acceptance_radius=1.5,  # looser — smooth orbit
        ))

    # Phase 10: return to center
    wps.append(Waypoint(np.array([0., 20., 0.]), hold_time=3.0, max_speed=3.0,
                        acceptance_radius=1.0))

    # Phase 11: land — target y=0.0 so the soft-landing code guides the rocket
    # down through the landing_altitude threshold (0.3m), triggering detection.
    wps.append(Waypoint(np.array([0., 0.0, 0.]), hold_time=60.0, max_speed=1.0,
                        acceptance_radius=0.5))

    return wps
