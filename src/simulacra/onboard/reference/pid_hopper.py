"""Cascaded PID hopper controller — the canonical 6-DOF reference.

Three nested loops:
  Outer  (position P):   pos_error  →  desired velocity
  Middle (velocity PID): vel_error  →  desired acceleration  →  thrust vector
  Inner  (attitude PD):  attitude_error  →  gimbal commands

Adapted from the legacy `onboard/hopper_controller.py` to the new
Controller interface. Waypoint sequencing and arrival detection have
been removed — they live in `WaypointGuidance` now. This controller
just tracks `setpoint.position` and decides how to push the sticks.

Soft-landing behaviour: when the setpoint altitude is below
LAND_THRESHOLD_ALT and the rocket is also low and slow, thrust is
cut. This lets the rocket settle on the pad cleanly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from simulacra.onboard.controller import (
    ActuatorCommand, SensorFrame, VehicleParams,
)
from simulacra.onboard.guidance import Mission, Setpoint


LAND_THRESHOLD_ALT = 0.5    # m — setpoint altitude below which "landing intent"
LAND_CUT_ALT = 0.3          # m — actual altitude to cut thrust when landing
LAND_CUT_SPEED = 0.5        # m/s — speed below which to cut thrust when landing


@dataclass
class HopperConfig:
    """Gains and physical limits for the cascaded controller."""
    # Position P gains (outer loop)
    Kp_pos_xy: float = 0.8
    Kp_pos_z: float = 1.0
    max_vel_xy: float = 10.0
    max_vel_z_up: float = 4.0
    max_vel_z_down: float = 3.0

    # Velocity PID gains (middle loop)
    Kp_vel_xy: float = 1.5
    Ki_vel_xy: float = 0.3
    Kd_vel_xy: float = 0.1
    Kp_vel_z: float = 3.0
    Ki_vel_z: float = 1.0
    Kd_vel_z: float = 0.0

    # Anti-windup
    integral_limit_xy: float = 2.0
    integral_limit_z: float = 3.0

    # Thrust decomposition
    max_tilt: float = math.radians(35.0)
    a_decel: float = 3.0       # m/s^2 — braking deceleration for speed profile

    # Attitude PD (inner loop)
    att_Kp: float = 500.0
    att_Kd: float = 200.0
    max_gimbal_angle: float = math.radians(15.0)
    thrust_moment_arm: float = 2.0   # m

    gravity: float = 9.81


class PIDHopperController:
    """Cascaded PID, tracks Setpoint.position."""

    rate_hz: float = 50.0

    def __init__(self, config: HopperConfig | None = None) -> None:
        self.config = config or HopperConfig()
        self._mass: float = 1.0
        self._max_thrust: float = 1.0
        self._vel_integral = np.zeros(3)
        self._prev_vel_error = np.zeros(3)

    def reset(self, mission: Mission, vehicle: VehicleParams) -> None:
        # Mass approximation: dry + full fuel. The controller is robust
        # to mass changes during burn (PID handles the difference).
        self._mass = vehicle.dry_mass + vehicle.full_fuel_mass
        self._max_thrust = vehicle.max_thrust
        self._vel_integral = np.zeros(3)
        self._prev_vel_error = np.zeros(3)

    def step(
        self,
        frame: SensorFrame,
        setpoint: Setpoint,
        dt: float,
    ) -> ActuatorCommand:
        cfg = self.config

        # ── Read sensed state ──
        # GPS x/z + baro for altitude (more accurate vertical than GPS).
        if frame.gps is not None:
            gps_x, gps_z = float(frame.gps.value[0]), float(frame.gps.value[2])
        else:
            gps_x, gps_z = 0.0, 0.0
        alt = float(frame.baro.value)
        pos = np.array([gps_x, alt, gps_z])

        vel_vec = np.array([float(v) for v in frame.velocity.value])
        omega = np.array([float(w) for w in frame.imu_gyro.value])
        quat = tuple(float(q) for q in frame.orientation.value.q)

        target = np.array([float(setpoint.position[0]),
                           float(setpoint.position[1]),
                           float(setpoint.position[2])])

        # ── Landing intent: setpoint near ground + actually low + slow → cut ──
        speed = float(np.linalg.norm(vel_vec))
        if (target[1] < LAND_THRESHOLD_ALT
                and alt < LAND_CUT_ALT
                and speed < LAND_CUT_SPEED):
            return ActuatorCommand(throttle=0.0, gimbal_pitch=0.0, gimbal_yaw=0.0)

        # ── Outer loop: position P → desired velocity ──
        pos_error = target - pos
        dist = float(np.linalg.norm(pos_error))

        vel_desired = np.array([
            cfg.Kp_pos_xy * pos_error[0],
            cfg.Kp_pos_z * pos_error[1],
            cfg.Kp_pos_xy * pos_error[2],
        ])

        # Speed limit via braking-distance profile.
        v_limit_brake = math.sqrt(2.0 * cfg.a_decel * max(dist, 0.01))
        v_limit = min(cfg.max_vel_xy, v_limit_brake)

        vel_desired[0] = np.clip(vel_desired[0], -cfg.max_vel_xy, cfg.max_vel_xy)
        vel_desired[2] = np.clip(vel_desired[2], -cfg.max_vel_xy, cfg.max_vel_xy)
        if vel_desired[1] > 0:
            vel_desired[1] = min(vel_desired[1], cfg.max_vel_z_up)
        else:
            vel_desired[1] = max(vel_desired[1], -cfg.max_vel_z_down)

        # Soft descent near ground when landing intent active.
        if target[1] < LAND_THRESHOLD_ALT and alt < 5.0:
            soft_max = max(0.3, 0.3 * alt)
            vel_desired[1] = max(vel_desired[1], -soft_max)

        # Overall speed cap.
        speed_desired = float(np.linalg.norm(vel_desired))
        if speed_desired > v_limit:
            vel_desired *= v_limit / speed_desired

        # ── Middle loop: velocity PID → desired acceleration ──
        vel_error = vel_desired - vel_vec
        vel_error_deriv = (vel_error - self._prev_vel_error) / dt
        self._prev_vel_error = vel_error.copy()

        self._vel_integral += vel_error * dt
        self._vel_integral[0] = np.clip(
            self._vel_integral[0], -cfg.integral_limit_xy, cfg.integral_limit_xy)
        self._vel_integral[2] = np.clip(
            self._vel_integral[2], -cfg.integral_limit_xy, cfg.integral_limit_xy)
        self._vel_integral[1] = np.clip(
            self._vel_integral[1], -cfg.integral_limit_z, cfg.integral_limit_z)

        acc_cmd = np.array([
            cfg.Kp_vel_xy * vel_error[0] + cfg.Ki_vel_xy * self._vel_integral[0]
            + cfg.Kd_vel_xy * vel_error_deriv[0],
            cfg.Kp_vel_z * vel_error[1] + cfg.Ki_vel_z * self._vel_integral[1]
            + cfg.Kd_vel_z * vel_error_deriv[1],
            cfg.Kp_vel_xy * vel_error[2] + cfg.Ki_vel_xy * self._vel_integral[2]
            + cfg.Kd_vel_xy * vel_error_deriv[2],
        ])

        # Gravity feed-forward (world frame).
        acc_total = acc_cmd + np.array([0.0, cfg.gravity, 0.0])

        # ── Thrust decomposition (vertical-priority) ──
        F = self._mass * acc_total
        Fx, Fy, Fz = F
        Fy = float(np.clip(Fy, 0.0, self._max_thrust))

        Fh_from_tilt = Fy * math.tan(cfg.max_tilt) if Fy > 1.0 else 0.0
        Fh_from_thrust = math.sqrt(max(0.0, self._max_thrust ** 2 - Fy ** 2))
        Fh_budget = min(Fh_from_tilt, Fh_from_thrust)

        Fh = math.sqrt(Fx ** 2 + Fz ** 2)
        if Fh > Fh_budget and Fh > 0.01:
            scale = Fh_budget / Fh
            Fx *= scale
            Fz *= scale

        thrust = math.sqrt(Fx ** 2 + Fy ** 2 + Fz ** 2)
        thrust = min(thrust, self._max_thrust)

        if thrust > 1.0:
            thrust_dir = np.array([Fx, Fy, Fz]) / math.sqrt(Fx ** 2 + Fy ** 2 + Fz ** 2)
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

        return ActuatorCommand(
            throttle=thrust / self._max_thrust if self._max_thrust > 0 else 0.0,
            gimbal_pitch=outer_angle,
            gimbal_yaw=inner_angle,
        )


# ──────────────────────────────────────────────────────────────
# Quaternion rotation helpers (pure functions)
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
