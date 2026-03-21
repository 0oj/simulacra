"""
G-FOLD trajectory-tracking controller for the hopper.

Solves for the fuel-optimal trajectory at startup using G-FOLD (SOCP),
then tracks it with closed-loop feedback: the SOCP provides the reference
trajectory, and a PD position/velocity controller corrects for tracking
error. The inner PD attitude loop converts desired thrust direction to
gimbal commands.

This gives us the "best possible classical controller" benchmark:
optimal trajectory + feedback tracking + proven attitude control.

Usage:
    from simulacra.onboard.gfold_controller import GFOLDHopperController
    controller = GFOLDHopperController(
        throttle=actuators._throttle,
        gimbal=actuators._gimbal,
        waypoints=waypoints,
    )
"""

import math
from typing import Optional

import numpy as np

from ..gfold import GFOLDSolver, GFOLDResult, HopperParams
from ..phenomena.actuators import (
    ThrottleActuator,
    ThrottleCommand,
    GimbalActuator,
    GimbalCommand,
)
from ..phenomena.sensors import Sensors
from .controller import Controller
from .hopper_controller import Waypoint, HopperConfig


class GFOLDHopperController(Controller):
    """Trajectory-tracking controller using G-FOLD + PD feedback + PD attitude.

    Architecture (three nested loops):
      1. G-FOLD SOCP: computes fuel-optimal reference trajectory (offline, once)
      2. PD position/velocity tracking: corrects thrust acceleration for
         deviations from the reference (outer loop, ~50Hz)
      3. PD attitude controller: converts desired thrust direction to gimbal
         commands (inner loop, ~50Hz)

    Same external interface as HopperController and RLHopperController.
    """

    # Inner PD attitude gains (identical to GuidanceWrapper / RLHopperController)
    ATT_KP = 500.0
    ATT_KD = 200.0
    MAX_THRUST = 2000.0
    MOMENT_ARM = 2.0
    MAX_GIMBAL = 0.26

    # Outer PD tracking gains
    TRACK_KP = 1.5   # position error → acceleration correction (1/s²)
    TRACK_KD = 2.5   # velocity error → acceleration correction (1/s)

    def __init__(
        self,
        throttle: ThrottleActuator,
        gimbal: GimbalActuator,
        waypoints: list[Waypoint],
        config: Optional[HopperConfig] = None,
        max_tilt: float = 0.25,
        gfold_N: int = 80,
    ):
        self.throttle = throttle
        self.gimbal = gimbal
        self.waypoints = waypoints
        self.config = config or HopperConfig()
        self.max_tilt = max_tilt
        self.gfold_N = gfold_N
        self.sensors: Optional[Sensors] = None

        # Trajectory state (computed on first run)
        self._trajectory_solved = False
        self._result: Optional[GFOLDResult] = None
        self._flight_time = 0.0
        self._solve_info: Optional[dict] = None

        # Step tracking
        self._step = 0
        self._t = 0.0
        self._dt = 0.02
        self._landed = False

        # Telemetry (same interface as other controllers)
        self.current_target = np.zeros(3)
        self.phase_name = "GFOLD_INIT"

    def with_sensors(self, sensors: Sensors) -> "GFOLDHopperController":
        self.sensors = sensors
        return self

    @property
    def finished(self) -> bool:
        return self._landed

    def run(self) -> None:
        if self.sensors is None:
            raise ValueError("Sensors not attached to controller")

        dt = self._dt

        # ── Read sensors ──
        gps = self.sensors.GPS.read()
        alt = self.sensors.Barometer.read()
        vel = self.sensors.VelocitySensor.read()
        orientation = self.sensors.OrientationSensor.read()
        omega_body = self.sensors.GyroSensor.read()

        pos = np.array([gps.x, alt, gps.z])
        vel_vec = np.array([vel.x, vel.y, vel.z])
        quat = orientation.as_array()  # [w, x, y, z]
        omega = np.array([omega_body.x, omega_body.y, omega_body.z])

        # ── Solve trajectory on first call ──
        if not self._trajectory_solved:
            self._solve_trajectory(pos, vel_vec)

        # ── Landing detection ──
        speed = np.linalg.norm(vel_vec)
        cfg = self.config
        past_midpoint = self._t > self._flight_time * 0.5
        if past_midpoint and alt < cfg.landing_altitude and speed < cfg.landing_speed:
            self.phase_name = "GFOLD_LANDED"
            self.throttle.apply(ThrottleCommand(thrust_N=0.0))
            self.gimbal.apply(GimbalCommand(desired_outer_angle=0.0, desired_inner_angle=0.0))
            self._landed = True
            return

        # ── Compute desired thrust acceleration with feedback ──
        u_desired = self._compute_tracking_accel(pos, vel_vec)

        # Convert acceleration to throttle and tilt
        thrust_mag = np.linalg.norm(u_desired)
        mass_est = cfg.dry_mass + cfg.full_fuel_mass * 0.5  # rough mass estimate
        thrust_n = np.clip(thrust_mag * mass_est, 0.0, self.MAX_THRUST)
        throttle_cmd = thrust_n / self.MAX_THRUST

        if thrust_mag > 0.1 and u_desired[1] > 0.1:
            tilt_x = u_desired[0] / u_desired[1]
            tilt_z = u_desired[2] / u_desired[1]
            tilt_x = np.clip(tilt_x, -self.max_tilt, self.max_tilt)
            tilt_z = np.clip(tilt_z, -self.max_tilt, self.max_tilt)
        else:
            tilt_x = 0.0
            tilt_z = 0.0

        self.phase_name = f"GFOLD_TRACK t={self._t:.1f}/{self._flight_time:.1f}s"
        self.current_target = self.waypoints[-1].position.copy() if self.waypoints else np.zeros(3)

        # ── Inner PD attitude controller (same as RLHopperController) ──
        desired_dir = np.array([tilt_x, 1.0, tilt_z])
        desired_dir = desired_dir / np.linalg.norm(desired_dir)

        qw, qx, qy, qz = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        nose_world = _rotate_body_to_world(qw, qx, qy, qz, [1, 0, 0])
        error_world = np.cross(nose_world, desired_dir)
        error_body = _rotate_world_to_body(qw, qx, qy, qz, error_world)

        tau = self.ATT_KP * error_body - self.ATT_KD * omega

        if thrust_n > 100:
            scale = 1.0 / (thrust_n * self.MOMENT_ARM)
            outer = float(np.clip(-tau[2] * scale, -self.MAX_GIMBAL, self.MAX_GIMBAL))
            inner = float(np.clip(-tau[1] * scale, -self.MAX_GIMBAL, self.MAX_GIMBAL))
        else:
            outer = 0.0
            inner = 0.0

        # ── Apply commands ──
        self.throttle.apply(ThrottleCommand(thrust_N=thrust_n))
        self.gimbal.apply(GimbalCommand(
            desired_outer_angle=outer,
            desired_inner_angle=inner,
        ))

        self._t += dt
        self._step += 1

    def _compute_tracking_accel(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Compute desired thrust acceleration with closed-loop feedback.

        u_desired = u_reference + Kp * (r_ref - r_actual) + Kd * (v_ref - v_actual) + g_compensation

        During the trajectory, tracks the reference. After trajectory ends,
        guides the rocket to the target with gentle descent.
        """
        cfg = self.config
        g_vec = np.array([0.0, cfg.gravity, 0.0])  # gravity compensation (upward)
        target = self.waypoints[-1].position if self.waypoints else np.zeros(3)

        if self._t < self._flight_time and self._result is not None:
            # ── During trajectory: feedforward + feedback ──
            r_ref = self._interp_vec(self._result.positions, self._t)
            v_ref = self._interp_vec(self._result.velocities, self._t)
            u_ref = self._interp_vec(self._result.thrust_accel, self._t)

            e_pos = r_ref - pos
            e_vel = v_ref - vel

            u_desired = u_ref + g_vec + self.TRACK_KP * e_pos + self.TRACK_KD * e_vel
        else:
            # ── After trajectory: PD guidance to target ──
            e_pos = target - pos
            e_vel = -vel  # want zero velocity

            u_desired = g_vec + self.TRACK_KP * e_pos + self.TRACK_KD * e_vel

        # Clamp magnitude to physical limits
        u_mag = np.linalg.norm(u_desired)
        mass_est = cfg.dry_mass + cfg.full_fuel_mass * 0.5
        max_accel = cfg.max_thrust / mass_est
        if u_mag > max_accel:
            u_desired = u_desired * (max_accel / u_mag)

        return u_desired

    def _interp_vec(self, data: np.ndarray, t: float) -> np.ndarray:
        """Interpolate a (N, 3) trajectory at time t."""
        times = self._result.times
        if t <= times[0]:
            return data[0]
        if t >= times[-1]:
            return data[-1]
        idx = np.searchsorted(times, t) - 1
        idx = max(0, min(idx, len(times) - 2))
        alpha = (t - times[idx]) / (times[idx + 1] - times[idx])
        return (1 - alpha) * data[idx] + alpha * data[idx + 1]

    def _solve_trajectory(self, pos: np.ndarray, vel: np.ndarray) -> None:
        """Solve G-FOLD from current state to final waypoint."""
        target = self.waypoints[-1].position if self.waypoints else np.zeros(3)

        params = HopperParams(
            dry_mass=self.config.dry_mass,
            fuel_mass=self.config.full_fuel_mass,
            max_thrust=self.config.max_thrust,
            gravity=self.config.gravity,
            max_tilt=self.max_tilt,
        )

        solver = GFOLDSolver(params)

        # Disable glideslope for near-ground starts
        gs_deg = 30.0 if pos[1] > 2.0 else 0.0

        result = solver.solve(
            r0=pos, v0=vel, rf=target, vf=np.zeros(3),
            N=self.gfold_N, glideslope_deg=gs_deg,
        )

        if result.status in ('optimal', 'optimal_inaccurate'):
            self._result = result
            self._flight_time = result.flight_time
            self._solve_info = {
                'status': result.status,
                'fuel_predicted': result.fuel_used,
                'solve_time_ms': result.solve_time * 1000,
                'smoothness': result.smoothness,
            }
            self.phase_name = f"GFOLD_TRACK (solved in {result.solve_time*1000:.0f}ms)"
        else:
            # Fallback: no reference trajectory, PD guidance only
            self._result = None
            self._flight_time = 0.0
            self._solve_info = {'status': result.status, 'fallback': True}
            self.phase_name = f"GFOLD_FALLBACK ({result.status})"

        self._trajectory_solved = True


# ──────────────────────────────────────────────────────────────
# Rotation helpers (same as rl_hopper_controller.py)
# ──────────────────────────────────────────────────────────────

def _rotate_body_to_world(qw, qx, qy, qz, vec_body):
    qv = np.array([qx, qy, qz])
    v = np.asarray(vec_body, dtype=np.float64)
    c1 = np.cross(qv, v)
    c2 = np.cross(qv, c1)
    return v + 2.0 * qw * c1 + 2.0 * c2


def _rotate_world_to_body(qw, qx, qy, qz, vec_world):
    qv = np.array([-qx, -qy, -qz])
    v = np.asarray(vec_world, dtype=np.float64)
    c1 = np.cross(qv, v)
    c2 = np.cross(qv, c1)
    return v + 2.0 * qw * c1 + 2.0 * c2
