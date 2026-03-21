"""
RL Hopper Controller: drop-in replacement for HopperController using a trained policy.

Loads a trained SB3 PPO model and runs it through the same sensor/actuator interface
as HopperController. Waypoint sequencing (hold timers, acceptance radius, advancement)
is handled identically to the PID version.

The policy expects 22-dim (or 28-dim with safety envelope) observations
[state_13 + goal_rel_3 + features_3 + mode_3 + (safety_6)]
and outputs 3-dim actions [throttle, tilt_x, tilt_z]. An inner PD controller (same as GuidanceWrapper)
converts tilt commands to gimbal angles.

Usage:
    from simulacra.onboard.rl_hopper_controller import RLHopperController
    controller = RLHopperController(
        throttle=actuators._throttle,
        gimbal=actuators._gimbal,
        waypoints=waypoints,
        model_path="runs/hopper_rl/final_model",
    )
"""

import math
from pathlib import Path
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
from .hopper_controller import Waypoint, HopperConfig
from .safety_envelope import SafetyEnvelope


class RLHopperController(Controller):
    """
    RL policy wrapped with waypoint sequencing and PD attitude control.

    Same external interface as HopperController:
      - .current_target, .phase_name, .finished for telemetry
      - .run() reads sensors and applies actuator commands
      - .with_sensors() attaches sensor suite
    """

    # Inner PD attitude gains (identical to GuidanceWrapper)
    ATT_KP = 500.0
    ATT_KD = 200.0
    MAX_THRUST = 2000.0
    MOMENT_ARM = 2.0
    MAX_GIMBAL = 0.26

    def __init__(
        self,
        throttle: ThrottleActuator,
        gimbal: GimbalActuator,
        waypoints: list[Waypoint],
        model_path: str | Path,
        config: Optional[HopperConfig] = None,
        deterministic: bool = True,
        max_tilt: float = 0.15,
        safety_envelope: Optional[SafetyEnvelope] = None,
    ):
        self.throttle = throttle
        self.gimbal = gimbal
        self.waypoints = waypoints
        self.safety_envelope = safety_envelope
        self.config = config or HopperConfig()
        self.deterministic = deterministic
        self.max_tilt = max_tilt
        self.sensors: Optional[Sensors] = None

        # Load trained model (SB3 .zip or JAX checkpoint dir)
        model_path = Path(model_path)
        if (model_path / "params.npz").exists():
            # JAX checkpoint — use lightweight numpy inference
            self._load_jax_model(model_path)
        else:
            from stable_baselines3 import PPO
            self.model = PPO.load(str(model_path))
            self._jax_params = None

        # Waypoint state (same as HopperController)
        self._wp_idx = 0
        self._wp_state = "TRACKING"
        self._hold_timer = 0.0

        # Telemetry
        self.current_target = np.zeros(3)
        self.phase_name = ""

    def _load_jax_model(self, checkpoint_dir: Path):
        """Load JAX checkpoint as plain numpy weights for CPU inference."""
        data = np.load(checkpoint_dir / "params.npz")
        # Normalize key names: flax saves as "['Dense_0']/['kernel']"
        self._jax_params = {}
        for k in data.files:
            clean = k.replace("['", "").replace("']/", "/").replace("']", "")
            self._jax_params[clean] = data[k]
        self.model = None  # no SB3 model

    def _jax_predict(self, obs: np.ndarray) -> np.ndarray:
        """Forward pass through actor network using numpy (no JAX needed)."""
        p = self._jax_params
        # Actor: Dense_0 -> tanh -> Dense_1 -> tanh -> Dense_2 (action mean)
        x = obs.astype(np.float32)
        x = np.tanh(x @ p["Dense_0/kernel"] + p["Dense_0/bias"])
        x = np.tanh(x @ p["Dense_1/kernel"] + p["Dense_1/bias"])
        action_mean = x @ p["Dense_2/kernel"] + p["Dense_2/bias"]
        # Clip: throttle [0,1], tilt [-1,1]
        action_mean[0] = np.clip(action_mean[0], 0.0, 1.0)
        action_mean[1:] = np.clip(action_mean[1:], -1.0, 1.0)
        return action_mean

    def with_sensors(self, sensors: Sensors) -> "RLHopperController":
        self.sensors = sensors
        return self

    @property
    def finished(self) -> bool:
        return self._wp_idx >= len(self.waypoints)

    def run(self) -> None:
        if self.sensors is None:
            raise ValueError("Sensors not attached to controller")
        if self.finished:
            self.throttle.apply(ThrottleCommand(thrust_N=0.0))
            self.gimbal.apply(GimbalCommand(desired_outer_angle=0.0, desired_inner_angle=0.0))
            return

        dt = 0.02

        # ── Read sensors ──
        gps = self.sensors.GPS.read()
        alt = self.sensors.Barometer.read()
        vel = self.sensors.VelocitySensor.read()
        orientation = self.sensors.OrientationSensor.read()
        omega_body = self.sensors.GyroSensor.read()
        fuel_frac = self.sensors.FuelSensor.read()

        pos = np.array([gps.x, alt, gps.z])
        vel_vec = np.array([vel.x, vel.y, vel.z])
        quat = orientation.as_array()  # [w, x, y, z]
        omega = np.array([omega_body.x, omega_body.y, omega_body.z])

        # ── Waypoint management (same logic as HopperController) ──
        target = self._get_target(pos, dt)
        self.current_target = target.copy()

        if self.finished:
            self.throttle.apply(ThrottleCommand(thrust_N=0.0))
            self.gimbal.apply(GimbalCommand(desired_outer_angle=0.0, desired_inner_angle=0.0))
            return

        # Landing detection (same as PID controller)
        speed = np.linalg.norm(vel_vec)
        cfg = self.config
        is_last_wp = (self._wp_idx == len(self.waypoints) - 1)
        if is_last_wp and alt < cfg.landing_altitude and speed < cfg.landing_speed:
            self.throttle.apply(ThrottleCommand(thrust_N=0.0))
            self.gimbal.apply(GimbalCommand(desired_outer_angle=0.0, desired_inner_angle=0.0))
            self._wp_idx = len(self.waypoints)
            return

        # ── Build observation (22-dim) ──
        # Base state: [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
        state = np.array([
            gps.x, alt, gps.z,
            vel.x, vel.y, vel.z,
            quat[0], quat[1], quat[2], quat[3],
            omega[0], omega[1], omega[2],
        ], dtype=np.float32)

        # Goal-relative: [dx, dy, dz]
        goal_rel = (target - pos).astype(np.float32)

        # Features
        dist_to_wp = float(np.linalg.norm(goal_rel))
        if dist_to_wp > 0.01:
            dir_to_wp = goal_rel / dist_to_wp
            vel_toward = float(np.dot(vel_vec, dir_to_wp))
        else:
            vel_toward = 0.0

        g = cfg.gravity
        total_mass = cfg.dry_mass + fuel_frac * cfg.full_fuel_mass
        a_max = max(cfg.max_thrust / total_mass - g, 1.0)
        brake_margin = dist_to_wp - speed ** 2 / (2 * a_max)

        features = np.array([dist_to_wp, vel_toward, brake_margin], dtype=np.float32)

        # Flight mode one-hot: [TRACK, LAND, LIFTOFF]
        is_land = target[1] < 1.0
        on_ground = alt < 0.5 and speed < 0.5
        is_liftoff = on_ground and not is_land
        is_track = not is_land and not is_liftoff
        mode = np.array([float(is_track), float(is_land), float(is_liftoff)], dtype=np.float32)

        obs = np.concatenate([state, goal_rel, features, mode])

        # ── Safety envelope (if attached) ──
        if self.safety_envelope is not None:
            a_max_env = max(cfg.max_thrust / total_mass - g, 1.0)
            v_safe = math.sqrt(2 * a_max_env * max(0.1, dist_to_wp))

            # Tilt from quaternion: R[1,0] = 2*(qx*qy + qw*qz)
            nose_y = 2 * (quat[1] * quat[2] + quat[0] * quat[3])
            tilt = math.acos(max(-1.0, min(1.0, nose_y)))

            self.safety_envelope.update_state(
                speed=speed, v_safe=v_safe, tilt=tilt,
                max_tilt=1.0, vy=float(vel.y), alt=alt,
            )
            safety_signals = self.safety_envelope.read()
            obs = np.concatenate([obs, safety_signals])

        # ── Run policy ──
        if self._jax_params is not None:
            action = self._jax_predict(obs)
        else:
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
        throttle_cmd = float(action[0])
        tilt_x = float(action[1]) * self.max_tilt
        tilt_z = float(action[2]) * self.max_tilt

        # ── Inner PD attitude controller (identical to GuidanceWrapper) ──
        desired_dir = np.array([tilt_x, 1.0, tilt_z])
        desired_dir = desired_dir / np.linalg.norm(desired_dir)

        qw, qx, qy, qz = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        nose_world = _rotate_body_to_world(qw, qx, qy, qz, [1, 0, 0])
        error_world = np.cross(nose_world, desired_dir)
        error_body = _rotate_world_to_body(qw, qx, qy, qz, error_world)

        tau = self.ATT_KP * error_body - self.ATT_KD * omega
        thrust_n = throttle_cmd * self.MAX_THRUST

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

    def _get_target(self, pos: np.ndarray, dt: float) -> np.ndarray:
        """Same waypoint sequencing as HopperController."""
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
                    self.phase_name = f"TRACK wp{self._wp_idx}"
                else:
                    self.phase_name = "DONE"

        if self._wp_idx < len(self.waypoints):
            return self.waypoints[self._wp_idx].position.copy()
        else:
            return wp.position.copy()


# ──────────────────────────────────────────────────────────────
# Rotation helpers (same as hopper_controller.py / GuidanceWrapper)
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
