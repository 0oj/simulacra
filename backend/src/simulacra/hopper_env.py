"""
Goal-conditioned waypoint tracking environment for VTVL rocket hopper.

Each episode: navigate from a random start state to a single target waypoint.
The policy learns a universal "go to position" skill. Waypoint sequencing
(hold timers, acceptance radius, advancement) is managed externally at
inference time by the RLHopperController.

Observation (22-dim):
    [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz,   # base state (13)
     dx, dy, dz,                                            # goal-relative (3)
     dist_to_wp, vel_toward, brake_margin_wp,               # features (3)
     is_track, is_land, is_liftoff]                         # flight mode (3)

Action (3-dim): [throttle, gimbal_outer, gimbal_inner]
    Designed to be wrapped with GuidanceWrapper for [throttle, tilt_x, tilt_z].

Usage:
    env = HopperTrackingEnv(scenarios=["vertical"], max_episode_steps=500)
    env = GuidanceWrapper(env)   # inner PD for attitude
    env = Monitor(env)
"""

import math
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

from .noumena import (
    Kernel,
    Rocket,
    RocketState,
    Environment,
    MassModel,
    get_upright_orientation,
)
from .noumena.mass_model import MassComponent, FuelComponent
from .phenomena.actuators import (
    Actuators,
    GimbalActuator,
    GimbalCommand,
    ThrottleActuator,
    ThrottleCommand,
)
from .phenomena.sensors import Sensors
from .phenomena.errors import GaussianNoise, Bias, CompositeError
from .physics import Orientation, RocketVec3, WorldVec3
from .gym_env import DomainRandomization
from .schema import (
    MetadataMessage, ConstantMessage, VariableMessage, DebugMessage,
    MLDebugMessage, collect, write_message,
)


# ──────────────────────────────────────────────────────────────
# Scenario generators
# ──────────────────────────────────────────────────────────────

def vertical_scenario(rng, alt_range=(5.0, 30.0), vel_range=(-1.0, 1.0)):
    """Vertical transitions: target directly above or below start."""
    start_y = rng.uniform(0.0, alt_range[1])
    target_y = rng.uniform(alt_range[0], alt_range[1])
    # Ensure meaningful altitude difference
    while abs(target_y - start_y) < 3.0:
        target_y = rng.uniform(alt_range[0], alt_range[1])
    start_vel = np.array([0.0, rng.uniform(*vel_range), 0.0])
    return np.array([0.0, start_y, 0.0]), start_vel, np.array([0.0, target_y, 0.0])


def lateral_scenario(rng, alt_range=(10.0, 30.0), horiz_range=(-15.0, 15.0),
                     vel_range=(-1.0, 1.0)):
    """Horizontal move at roughly constant altitude."""
    alt = rng.uniform(*alt_range)
    start_x = rng.uniform(horiz_range[0] * 0.5, horiz_range[1] * 0.5)
    start_z = rng.uniform(horiz_range[0] * 0.5, horiz_range[1] * 0.5)
    target_x = rng.uniform(*horiz_range)
    target_z = rng.uniform(*horiz_range)
    while math.sqrt((target_x - start_x)**2 + (target_z - start_z)**2) < 3.0:
        target_x = rng.uniform(*horiz_range)
        target_z = rng.uniform(*horiz_range)
    vx = rng.uniform(*vel_range)
    vz = rng.uniform(*vel_range)
    return (np.array([start_x, alt, start_z]),
            np.array([vx, 0.0, vz]),
            np.array([target_x, alt, target_z]))


def diagonal_scenario(rng, alt_range=(5.0, 40.0), horiz_range=(-15.0, 15.0),
                      vel_range=(-1.5, 1.5)):
    """Combined 3D transition."""
    start = np.array([
        rng.uniform(horiz_range[0] * 0.5, horiz_range[1] * 0.5),
        rng.uniform(2.0, alt_range[1]),
        rng.uniform(horiz_range[0] * 0.5, horiz_range[1] * 0.5),
    ])
    target = np.array([
        rng.uniform(*horiz_range),
        rng.uniform(*alt_range),
        rng.uniform(*horiz_range),
    ])
    while np.linalg.norm(target - start) < 5.0:
        target[1] = rng.uniform(*alt_range)
    vel = np.array([rng.uniform(*vel_range), rng.uniform(*vel_range),
                    rng.uniform(*vel_range)])
    return start, vel, target


def landing_scenario(rng, alt_range=(10.0, 40.0), horiz_range=(-10.0, 10.0),
                     vel_range=(-2.0, 0.0)):
    """Approach to ground (target near y=0)."""
    start = np.array([
        rng.uniform(*horiz_range),
        rng.uniform(*alt_range),
        rng.uniform(*horiz_range),
    ])
    target = np.array([0.0, 0.0, 0.0])  # land at ground level
    vy = rng.uniform(*vel_range)
    return start, np.array([0.0, vy, 0.0]), target


def liftoff_scenario(rng, alt_range=(5.0, 25.0), horiz_range=(-5.0, 5.0)):
    """Take off from ground to target altitude."""
    target = np.array([
        rng.uniform(*horiz_range),
        rng.uniform(*alt_range),
        rng.uniform(*horiz_range),
    ])
    return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), target


def vertical_micro_scenario(rng, alt_range=(1.0, 5.0), vel_range=(-0.5, 0.5)):
    """Precision micro vertical moves (1-5m)."""
    return vertical_scenario(rng, alt_range=alt_range, vel_range=vel_range)


def lateral_short_scenario(rng, alt_range=(10.0, 20.0), horiz_range=(-8.0, 8.0),
                            vel_range=(-0.5, 0.5)):
    """Precision short lateral moves at fixed altitude."""
    return lateral_scenario(rng, alt_range=alt_range, horiz_range=horiz_range,
                            vel_range=vel_range)


def landing_precision_scenario(rng, alt_range=(5.0, 15.0), horiz_range=(-3.0, 3.0),
                                vel_range=(-1.0, 0.0)):
    """Tight precision landing from low altitude."""
    return landing_scenario(rng, alt_range=alt_range, horiz_range=horiz_range,
                            vel_range=vel_range)


def landing_high_scenario(rng, alt_range=(50.0, 100.0), horiz_range=(-15.0, 15.0),
                           vel_range=(-5.0, 0.0)):
    """Extended landing approach from high altitude."""
    return landing_scenario(rng, alt_range=alt_range, horiz_range=horiz_range,
                            vel_range=vel_range)


def liftoff_fast_scenario(rng, alt_range=(50.0, 100.0), horiz_range=(-10.0, 10.0)):
    """Aggressive liftoff with lateral component."""
    target = np.array([
        rng.uniform(*horiz_range),
        rng.uniform(*alt_range),
        rng.uniform(*horiz_range),
    ])
    return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), target


SCENARIO_GENERATORS = {
    "vertical": vertical_scenario,
    "vertical_micro": vertical_micro_scenario,
    "lateral": lateral_scenario,
    "lateral_short": lateral_short_scenario,
    "diagonal": diagonal_scenario,
    "landing": landing_scenario,
    "landing_precision": landing_precision_scenario,
    "landing_high": landing_high_scenario,
    "liftoff": liftoff_scenario,
    "liftoff_fast": liftoff_fast_scenario,
}


# ──────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────

class HopperTrackingEnv(gym.Env):
    """
    Goal-conditioned single-waypoint tracking environment.

    The RL agent learns to navigate from any start state to a target position.
    At inference, an external waypoint sequencer feeds targets one at a time.
    """

    metadata = {"render_modes": ["human"], "render_fps": 50}

    # Physics features constants (matching Variant A from ablation study)
    A_MAX = 4.0  # conservative max deceleration for brake margin

    def __init__(
        self,
        scenarios: list[str] | str = "vertical",
        max_episode_steps: int = 500,
        dt: float = 0.02,
        # Scenario ranges (passed to generators)
        alt_range: Tuple[float, float] = (5.0, 30.0),
        horiz_range: Tuple[float, float] = (-15.0, 15.0),
        vel_range: Tuple[float, float] = (-1.5, 1.5),
        init_tilt_range: Tuple[float, float] = (-0.15, 0.15),
        init_fuel_range: Tuple[float, float] = (0.6, 1.0),
        # Tracking parameters
        acceptance_radius: float = 1.5,
        max_arrival_speed: float = 1.0,
        # Rocket parameters
        dry_mass: float = 60.0,
        fuel_mass: float = 80.0,
        length: float = 4.0,
        max_thrust: float = 2000.0,
        burn_rate: float = 0.68,
        max_gimbal_angle: float = 0.26,
        gimbal_rate_limit: float = 1.0,
        # Bounds
        max_altitude: float = 100.0,
        max_tilt: float = 1.0,
        # Domain randomization
        domain_randomization: Optional[DomainRandomization] = None,
        # NDJSON recording
        record_ndjson: bool = False,
    ):
        super().__init__()

        if isinstance(scenarios, str):
            scenarios = [scenarios]
        self.scenarios = scenarios
        self.max_episode_steps = max_episode_steps
        self.dt = dt

        # Scenario ranges
        self.alt_range = alt_range
        self.horiz_range = horiz_range
        self.vel_range = vel_range
        self.init_tilt_range = init_tilt_range
        self.init_fuel_range = init_fuel_range

        # Tracking
        self.acceptance_radius = acceptance_radius
        self.max_arrival_speed = max_arrival_speed

        # Rocket
        self.dry_mass = dry_mass
        self.fuel_mass = fuel_mass
        self.length = length
        self.max_thrust = max_thrust
        self.burn_rate = burn_rate
        self.max_gimbal_angle = max_gimbal_angle
        self.gimbal_rate_limit = gimbal_rate_limit

        # Termination
        self.max_altitude = max_altitude
        self.max_tilt = max_tilt
        self.domain_randomization = domain_randomization
        self.record_ndjson = record_ndjson
        self._ndjson_messages: list = []

        # Action space: [throttle, gimbal_outer, gimbal_inner]
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )

        # Observation space: base_state(13) + goal_rel(3) + features(3) + mode(3) = 22
        alt_bound = max(150.0, self.max_altitude * 1.2)
        vel_bound = max(50.0, math.sqrt(2 * 9.81 * self.max_altitude) * 1.2)
        dist_bound = 200.0

        obs_low = np.array([
            -100, -10, -100,                          # position
            -vel_bound, -vel_bound, -vel_bound,       # velocity
            -1, -1, -1, -1,                           # quaternion
            -10, -10, -10,                            # angular velocity
            -dist_bound, -dist_bound, -dist_bound,    # goal relative
            0.0, -vel_bound, -dist_bound,             # features
            0, 0, 0,                                  # mode one-hot
        ], dtype=np.float32)

        obs_high = np.array([
            100, alt_bound, 100,
            vel_bound, vel_bound, vel_bound,
            1, 1, 1, 1,
            10, 10, 10,
            dist_bound, dist_bound, dist_bound,
            dist_bound, vel_bound, dist_bound,
            1, 1, 1,                                  # mode one-hot
        ], dtype=np.float32)

        self.observation_space = spaces.Box(low=obs_low, high=obs_high)

        # Initialized in reset()
        self._kernel: Optional[Kernel] = None
        self._sensors: Optional[Sensors] = None
        self._actuators: Optional[Actuators] = None
        self._target: Optional[np.ndarray] = None
        self._step_count: int = 0
        self._time: float = 0.0
        self._prev_dist: float = 0.0
        self._prev_throttle: float = 0.0
        self._prev_omega_mag: float = 0.0
        self._last_reward_components: dict = {}
        self._last_action: Optional[np.ndarray] = None
        self._ep_params: Optional[dict] = None

    # ── Factory methods (match RocketLandingEnvV2) ──

    def _sample_episode_params(self) -> dict:
        dr = self.domain_randomization
        rng = self.np_random
        if dr is None:
            return {
                "dry_mass": self.dry_mass, "fuel_mass": self.fuel_mass,
                "length": self.length, "max_thrust": self.max_thrust,
                "burn_rate": self.burn_rate,
                "gimbal_rate": self.gimbal_rate_limit, "gimbal_tau": 0.05,
                "gravity": 9.81, "air_density": 1.225,
                "wind": WorldVec3(0, 0, 0),
                "gps_noise_std": 0.0, "baro_noise_std": 0.0,
                "velocity_noise_std": 0.0, "orientation_noise_std": 0.0,
                "gyro_noise_std": 0.0, "gyro_bias": 0.0,
            }
        wind_speed = rng.uniform(*dr.wind_speed_range)
        wind_angle = rng.uniform(0, 2 * math.pi)
        wind = WorldVec3(wind_speed * math.cos(wind_angle), 0,
                         wind_speed * math.sin(wind_angle))
        return {
            "dry_mass": self.dry_mass * rng.uniform(*dr.dry_mass_scale),
            "fuel_mass": self.fuel_mass * rng.uniform(*dr.fuel_mass_scale),
            "length": self.length * rng.uniform(*dr.length_scale),
            "max_thrust": self.max_thrust * rng.uniform(*dr.max_thrust_scale),
            "burn_rate": self.burn_rate * rng.uniform(*dr.burn_rate_scale),
            "gimbal_rate": self.gimbal_rate_limit * rng.uniform(*dr.gimbal_rate_scale),
            "gimbal_tau": 0.05 * rng.uniform(*dr.gimbal_tau_scale),
            "gravity": 9.81 * rng.uniform(*dr.gravity_scale),
            "air_density": 1.225 * rng.uniform(*dr.air_density_scale),
            "wind": wind,
            "gps_noise_std": rng.uniform(*dr.gps_noise_std),
            "baro_noise_std": rng.uniform(*dr.baro_noise_std),
            "velocity_noise_std": rng.uniform(*dr.velocity_noise_std),
            "orientation_noise_std": rng.uniform(*dr.orientation_noise_std),
            "gyro_noise_std": rng.uniform(*dr.gyro_noise_std),
            "gyro_bias": rng.uniform(*dr.gyro_bias_range),
        }

    def _make_rocket(self, init_state, fuel_fraction, ep):
        dry_mass = ep["dry_mass"]
        fuel_mass_full = ep["fuel_mass"]
        length = ep["length"]
        burn_rate = ep["burn_rate"]

        mass_model = MassModel(components=[
            MassComponent("dry", dry_mass, RocketVec3(length / 2, 0, 0)),
            FuelComponent(
                "fuel",
                mass=fuel_mass_full * fuel_fraction,
                position=RocketVec3(length * 0.3, 0, 0),
                full_mass=fuel_mass_full,
                burn_rate=burn_rate,
            ),
        ])
        props = mass_model.to_rocket_props(
            length=length,
            thrust_application_offset=RocketVec3(0, 0, 0),
        )
        return Rocket(props=props, state=init_state, mass_model=mass_model)

    def _make_environment(self, ep):
        return Environment(
            gravity=WorldVec3(0.0, -ep["gravity"], 0.0),
            wind=ep["wind"],
            air_density=ep["air_density"],
        )

    def _make_actuators(self, rocket, ep):
        max_thrust = ep["max_thrust"]
        gimbal = GimbalActuator(
            rocket=rocket,
            max_outer_angle=self.max_gimbal_angle,
            max_inner_angle=self.max_gimbal_angle,
            max_rate=ep["gimbal_rate"],
            tau=ep["gimbal_tau"],
        )
        throttle = ThrottleActuator(
            rocket=rocket,
            max_thrust_rate=max_thrust * 2,
            min_thrust=0.0,
        )
        return Actuators(gimbal=gimbal, throttle=throttle)

    def _make_sensors(self, kernel, ep):
        if ep is None or self.domain_randomization is None:
            return Sensors(kernel)
        gps_errors = None
        if ep["gps_noise_std"] > 0:
            gps_errors = {
                "x": GaussianNoise(std=ep["gps_noise_std"]),
                "z": GaussianNoise(std=ep["gps_noise_std"]),
            }
        baro_errors = None
        if ep["baro_noise_std"] > 0:
            baro_errors = {"altitude": GaussianNoise(std=ep["baro_noise_std"])}
        velocity_errors = None
        if ep["velocity_noise_std"] > 0:
            velocity_errors = {
                ch: GaussianNoise(std=ep["velocity_noise_std"]) for ch in ("x", "y", "z")
            }
        orientation_errors = None
        if ep["orientation_noise_std"] > 0:
            orientation_errors = {
                ch: GaussianNoise(std=ep["orientation_noise_std"])
                for ch in ("roll", "pitch", "yaw")
            }
        gyro_errors = None
        if ep["gyro_noise_std"] > 0 or ep["gyro_bias"] != 0:
            gyro_errors = {}
            for ch in ("x", "y", "z"):
                errors = []
                if ep["gyro_bias"] != 0:
                    errors.append(Bias(offset=self.np_random.uniform(
                        -abs(ep["gyro_bias"]), abs(ep["gyro_bias"]))))
                if ep["gyro_noise_std"] > 0:
                    errors.append(GaussianNoise(std=ep["gyro_noise_std"]))
                gyro_errors[ch] = errors[0] if len(errors) == 1 else CompositeError(errors=errors)
        return Sensors(kernel, gps_errors=gps_errors, velocity_errors=velocity_errors,
                       orientation_errors=orientation_errors, gyro_errors=gyro_errors,
                       baro_errors=baro_errors)

    # ── Observation ──

    def _get_obs(self) -> np.ndarray:
        gps = self._sensors.GPS.read()
        alt = self._sensors.Barometer.read()
        vel = self._sensors.VelocitySensor.read()
        orient = self._sensors.OrientationSensor.read()
        omega = self._sensors.GyroSensor.read()

        qw, qx, qy, qz = orient.as_array()
        pos = np.array([gps.x, alt, gps.z])
        vel_vec = np.array([vel.x, vel.y, vel.z])

        # Goal-relative
        goal_rel = self._target - pos

        # Features
        dist = np.linalg.norm(goal_rel)
        if dist > 0.01:
            dir_to_wp = goal_rel / dist
            vel_toward = float(np.dot(vel_vec, dir_to_wp))
        else:
            vel_toward = 0.0
        speed = np.linalg.norm(vel_vec)
        brake_margin = dist - speed**2 / (2 * self.A_MAX)

        # Flight mode (one-hot): TRACK, LAND, LIFTOFF
        is_land = self._target[1] < 1.0
        on_ground = self._kernel.ground_contact.on_ground
        is_liftoff = on_ground and not is_land
        is_track = not is_land and not is_liftoff

        return np.array([
            gps.x, alt, gps.z,
            vel.x, vel.y, vel.z,
            qw, qx, qy, qz,
            omega.x, omega.y, omega.z,
            goal_rel[0], goal_rel[1], goal_rel[2],
            dist, vel_toward, brake_margin,
            float(is_track), float(is_land), float(is_liftoff),
        ], dtype=np.float32)

    def _get_tilt_angle(self) -> float:
        orient = self._kernel.rocket.state.orientation
        nose_body = RocketVec3(1.0, 0.0, 0.0)
        nose_world = nose_body.to_world(orient)
        cos_angle = max(-1.0, min(1.0, nose_world.y))
        return math.acos(cos_angle)

    # ── Reward R8 ──
    #
    # Unified design: 7 dense terms, no scenario branching.
    # Follows R7 structure but with 2-3x stronger smoothness penalties
    # and new gimbal rate + fuel efficiency terms.
    #
    # Key improvements over R7:
    #   - Uprightness: 0.05 → 0.08 (1.6x)
    #   - Angular velocity: 0.025 → 0.06 (2.4x)
    #   - Throttle jerk: 0.01 → 0.025 (2.5x)
    #   - Angular acceleration: 0.005 → 0.015 (3x)
    #   - NEW: gimbal rate penalty (-0.02 × Δgimbal²)
    #   - NEW: fuel efficiency (-0.001 × throttle)

    MAX_DECEL = 8.0  # mid-fuel TWR for braking profile

    def _compute_reward(self, obs, action, terminated, truncated, info):
        dist = float(obs[16])
        speed = math.sqrt(float(obs[3])**2 + float(obs[4])**2 + float(obs[5])**2)
        tilt = self._get_tilt_angle()
        omega_mag = math.sqrt(float(obs[10])**2 + float(obs[11])**2 + float(obs[12])**2)
        throttle = float(action[0])
        fuel_frac = self._kernel.rocket.mass_model.fuel_fraction

        outcome = info.get("outcome")

        # ── Terminal rewards (unified, no branching) ──
        if outcome in ("arrived", "landed"):
            reward = (200.0
                      + 50.0 * max(0.0, 1.0 - speed / 2.0)
                      + 30.0 * max(0.0, 1.0 - dist / self.acceptance_radius)
                      + 10.0 * fuel_frac)
            self._last_reward_components = {"terminal": reward}
            self._prev_dist = dist
            self._prev_throttle = throttle
            self._prev_omega_mag = omega_mag
            return reward

        if outcome == "crash":
            self._last_reward_components = {"terminal": -200.0}
            self._prev_dist = dist
            self._prev_throttle = throttle
            self._prev_omega_mag = omega_mag
            return -200.0

        if truncated:
            reward = -50.0 - 0.5 * min(dist, 50.0)
            self._last_reward_components = {"terminal": reward}
            self._prev_dist = dist
            self._prev_throttle = throttle
            self._prev_omega_mag = omega_mag
            return reward

        # ── Dense shaping (7 unified terms, all scenarios) ──

        # 1. Distance potential (unchanged from R7)
        r_approach = 0.15 * (self._prev_dist - dist)

        # 2. Braking profile — GLOBAL
        v_safe = math.sqrt(2.0 * self.MAX_DECEL * max(0.1, dist))
        excess = max(0.0, speed - v_safe)
        r_brake = -0.3 * excess ** 2

        # 3. Hover quality — near-target slow-down
        r_hover = 0.0
        if dist < 3.0:
            r_hover = 0.15 * max(0.0, 1.0 - speed / 1.5)

        # 4. Uprightness (strengthened: 0.08 → 0.20 to prevent dangerous tilts)
        r_tilt = -0.20 * tilt ** 2

        # 5. Smoothness (all penalties 2-3x stronger)
        r_smooth = (
            -0.06 * omega_mag                                    # angular rate (was -0.025)
            - 0.025 * (throttle - self._prev_throttle) ** 2      # throttle jerk (was -0.01)
            - 0.015 * (omega_mag - self._prev_omega_mag) ** 2    # angular accel (was -0.005)
        )

        # 6. Gimbal rate penalty (NEW)
        gimbal_outer = self._actuators.gimbal_outer_angle or 0.0
        gimbal_inner = self._actuators.gimbal_inner_angle or 0.0
        prev_go = getattr(self, '_prev_gimbal_outer', 0.0)
        prev_gi = getattr(self, '_prev_gimbal_inner', 0.0)
        gimbal_delta_sq = (gimbal_outer - prev_go) ** 2 + (gimbal_inner - prev_gi) ** 2
        r_gimbal = -0.02 * gimbal_delta_sq
        self._prev_gimbal_outer = gimbal_outer
        self._prev_gimbal_inner = gimbal_inner

        # 7. Fuel efficiency (NEW — tiny dense penalty)
        r_fuel = -0.001 * throttle

        # Store components for ML debug recording
        self._last_reward_components = {
            "approach": r_approach,
            "brake": r_brake,
            "hover": r_hover,
            "tilt": r_tilt,
            "smooth": r_smooth,
            "gimbal": r_gimbal,
            "fuel": r_fuel,
        }

        # Update tracking state
        self._prev_dist = dist
        self._prev_throttle = throttle
        self._prev_omega_mag = omega_mag

        return r_approach + r_brake + r_hover + r_tilt + r_smooth + r_gimbal + r_fuel

    # ── Termination ──

    def _check_termination(self, obs, info):
        state = self._kernel.rocket.state
        ground = self._kernel.ground_contact
        dist = float(obs[16])
        speed = math.sqrt(float(obs[3])**2 + float(obs[4])**2 + float(obs[5])**2)

        terminated = False
        truncated = self._step_count >= self.max_episode_steps

        # Success: arrived at aerial waypoint
        is_landing_wp = self._target[1] < 1.0
        if not is_landing_wp and dist < self.acceptance_radius and speed < self.max_arrival_speed:
            terminated = True
            info["outcome"] = "arrived"
            info["arrival_dist"] = dist
            info["arrival_speed"] = speed

        # Success: landed at ground waypoint
        if is_landing_wp and ground.on_ground:
            terminated = True
            if ground.crashed or ground.tipped_over:
                info["outcome"] = "crash"
            else:
                vel_mag = state.velocity.norm
                tilt_deg = math.degrees(self._get_tilt_angle())
                info["outcome"] = "landed"
                info["landing_vel"] = ground.impact_speed
                info["landing_tilt"] = math.degrees(ground.landing_angle)
                info["landing_dist"] = math.sqrt(state.position.x**2 + state.position.z**2)

        # Crash: ground contact when target is in the air
        if not is_landing_wp and ground.on_ground and state.position.y <= 0.01:
            # Grace period must exceed thrust ramp time: at max_thrust_rate=4000 N/s,
            # reaching hover thrust (1373N for 140kg) takes ~18 steps (0.36s).
            if self._step_count > 50:
                terminated = True
                info["outcome"] = "crash"

        # Excessive tilt
        if self._get_tilt_angle() > self.max_tilt:
            terminated = True
            if "outcome" not in info:
                info["outcome"] = "crash"

        # Out of fuel
        if self._kernel.rocket.mass_model.fuel_fraction <= 0:
            terminated = True
            if "outcome" not in info:
                info["outcome"] = "crash"

        # Out of bounds
        if state.position.y > self.max_altitude:
            terminated = True
            if "outcome" not in info:
                info["outcome"] = "crash"

        # Kinematic early termination (RAJS/Jiang 2024):
        # If braking distance exceeds remaining distance, episode is infeasible
        if not terminated and "outcome" not in info and self._step_count > 10:
            brake_dist = speed ** 2 / (2.0 * self.A_MAX)
            if brake_dist > dist * 2.0 and dist > 5.0:
                terminated = True
                info["outcome"] = "crash"

        return terminated, truncated

    # ── Reset ──

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Sample scenario type
        scenario_name = self.np_random.choice(self.scenarios)
        gen = SCENARIO_GENERATORS[scenario_name]

        # Scenario-specific kwargs — pass ranges to all that accept them
        gen_kwargs = {}
        import inspect
        sig = inspect.signature(gen)
        if "alt_range" in sig.parameters:
            gen_kwargs["alt_range"] = self.alt_range
        if "horiz_range" in sig.parameters:
            gen_kwargs["horiz_range"] = self.horiz_range
        if "vel_range" in sig.parameters:
            gen_kwargs["vel_range"] = self.vel_range

        start_pos, start_vel, target = gen(self.np_random, **gen_kwargs)
        self._target = target.copy()

        # Orientation: upright with small random tilt
        init_tilt_y = self.np_random.uniform(*self.init_tilt_range)
        init_tilt_z = self.np_random.uniform(*self.init_tilt_range)
        base_rot = get_upright_orientation()._as_scipy_rotation()
        perturb = R.from_euler("yz", [init_tilt_y, init_tilt_z])
        combined = perturb * base_rot
        sx, sy, sz, sw = combined.as_quat()
        init_orient = Orientation.from_wxyz(sw, sx, sy, sz)

        # Fuel
        init_fuel = self.np_random.uniform(*self.init_fuel_range)

        # For on-ground starts, ensure position is exactly 0
        if start_pos[1] < 0.5:
            start_pos[1] = 0.0
            start_vel[:] = 0.0  # on ground = zero velocity

        init_state = RocketState(
            position=WorldVec3(float(start_pos[0]), float(start_pos[1]),
                               float(start_pos[2])),
            velocity=WorldVec3(float(start_vel[0]), float(start_vel[1]),
                               float(start_vel[2])),
            orientation=init_orient,
            angular_velocity_body=RocketVec3(0.0, 0.0, 0.0),
            thrust_body=RocketVec3(0.0, 0.0, 0.0),
        )

        # Sample episode parameters (DR)
        ep = self._sample_episode_params()
        self._ep_params = ep

        rocket = self._make_rocket(init_state, init_fuel, ep)
        environment = self._make_environment(ep)
        self._kernel = Kernel(rocket, environment)

        # Set ground contact for on-ground starts
        if start_pos[1] <= 0.0:
            self._kernel.ground_contact.on_ground = True

        self._sensors = self._make_sensors(self._kernel, ep)
        self._actuators = self._make_actuators(rocket, ep)

        self._step_count = 0
        self._time = 0.0
        self._prev_throttle = 0.0
        self._prev_omega_mag = 0.0
        self._last_reward_components = {}
        self._last_action = None
        self._is_liftoff = (start_pos[1] <= 0.0 and target[1] > 1.0)

        obs = self._get_obs()
        self._prev_dist = float(obs[16])

        # NDJSON recording — serialize immediately to snapshot mutable state
        if self.record_ndjson:
            self._ndjson_messages = []
            meta = MetadataMessage(
                scenario_name=scenario_name,
                notes=f"target={target.tolist()}, start={start_pos.tolist()}",
                dt=self.dt,
            )
            self._ndjson_messages.append(meta.model_dump_json(exclude_none=True))
            const = collect(
                ConstantMessage, 0.0,
                props=self._kernel.rocket.props,
                env=self._kernel.environment,
            )
            self._ndjson_messages.append(const.model_dump_json(exclude_none=True))
            self._record_frame()

        info = {
            "scenario": scenario_name,
            "target": target.tolist(),
            "start_pos": start_pos.tolist(),
        }
        return obs, info

    def _record_frame(self):
        """Record current state as a serialized JSON line for NDJSON replay.

        We serialize immediately because the kernel's WorldVec3 objects are
        mutable — storing references would give stale data when ground
        clamping later sets position.y = 0.
        """
        if not self.record_ndjson:
            return
        msg = collect(
            VariableMessage, self._time,
            state=self._kernel.rocket.state,
            props=self._kernel.rocket.props,
            model=self._kernel.rocket.mass_model,
            ground=self._kernel.ground_contact,
            sensors=self._sensors,
            actuators=self._actuators,
        )
        # Serialize NOW to snapshot mutable state
        self._ndjson_messages.append(msg.model_dump_json(exclude_none=True))

        # ML debug message (reward components, actions, features)
        if self._last_reward_components and self._last_action is not None:
            obs = self._get_obs()
            ml_msg = MLDebugMessage(
                timestamp=self._time,
                reward_total=sum(self._last_reward_components.values()),
                reward_components=self._last_reward_components,
                action_throttle=float(self._last_action[0]),
                action_tilt_x=float(self._last_action[1]),
                action_tilt_z=float(self._last_action[2]),
                dist_to_target=float(obs[16]),
                vel_toward=float(obs[17]),
                brake_margin=float(obs[18]),
                tilt_angle=self._get_tilt_angle(),
                omega_mag=math.sqrt(
                    float(obs[10])**2 + float(obs[11])**2 + float(obs[12])**2
                ),
                target=self._target.tolist(),
            )
            self._ndjson_messages.append(
                ml_msg.model_dump_json(exclude_none=True)
            )

    def save_ndjson(self, path: str) -> None:
        """Write recorded NDJSON messages to a file."""
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for line in self._ndjson_messages:
                f.write(line + "\n")

    # ── Step ──

    def step(self, action):
        self._last_action = action
        # Scale action
        ep_thrust = self._ep_params["max_thrust"]
        throttle_N = float(action[0]) * ep_thrust
        gimbal_outer = float(action[1]) * self.max_gimbal_angle
        gimbal_inner = float(action[2]) * self.max_gimbal_angle

        # Apply commands
        self._actuators._throttle.apply(ThrottleCommand(thrust_N=throttle_N))
        self._actuators._gimbal.apply(GimbalCommand(
            desired_outer_angle=gimbal_outer,
            desired_inner_angle=gimbal_inner,
        ))

        # Advance simulation
        self._sensors.set_time(self._time)
        self._actuators.set_time(self._time)
        self._actuators.step(self.dt)
        self._kernel.step(self.dt)
        self._time += self.dt
        self._step_count += 1

        # Observe
        obs = self._get_obs()
        info = {"time": self._time}

        # Termination
        terminated, truncated = self._check_termination(obs, info)

        # Reward
        reward = self._compute_reward(obs, action, terminated, truncated, info)

        # Record frame
        self._record_frame()

        return obs, reward, terminated, truncated, info
