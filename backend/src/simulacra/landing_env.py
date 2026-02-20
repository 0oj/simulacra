"""
Landing scenario generator and Gymnasium environment.

Generates apogee-start scenarios where the rocket must land softly at origin.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

from .noumena import (
    Kernel,
    Rocket,
    RocketProps,
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
from .physics import Orientation, RocketVec3, WorldVec3


@dataclass
class LandingScenarioConfig:
    """Configuration for landing scenario generation."""

    # Altitude range at apogee
    altitude_range: Tuple[float, float] = (80.0, 120.0)

    # Horizontal displacement from landing pad (0,0)
    horizontal_range: Tuple[float, float] = (-20.0, 20.0)

    # Initial velocity at apogee (near zero vertical, some horizontal drift)
    vertical_velocity_range: Tuple[float, float] = (-2.0, 2.0)  # near apogee
    horizontal_velocity_range: Tuple[float, float] = (-5.0, 5.0)

    # Initial tilt from vertical (radians)
    tilt_range: Tuple[float, float] = (-0.15, 0.15)  # ~8 degrees

    # Initial angular velocity (rad/s)
    angular_velocity_range: Tuple[float, float] = (-0.1, 0.1)

    # Rocket parameters
    dry_mass: float = 60.0  # kg without fuel
    fuel_mass: float = 40.0  # kg of fuel
    length: float = 4.0
    max_thrust: float = 2000.0  # ~2x weight for T/W > 1
    max_gimbal_angle: float = 0.26  # ~15 degrees
    fuel_burn_rate: float = 8.0  # kg/s at full thrust

    # Computed property
    @property
    def mass(self) -> float:
        return self.dry_mass + self.fuel_mass

    # Success criteria
    max_landing_velocity: float = 2.0  # m/s
    max_landing_tilt: float = 0.17  # ~10 degrees
    landing_pad_radius: float = 5.0  # meters from origin


class LandingScenarioGenerator:
    """
    Generates randomized landing scenarios.

    Each scenario starts the rocket at apogee with random position,
    velocity, and orientation. The task is to land softly at origin.
    """

    def __init__(
        self,
        config: Optional[LandingScenarioConfig] = None,
        seed: Optional[int] = None,
    ):
        self.config = config or LandingScenarioConfig()
        self.rng = np.random.default_rng(seed)

    def generate(self) -> Tuple[Rocket, dict]:
        """
        Generate a random landing scenario.

        Returns:
            rocket: Configured Rocket object at initial state
            info: Dict with scenario parameters for logging
        """
        cfg = self.config

        # Random initial position
        x = self.rng.uniform(*cfg.horizontal_range)
        y = self.rng.uniform(*cfg.altitude_range)
        z = self.rng.uniform(*cfg.horizontal_range)

        # Random initial velocity (near apogee = low vertical velocity)
        vx = self.rng.uniform(*cfg.horizontal_velocity_range)
        vy = self.rng.uniform(*cfg.vertical_velocity_range)
        vz = self.rng.uniform(*cfg.horizontal_velocity_range)

        # Random orientation (tilted from vertical)
        tilt_y = self.rng.uniform(*cfg.tilt_range)
        tilt_z = self.rng.uniform(*cfg.tilt_range)
        base_rot = get_upright_orientation()._as_scipy_rotation()
        perturb_y = R.from_euler("y", tilt_y)
        perturb_z = R.from_euler("z", tilt_z)
        combined = perturb_z * perturb_y * base_rot
        qx, qy, qz, qw = combined.as_quat()
        orientation = Orientation.from_wxyz(qw, qx, qy, qz)

        # Random angular velocity
        omega_x = self.rng.uniform(*cfg.angular_velocity_range)
        omega_y = self.rng.uniform(*cfg.angular_velocity_range)
        omega_z = self.rng.uniform(*cfg.angular_velocity_range)

        # Build mass model with fuel
        # Dry mass at CoM, fuel tank slightly lower (closer to engine)
        com_x = cfg.length / 2  # center of mass along rocket axis
        fuel_x = cfg.length * 0.3  # fuel tank closer to base

        mass_model = MassModel(components=[
            MassComponent("dry_mass", cfg.dry_mass, RocketVec3(com_x, 0, 0)),
            FuelComponent(
                "fuel",
                cfg.fuel_mass,  # current mass
                RocketVec3(fuel_x, 0, 0),
                full_mass=cfg.fuel_mass,
                burn_rate=cfg.fuel_burn_rate,
            ),
        ])

        # Build rocket props from mass model
        props = mass_model.to_rocket_props(
            length=cfg.length,
            thrust_application_offset=RocketVec3(0, 0, 0),
        )

        # Start with zero thrust (engine off at apogee)
        state = RocketState(
            position=WorldVec3(x, y, z),
            velocity=WorldVec3(vx, vy, vz),
            orientation=orientation,
            angular_velocity_body=RocketVec3(omega_x, omega_y, omega_z),
            thrust_body=RocketVec3(0, 0, 0),  # engine off
        )

        rocket = Rocket(props=props, state=state, mass_model=mass_model)

        info = {
            "initial_position": (x, y, z),
            "initial_velocity": (vx, vy, vz),
            "initial_tilt_deg": (math.degrees(tilt_y), math.degrees(tilt_z)),
            "distance_to_pad": math.sqrt(x**2 + z**2),
        }

        return rocket, info


class RocketLandingEnv(gym.Env):
    """
    Gymnasium environment for rocket landing.

    Observation (17-dim):
        [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz, thrust, dx, dz, dist]
        where dx, dz = displacement to landing pad, dist = horizontal distance

    Action (3-dim, continuous):
        [throttle, gimbal_outer, gimbal_inner]

    Reward:
        - Shaping reward for getting closer to pad and slowing down
        - Large bonus for soft landing
        - Large penalty for crash or tipover

    Termination:
        - Landed (y ≈ 0, low velocity, upright)
        - Crashed (high velocity impact)
        - Tipped over
        - Out of bounds
    """

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        config: Optional[LandingScenarioConfig] = None,
        dt: float = 0.02,
        max_episode_steps: int = 1000,  # 20 seconds
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.config = config or LandingScenarioConfig()
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        self.generator = LandingScenarioGenerator(self.config)

        # Action space: [throttle, gimbal_outer, gimbal_inner]
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )

        # Observation space: state (14) + target info (3)
        obs_low = np.array([
            -200, -10, -200,      # position (allow below ground for detection)
            -100, -100, -100,     # velocity
            -1, -1, -1, -1,       # quaternion
            -20, -20, -20,        # angular velocity
            0,                    # thrust
            -200, -200,           # displacement to pad (dx, dz)
            0,                    # distance to pad
        ], dtype=np.float32)

        obs_high = np.array([
            200, 200, 200,
            100, 100, 100,
            1, 1, 1, 1,
            20, 20, 20,
            self.config.max_thrust,
            200, 200,
            300,
        ], dtype=np.float32)

        self.observation_space = spaces.Box(low=obs_low, high=obs_high)

        # Internal state
        self._kernel: Optional[Kernel] = None
        self._sensors: Optional[Sensors] = None
        self._actuators: Optional[Actuators] = None
        self._step_count: int = 0
        self._prev_shaping: Optional[float] = None

    def _make_environment(self) -> Environment:
        return Environment(
            gravity=WorldVec3(0, -9.81, 0),
            wind=WorldVec3(0, 0, 0),
            air_density=1.225,
        )

    def _make_actuators(self, rocket: Rocket) -> Actuators:
        gimbal = GimbalActuator(
            rocket=rocket,
            max_outer_angle=self.config.max_gimbal_angle,
            max_inner_angle=self.config.max_gimbal_angle,
            max_rate=1.0,
            tau=0.05,
        )
        throttle = ThrottleActuator(
            rocket=rocket,
            max_thrust_rate=self.config.max_thrust * 2,
            min_thrust=0.0,
        )
        return Actuators(gimbal=gimbal, throttle=throttle)

    def _get_obs(self) -> np.ndarray:
        gps = self._sensors.GPS.read()  # Horizontal position (x, z)
        alt = self._sensors.Barometer.read()  # Altitude (y)
        vel = self._sensors.VelocitySensor.read()
        orient = self._sensors.OrientationSensor.read()
        omega = self._sensors.GyroSensor.read()
        thrust = self._sensors.ThrustSensor.read()

        qw, qx, qy, qz = orient.as_array()

        # Target info: displacement to landing pad at origin
        dx = -gps.x  # how far we need to go in x
        dz = -gps.z  # how far we need to go in z
        dist = math.sqrt(gps.x**2 + gps.z**2)

        return np.array([
            gps.x, alt, gps.z,
            vel.x, vel.y, vel.z,
            qw, qx, qy, qz,
            omega.x, omega.y, omega.z,
            thrust,
            dx, dz, dist,
        ], dtype=np.float32)

    def _get_tilt_angle(self) -> float:
        """Tilt from vertical in radians."""
        orient = self._kernel.rocket.state.orientation
        nose_body = RocketVec3(1.0, 0.0, 0.0)
        nose_world = nose_body.to_world(orient)
        cos_angle = max(-1.0, min(1.0, nose_world.y))
        return math.acos(cos_angle)

    def _compute_shaping(self) -> float:
        """
        Potential-based shaping reward.

        We want to minimize:
        - Distance to pad (x, z)
        - Altitude (want to descend)
        - Velocity magnitude
        - Tilt angle
        """
        state = self._kernel.rocket.state

        # Horizontal distance to pad
        dist_to_pad = math.sqrt(state.position.x**2 + state.position.z**2)

        # Altitude (lower is better, but not negative)
        altitude = max(0, state.position.y)

        # Velocity magnitude
        vel_mag = state.velocity.norm

        # Tilt
        tilt = self._get_tilt_angle()

        # Potential function (negative = good state)
        # Weights tuned for reasonable shaping
        potential = -(
            1.0 * dist_to_pad +
            0.5 * altitude +
            0.3 * vel_mag +
            5.0 * tilt
        )

        return potential

    def _compute_reward(self, action: np.ndarray) -> float:
        """Compute reward with potential-based shaping."""

        # Shaping: reward improvement in potential
        shaping = self._compute_shaping()
        if self._prev_shaping is not None:
            reward = shaping - self._prev_shaping
        else:
            reward = 0.0
        self._prev_shaping = shaping

        # Small action penalty
        reward -= 0.001 * np.linalg.norm(action)

        return float(reward)

    def _check_termination(self) -> Tuple[bool, bool, dict]:
        """
        Check termination conditions.

        Returns (terminated, truncated, info)
        """
        state = self._kernel.rocket.state
        ground = self._kernel.ground_contact
        cfg = self.config

        info = {}

        # Truncation (time limit)
        truncated = self._step_count >= self.max_episode_steps

        terminated = False

        # Check landing
        if ground.on_ground:
            vel_mag = state.velocity.norm
            tilt = self._get_tilt_angle()
            dist_to_pad = math.sqrt(state.position.x**2 + state.position.z**2)

            # Successful landing?
            soft = vel_mag < cfg.max_landing_velocity
            upright = tilt < cfg.max_landing_tilt
            on_pad = dist_to_pad < cfg.landing_pad_radius

            if soft and upright:
                terminated = True
                info["landed"] = True
                info["on_pad"] = on_pad
                info["landing_velocity"] = vel_mag
                info["landing_tilt_deg"] = math.degrees(tilt)
                info["landing_distance"] = dist_to_pad
            elif ground.tipped_over:
                terminated = True
                info["tipped_over"] = True
            else:
                # Hard landing but not tipped
                terminated = True
                info["crashed"] = True
                info["impact_velocity"] = vel_mag

        # Out of bounds
        if abs(state.position.x) > 150 or abs(state.position.z) > 150:
            terminated = True
            info["out_of_bounds"] = True

        if state.position.y > 200:
            terminated = True
            info["too_high"] = True

        return terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        if seed is not None:
            self.generator = LandingScenarioGenerator(self.config, seed=seed)

        rocket, scenario_info = self.generator.generate()
        environment = self._make_environment()

        self._kernel = Kernel(rocket, environment)
        self._sensors = Sensors(self._kernel)
        self._actuators = self._make_actuators(rocket)

        self._step_count = 0
        self._prev_shaping = None

        obs = self._get_obs()

        return obs, scenario_info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Scale action
        throttle = float(action[0]) * self.config.max_thrust
        gimbal_outer = float(action[1]) * self.config.max_gimbal_angle
        gimbal_inner = float(action[2]) * self.config.max_gimbal_angle

        # Apply commands
        self._actuators._throttle.apply(ThrottleCommand(thrust_N=throttle))
        self._actuators._gimbal.apply(GimbalCommand(
            desired_outer_angle=gimbal_outer,
            desired_inner_angle=gimbal_inner,
        ))

        # Step simulation
        self._actuators.step(self.dt)
        self._kernel.step(self.dt)
        self._step_count += 1

        # Get outputs
        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated, truncated, term_info = self._check_termination()

        # Add terminal rewards
        if "landed" in term_info:
            reward += 100.0  # big bonus for landing
            if term_info.get("on_pad"):
                reward += 50.0  # bonus for precision
        elif "crashed" in term_info or "tipped_over" in term_info:
            reward -= 50.0

        info = {
            "time": self._step_count * self.dt,
            "altitude": self._kernel.rocket.state.position.y,
            "tilt_deg": math.degrees(self._get_tilt_angle()),
            "velocity": self._kernel.rocket.state.velocity.norm,
            "distance_to_pad": math.sqrt(
                self._kernel.rocket.state.position.x**2 +
                self._kernel.rocket.state.position.z**2
            ),
            **term_info,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
