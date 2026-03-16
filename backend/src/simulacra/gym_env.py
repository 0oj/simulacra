"""
Gymnasium environment wrapper for Simulacra.

Exposes the rocket simulation as a standard Gymnasium environment
for RL training (SAC, PPO, SINDy-RL, etc.).
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple
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
from .phenomena.errors import GaussianNoise, Bias, CompositeError
from .physics import Orientation, RocketVec3, WorldVec3


class RocketHoverEnv(gym.Env):
    """
    Gymnasium environment for VTVL rocket hover control.

    Observation (14-dim):
        [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz, thrust_mag]

    Action (3-dim, continuous):
        [throttle, gimbal_outer, gimbal_inner]
        - throttle: 0 to 1 (scaled to max_thrust)
        - gimbal_outer: -1 to 1 (scaled to max_outer_angle)
        - gimbal_inner: -1 to 1 (scaled to max_inner_angle)

    Task: Hover at target altitude with upright attitude.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        dt: float = 0.02,
        max_episode_steps: int = 500,
        target_altitude: float = 10.0,
        # Rocket parameters
        mass: float = 100.0,
        length: float = 2.0,
        max_thrust: float = 2000.0,
        max_gimbal_angle: float = 0.26,  # ~15 degrees
        gimbal_rate_limit: float = 1.0,  # rad/s
        # Initial condition ranges (for domain randomization)
        init_altitude_range: Tuple[float, float] = (8.0, 12.0),
        init_velocity_range: Tuple[float, float] = (-2.0, 2.0),
        init_tilt_range: Tuple[float, float] = (-0.1, 0.1),
        # Reward weights
        altitude_weight: float = 1.0,
        velocity_weight: float = 0.1,
        tilt_weight: float = 10.0,
        action_weight: float = 0.01,
        # Termination conditions
        max_tilt: float = 0.785,  # ~45 degrees
        max_velocity: float = 20.0,
        min_altitude: float = 0.0,
        max_altitude: float = 50.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.target_altitude = target_altitude

        # Rocket config
        self.mass = mass
        self.length = length
        self.max_thrust = max_thrust
        self.max_gimbal_angle = max_gimbal_angle
        self.gimbal_rate_limit = gimbal_rate_limit

        # Initial conditions
        self.init_altitude_range = init_altitude_range
        self.init_velocity_range = init_velocity_range
        self.init_tilt_range = init_tilt_range

        # Reward weights
        self.altitude_weight = altitude_weight
        self.velocity_weight = velocity_weight
        self.tilt_weight = tilt_weight
        self.action_weight = action_weight

        # Termination
        self.max_tilt = max_tilt
        self.max_velocity = max_velocity
        self.min_altitude = min_altitude
        self.max_altitude = max_altitude

        self.render_mode = render_mode

        # Action space: [throttle, gimbal_outer, gimbal_inner]
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )

        # Observation space: [pos(3), vel(3), quat(4), omega(3), thrust(1)]
        # Use generous bounds
        obs_low = np.array(
            [
                -100, -10, -100,     # position
                -50, -50, -50,       # velocity
                -1, -1, -1, -1,      # quaternion
                -10, -10, -10,       # angular velocity
                0,                   # thrust
            ],
            dtype=np.float32,
        )
        obs_high = np.array(
            [
                100, 100, 100,       # position
                50, 50, 50,          # velocity
                1, 1, 1, 1,          # quaternion
                10, 10, 10,          # angular velocity
                max_thrust,          # thrust
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high)

        # Will be initialized in reset()
        self._kernel: Optional[Kernel] = None
        self._sensors: Optional[Sensors] = None
        self._actuators: Optional[Actuators] = None
        self._step_count: int = 0
        self._time: float = 0.0

    def _make_rocket(self, init_state: RocketState) -> Rocket:
        """Create rocket with current configuration."""
        # Simple cylinder approximation for inertia
        r = 0.1  # radius ~10cm
        Ix = (1 / 12) * self.mass * (3 * r**2 + self.length**2)
        Iy = Ix
        Iz = (1 / 2) * self.mass * r**2

        props = RocketProps(
            length=self.length,
            mass=self.mass,
            center_of_mass_offset=RocketVec3(self.length / 2, 0, 0),
            inertia_diag=(Ix, Iy, Iz),
            thrust_application_offset=RocketVec3(0, 0, 0),
        )

        mass_model = MassModel.constant(self.mass, props.center_of_mass_offset)

        return Rocket(props=props, state=init_state, mass_model=mass_model)

    def _make_environment(self) -> Environment:
        """Create world environment."""
        return Environment(
            gravity=WorldVec3(0.0, -9.81, 0.0),
            wind=WorldVec3(0.0, 0.0, 0.0),
            air_density=1.225,
        )

    def _make_actuators(self, rocket: Rocket) -> Actuators:
        """Create actuator bundle."""
        gimbal = GimbalActuator(
            rocket=rocket,
            max_outer_angle=self.max_gimbal_angle,
            max_inner_angle=self.max_gimbal_angle,
            max_rate=self.gimbal_rate_limit,
            tau=0.05,  # 50ms lag
        )
        throttle = ThrottleActuator(
            rocket=rocket,
            max_thrust_rate=self.max_thrust * 2,  # can go 0-100% in 0.5s
            min_thrust=0.0,
        )
        return Actuators(gimbal=gimbal, throttle=throttle)

    def _get_obs(self) -> np.ndarray:
        """Extract observation from sensors (realistic sensor suite)."""
        gps = self._sensors.GPS.read()  # Horizontal position (x, z)
        alt = self._sensors.Barometer.read()  # Altitude (y)
        vel = self._sensors.VelocitySensor.read()  # TODO: replace with accel integration
        orient = self._sensors.OrientationSensor.read()  # Ground truth + noise
        omega = self._sensors.GyroSensor.read()
        thrust = self._sensors.ThrustSensor.read()  # TODO: replace with commanded thrust

        qw, qx, qy, qz = orient.as_array()

        return np.array(
            [
                gps.x, alt, gps.z,
                vel.x, vel.y, vel.z,
                qw, qx, qy, qz,
                omega.x, omega.y, omega.z,
                thrust,
            ],
            dtype=np.float32,
        )

    def _get_tilt_angle(self) -> float:
        """Compute tilt angle from vertical (radians)."""
        # Use ground truth for termination checks (not observation)
        orient = self._kernel.rocket.state.orientation
        # Nose direction in world frame
        nose_body = RocketVec3(1.0, 0.0, 0.0)
        nose_world = nose_body.to_world(orient)
        # Angle from vertical (+Y)
        cos_angle = max(-1.0, min(1.0, nose_world.y))
        return math.acos(cos_angle)

    def _compute_reward(self, action: np.ndarray) -> float:
        """Compute reward for current state and action."""
        state = self._kernel.rocket.state

        # Altitude error
        alt_error = abs(state.position.y - self.target_altitude)

        # Velocity magnitude
        vel_mag = state.velocity.norm

        # Tilt angle
        tilt = self._get_tilt_angle()

        # Action magnitude (penalize large actions)
        action_mag = np.linalg.norm(action)

        reward = (
            -self.altitude_weight * alt_error
            - self.velocity_weight * vel_mag
            - self.tilt_weight * tilt**2
            - self.action_weight * action_mag
        )

        return float(reward)

    def _check_termination(self) -> Tuple[bool, bool]:
        """Check if episode should terminate. Returns (terminated, truncated)."""
        state = self._kernel.rocket.state
        ground = self._kernel.ground_contact

        # Truncation (time limit)
        truncated = self._step_count >= self.max_episode_steps

        # Termination conditions
        terminated = False

        # Crashed / tipped over
        if ground.tipped_over:
            terminated = True

        # Excessive tilt
        if self._get_tilt_angle() > self.max_tilt:
            terminated = True

        # Excessive velocity
        if state.velocity.norm > self.max_velocity:
            terminated = True

        # Out of bounds
        if state.position.y < self.min_altitude or state.position.y > self.max_altitude:
            terminated = True

        return terminated, truncated

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Sample initial conditions
        init_alt = self.np_random.uniform(*self.init_altitude_range)
        init_vy = self.np_random.uniform(*self.init_velocity_range)
        init_tilt_y = self.np_random.uniform(*self.init_tilt_range)
        init_tilt_z = self.np_random.uniform(*self.init_tilt_range)

        # Create perturbed orientation by composing rotations:
        # Start with upright, then apply tilt about Y and Z axes
        base_rot = get_upright_orientation()._as_scipy_rotation()
        perturb_y = R.from_euler("y", init_tilt_y)
        perturb_z = R.from_euler("z", init_tilt_z)
        combined = perturb_z * perturb_y * base_rot
        x, y, z, w = combined.as_quat()
        init_orient = Orientation.from_wxyz(w, x, y, z)

        init_state = RocketState(
            position=WorldVec3(0.0, init_alt, 0.0),
            velocity=WorldVec3(0.0, init_vy, 0.0),
            orientation=init_orient,
            angular_velocity_body=RocketVec3(0.0, 0.0, 0.0),
            thrust_body=RocketVec3(self.mass * 9.81, 0.0, 0.0),  # hover thrust
        )

        rocket = self._make_rocket(init_state)
        environment = self._make_environment()

        self._kernel = Kernel(rocket, environment)
        self._sensors = Sensors(self._kernel)
        self._actuators = self._make_actuators(rocket)

        self._step_count = 0
        self._time = 0.0

        obs = self._get_obs()
        info = {"time": self._time}

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment."""
        # Unpack and scale action
        throttle = float(action[0]) * self.max_thrust
        gimbal_outer = float(action[1]) * self.max_gimbal_angle
        gimbal_inner = float(action[2]) * self.max_gimbal_angle

        # Apply commands to actuators
        self._actuators._throttle.apply(ThrottleCommand(thrust_N=throttle))
        self._actuators._gimbal.apply(
            GimbalCommand(
                desired_outer_angle=gimbal_outer,
                desired_inner_angle=gimbal_inner,
            )
        )

        # Advance simulation
        self._sensors.set_time(self._time)
        self._actuators.set_time(self._time)
        self._actuators.step(self.dt)
        self._kernel.step(self.dt)

        # Update attitude estimator (fuses accel + gyro + mag)
        self._sensors.update_attitude_estimator(self.dt)

        self._time += self.dt
        self._step_count += 1

        # Get outputs
        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated, truncated = self._check_termination()
        info = {
            "time": self._time,
            "altitude": self._kernel.rocket.state.position.y,
            "tilt": self._get_tilt_angle(),
            "on_ground": self._kernel.ground_contact.on_ground,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render is not implemented (use telemetry logging instead)."""
        pass

    def close(self):
        """Clean up resources."""
        pass


class RocketLandingEnv(gym.Env):
    """
    Gymnasium environment for VTVL rocket landing.

    Observation (15-dim):
        [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz, thrust_mag, fuel_frac]

    Action (3-dim, continuous):
        [throttle, gimbal_outer, gimbal_inner]
        - throttle: 0 to 1 (scaled to max_thrust)
        - gimbal_outer: -1 to 1 (scaled to max_outer_angle)
        - gimbal_inner: -1 to 1 (scaled to max_inner_angle)

    Task: Land softly from altitude with upright attitude.
    Success: on_ground with velocity < max_landing_speed and tilt < max_landing_tilt
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        dt: float = 0.02,
        max_episode_steps: int = 1000,
        # Rocket parameters
        dry_mass: float = 60.0,
        fuel_mass: float = 80.0,
        length: float = 4.0,
        max_thrust: float = 2000.0,
        burn_rate: float = 4.0,  # kg/s at max thrust
        max_gimbal_angle: float = 0.26,  # ~15 degrees
        gimbal_rate_limit: float = 1.0,  # rad/s
        # Initial condition ranges
        init_altitude_range: Tuple[float, float] = (40.0, 60.0),
        init_velocity_range: Tuple[float, float] = (-5.0, 0.0),  # downward
        init_horizontal_range: Tuple[float, float] = (-5.0, 5.0),
        init_tilt_range: Tuple[float, float] = (-0.15, 0.15),
        init_fuel_range: Tuple[float, float] = (0.6, 1.0),  # fraction
        # Landing thresholds (from Kernel.GroundContact)
        max_landing_speed: float = 3.0,  # m/s
        max_landing_tilt: float = 0.26,  # rad (~15 deg)
        # Termination
        max_tilt: float = 1.0,  # ~57 deg, terminate if exceeded mid-flight
        max_altitude: float = 100.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.dt = dt
        self.max_episode_steps = max_episode_steps

        # Rocket config
        self.dry_mass = dry_mass
        self.fuel_mass = fuel_mass
        self.length = length
        self.max_thrust = max_thrust
        self.burn_rate = burn_rate
        self.max_gimbal_angle = max_gimbal_angle
        self.gimbal_rate_limit = gimbal_rate_limit

        # Initial conditions
        self.init_altitude_range = init_altitude_range
        self.init_velocity_range = init_velocity_range
        self.init_horizontal_range = init_horizontal_range
        self.init_tilt_range = init_tilt_range
        self.init_fuel_range = init_fuel_range

        # Landing thresholds
        self.max_landing_speed = max_landing_speed
        self.max_landing_tilt = max_landing_tilt

        # Termination
        self.max_tilt = max_tilt
        # Auto-derive max_altitude from init range if using the default.
        # Must be above the highest possible starting altitude.
        if max_altitude == 100.0 and init_altitude_range[1] > 80.0:
            self.max_altitude = init_altitude_range[1] * 1.5
        else:
            self.max_altitude = max_altitude

        self.render_mode = render_mode

        # Action space: [throttle, gimbal_outer, gimbal_inner]
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )

        # Observation space: [pos(3), vel(3), quat(4), omega(3)] = 13 dims
        # Bounds scale with init range to avoid clipping at high altitudes/speeds
        alt_bound = max(150.0, self.max_altitude * 1.2)
        vel_bound = max(50.0, math.sqrt(2 * 9.81 * self.max_altitude) * 1.2)
        obs_low = np.array(
            [
                -100, -10, -100,             # position (gps_x, baro_alt, gps_z)
                -vel_bound, -vel_bound, -vel_bound,  # velocity
                -1, -1, -1, -1,              # quaternion (from attitude estimator)
                -10, -10, -10,               # angular velocity
            ],
            dtype=np.float32,
        )
        obs_high = np.array(
            [
                100, alt_bound, 100,             # position (alt_bound scales with init range)
                vel_bound, vel_bound, vel_bound, # velocity
                1, 1, 1, 1,                      # quaternion
                10, 10, 10,                      # angular velocity
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high)

        # Initialized in reset()
        self._kernel: Optional[Kernel] = None
        self._sensors: Optional[Sensors] = None
        self._actuators: Optional[Actuators] = None
        self._step_count: int = 0
        self._time: float = 0.0

    def _make_rocket(self, init_state: RocketState, fuel_fraction: float) -> Rocket:
        """Create rocket with fuel model."""
        mass_model = MassModel(components=[
            MassComponent("dry", self.dry_mass, RocketVec3(self.length / 2, 0, 0)),
            FuelComponent(
                "fuel",
                mass=self.fuel_mass * fuel_fraction,
                position=RocketVec3(self.length * 0.3, 0, 0),
                full_mass=self.fuel_mass,
                burn_rate=self.burn_rate,
            ),
        ])

        props = mass_model.to_rocket_props(
            length=self.length,
            thrust_application_offset=RocketVec3(0, 0, 0),
        )

        return Rocket(props=props, state=init_state, mass_model=mass_model)

    def _make_environment(self) -> Environment:
        """Create world environment."""
        return Environment(
            gravity=WorldVec3(0.0, -9.81, 0.0),
            wind=WorldVec3(0.0, 0.0, 0.0),
            air_density=1.225,
        )

    def _make_actuators(self, rocket: Rocket) -> Actuators:
        """Create actuator bundle."""
        gimbal = GimbalActuator(
            rocket=rocket,
            max_outer_angle=self.max_gimbal_angle,
            max_inner_angle=self.max_gimbal_angle,
            max_rate=self.gimbal_rate_limit,
            tau=0.05,
        )
        throttle = ThrottleActuator(
            rocket=rocket,
            max_thrust_rate=self.max_thrust * 2,
            min_thrust=0.0,
        )
        return Actuators(gimbal=gimbal, throttle=throttle)

    def _get_obs(self) -> np.ndarray:
        """
        Extract observation from sensors (realistic sensor suite).

        13-dim observation:
            [gps_x, baro_alt, gps_z,     # position (3)
             vel_x, vel_y, vel_z,         # velocity (3)
             qw, qx, qy, qz,              # orientation from AHRS (4)
             omega_x, omega_y, omega_z]   # angular velocity (3)

        No thrust or fuel - agent doesn't directly observe these.
        """
        gps = self._sensors.GPS.read()  # Horizontal position (x, z)
        alt = self._sensors.Barometer.read()  # Altitude (y)
        vel = self._sensors.VelocitySensor.read()  # TODO: replace with accel integration
        orient = self._sensors.OrientationSensor.read()  # Ground truth + noise
        omega = self._sensors.GyroSensor.read()

        qw, qx, qy, qz = orient.as_array()

        return np.array(
            [
                gps.x, alt, gps.z,
                vel.x, vel.y, vel.z,
                qw, qx, qy, qz,
                omega.x, omega.y, omega.z,
            ],
            dtype=np.float32,
        )

    def _get_tilt_angle(self) -> float:
        """Compute tilt angle from vertical (radians)."""
        # Use ground truth for termination checks (not observation)
        orient = self._kernel.rocket.state.orientation
        nose_body = RocketVec3(1.0, 0.0, 0.0)
        nose_world = nose_body.to_world(orient)
        cos_angle = max(-1.0, min(1.0, nose_world.y))
        return math.acos(cos_angle)

    def _compute_reward(self, action: np.ndarray) -> Tuple[float, dict]:
        """
        Compute reward for landing task.

        Reward shaping:
        - Penalty for distance from target (0, 0)
        - Penalty for altitude (encourages descent)
        - Penalty for high velocity (encourages controlled descent)
        - Penalty for tilt (encourages upright)
        - Small penalty for fuel use (encourages efficiency)
        - Large bonus for successful landing at target
        - Large penalty for crash
        """
        state = self._kernel.rocket.state
        ground = self._kernel.ground_contact
        tilt = self._get_tilt_angle()
        vel_mag = state.velocity.norm
        alt = state.position.y
        # Horizontal distance from landing target (0, 0)
        horiz_dist = math.hypot(state.position.x, state.position.z)

        reward = 0.0
        info = {}

        if ground.on_ground:
            if ground.crashed or ground.tipped_over:
                # Crash penalty
                reward = -100.0
                info["outcome"] = "crash"
            else:
                # Successful landing bonus
                landing_vel = vel_mag
                landing_tilt = tilt
                # Score components: velocity, tilt, and position accuracy
                vel_score = max(0, 1 - landing_vel / self.max_landing_speed)
                tilt_score = max(0, 1 - landing_tilt / self.max_landing_tilt)
                # Position score: full points within 1m, decreasing beyond
                pos_score = max(0, 1 - horiz_dist / 10.0)
                reward = 100.0 * vel_score * tilt_score * (0.5 + 0.5 * pos_score)
                info["outcome"] = "landed"
                info["landing_vel"] = landing_vel
                info["landing_tilt"] = math.degrees(landing_tilt)
                info["landing_dist"] = horiz_dist
        else:
            # Shaping reward during flight
            # Reward for being closer to ground
            alt_reward = -0.01 * alt

            # Penalty for horizontal distance from target
            horiz_penalty = -0.005 * horiz_dist

            # Penalty for high velocity (quadratic)
            vel_penalty = -0.001 * vel_mag**2

            # Penalty for tilt (quadratic)
            tilt_penalty = -0.1 * tilt**2

            # Small fuel penalty (encourages efficiency)
            fuel_penalty = -0.01 * action[0]  # throttle

            reward = alt_reward + horiz_penalty + vel_penalty + tilt_penalty + fuel_penalty

        return reward, info

    def _check_termination(self) -> Tuple[bool, bool]:
        """Check if episode should terminate."""
        state = self._kernel.rocket.state
        ground = self._kernel.ground_contact

        truncated = self._step_count >= self.max_episode_steps

        terminated = False

        # Landed (success or crash)
        if ground.on_ground:
            terminated = True

        # Excessive tilt mid-flight
        if self._get_tilt_angle() > self.max_tilt:
            terminated = True

        # Out of fuel
        if self._kernel.rocket.mass_model.fuel_fraction <= 0:
            terminated = True

        # Out of bounds (too high)
        if state.position.y > self.max_altitude:
            terminated = True

        return terminated, truncated

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Sample initial conditions
        init_alt = self.np_random.uniform(*self.init_altitude_range)
        init_vy = self.np_random.uniform(*self.init_velocity_range)
        init_vx = self.np_random.uniform(*self.init_horizontal_range) * 0.1
        init_vz = self.np_random.uniform(*self.init_horizontal_range) * 0.1
        init_x = self.np_random.uniform(*self.init_horizontal_range)
        init_z = self.np_random.uniform(*self.init_horizontal_range)
        init_tilt_y = self.np_random.uniform(*self.init_tilt_range)
        init_tilt_z = self.np_random.uniform(*self.init_tilt_range)
        init_fuel = self.np_random.uniform(*self.init_fuel_range)

        # Perturbed orientation
        base_rot = get_upright_orientation()._as_scipy_rotation()
        perturb_y = R.from_euler("y", init_tilt_y)
        perturb_z = R.from_euler("z", init_tilt_z)
        combined = perturb_z * perturb_y * base_rot
        x, y, z, w = combined.as_quat()
        init_orient = Orientation.from_wxyz(w, x, y, z)

        init_state = RocketState(
            position=WorldVec3(init_x, init_alt, init_z),
            velocity=WorldVec3(init_vx, init_vy, init_vz),
            orientation=init_orient,
            angular_velocity_body=RocketVec3(0.0, 0.0, 0.0),
            thrust_body=RocketVec3(0.0, 0.0, 0.0),
        )

        rocket = self._make_rocket(init_state, init_fuel)
        environment = self._make_environment()

        self._kernel = Kernel(rocket, environment)
        self._sensors = Sensors(self._kernel)
        self._actuators = self._make_actuators(rocket)

        self._step_count = 0
        self._time = 0.0

        obs = self._get_obs()
        info = {
            "time": self._time,
            "init_altitude": init_alt,
            "init_fuel": init_fuel,
        }

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment."""
        # Scale action
        throttle = float(action[0]) * self.max_thrust
        gimbal_outer = float(action[1]) * self.max_gimbal_angle
        gimbal_inner = float(action[2]) * self.max_gimbal_angle

        # Apply commands
        self._actuators._throttle.apply(ThrottleCommand(thrust_N=throttle))
        self._actuators._gimbal.apply(
            GimbalCommand(
                desired_outer_angle=gimbal_outer,
                desired_inner_angle=gimbal_inner,
            )
        )

        # Advance simulation
        self._sensors.set_time(self._time)
        self._actuators.set_time(self._time)
        self._actuators.step(self.dt)
        self._kernel.step(self.dt)

        # Update attitude estimator (fuses accel + gyro + mag)
        self._sensors.update_attitude_estimator(self.dt)

        self._time += self.dt
        self._step_count += 1

        # Get outputs
        obs = self._get_obs()
        reward, reward_info = self._compute_reward(action)
        terminated, truncated = self._check_termination()

        pos = self._kernel.rocket.state.position
        info = {
            "time": self._time,
            "altitude": pos.y,
            "horiz_dist": math.hypot(pos.x, pos.z),
            "velocity": self._kernel.rocket.state.velocity.norm,
            "tilt_deg": math.degrees(self._get_tilt_angle()),
            "fuel": self._kernel.rocket.mass_model.fuel_fraction,
            "on_ground": self._kernel.ground_contact.on_ground,
            **reward_info,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass


@dataclass
class DomainRandomization:
    """
    Per-episode randomization of physical parameters for robust RL training.

    Motivated by Dresia (2025): without domain randomization, RL controllers
    overfit to the training distribution and fail on real hardware. With DR,
    the policy learns representations invariant to parameter variation.

    Scale parameters are multiplicative: sampled_value = nominal * uniform(low, high).
    Noise parameters specify the std dev range: std = uniform(low, high).
    Default (1.0, 1.0) or (0.0, 0.0) means no randomization.

    Example — moderate DR for landing:
        DomainRandomization(
            dry_mass_scale=(0.9, 1.1),      # ±10% mass
            max_thrust_scale=(0.95, 1.05),   # ±5% thrust
            gimbal_tau_scale=(0.7, 1.3),     # ±30% actuator lag
            wind_speed_range=(0.0, 3.0),     # 0–3 m/s wind
            gyro_noise_std=(0.0, 0.01),      # 0–0.01 rad/s gyro noise
        )
    """

    # Rocket physics (multiplicative scale on nominal values)
    dry_mass_scale: Tuple[float, float] = (1.0, 1.0)
    fuel_mass_scale: Tuple[float, float] = (1.0, 1.0)
    max_thrust_scale: Tuple[float, float] = (1.0, 1.0)
    burn_rate_scale: Tuple[float, float] = (1.0, 1.0)
    length_scale: Tuple[float, float] = (1.0, 1.0)

    # Actuators (multiplicative scale)
    gimbal_rate_scale: Tuple[float, float] = (1.0, 1.0)
    gimbal_tau_scale: Tuple[float, float] = (1.0, 1.0)

    # Environment
    wind_speed_range: Tuple[float, float] = (0.0, 0.0)  # m/s, horizontal magnitude
    gravity_scale: Tuple[float, float] = (1.0, 1.0)
    air_density_scale: Tuple[float, float] = (1.0, 1.0)

    # Sensor noise (std dev sampled uniformly from range)
    gps_noise_std: Tuple[float, float] = (0.0, 0.0)        # meters
    baro_noise_std: Tuple[float, float] = (0.0, 0.0)       # meters
    velocity_noise_std: Tuple[float, float] = (0.0, 0.0)   # m/s
    orientation_noise_std: Tuple[float, float] = (0.0, 0.0) # radians
    gyro_noise_std: Tuple[float, float] = (0.0, 0.0)       # rad/s
    gyro_bias_range: Tuple[float, float] = (0.0, 0.0)      # rad/s, constant per episode


class RocketLandingEnvV2(gym.Env):
    """
    Improved gymnasium environment for VTVL rocket landing.

    Key improvements over V1:
    - Additive terminal reward (not multiplicative)
    - Rebalanced flight penalties (no fuel penalty)
    - Better reward shaping for learning
    - Optional domain randomization (DomainRandomization config)

    Observation (13-dim):
        [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]

    Action (3-dim, continuous):
        [throttle, gimbal_outer, gimbal_inner]
        - throttle: 0 to 1 (scaled to max_thrust)
        - gimbal_outer: -1 to 1 (scaled to max_outer_angle)
        - gimbal_inner: -1 to 1 (scaled to max_inner_angle)

    Reward function:
        Landed successfully: 50 + velocity_bonus(0-30) + tilt_bonus(0-15) + position_bonus(0-5)
        Crashed: -100
        In flight: -0.005*alt - 0.005*horiz_dist - 0.01*vel - 0.01*tilt
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        dt: float = 0.02,
        max_episode_steps: int = 1000,
        # Rocket parameters
        dry_mass: float = 60.0,
        fuel_mass: float = 80.0,
        length: float = 4.0,
        max_thrust: float = 2000.0,
        burn_rate: float = 4.0,  # kg/s at max thrust
        max_gimbal_angle: float = 0.26,  # ~15 degrees
        gimbal_rate_limit: float = 1.0,  # rad/s
        # Initial condition ranges (default to easier curriculum start)
        init_altitude_range: Tuple[float, float] = (10.0, 60.0),
        init_velocity_range: Tuple[float, float] = (-8.0, 0.0),  # downward
        init_horizontal_range: Tuple[float, float] = (-5.0, 5.0),
        init_tilt_range: Tuple[float, float] = (-0.15, 0.15),
        init_fuel_range: Tuple[float, float] = (0.6, 1.0),  # fraction
        # Landing thresholds
        max_landing_speed: float = 3.0,  # m/s
        max_landing_tilt: float = 0.26,  # rad (~15 deg)
        # Termination
        max_tilt: float = 1.0,  # ~57 deg, terminate if exceeded mid-flight
        max_altitude: float = 100.0,
        # Domain randomization
        domain_randomization: Optional[DomainRandomization] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.dt = dt
        self.max_episode_steps = max_episode_steps

        # Nominal rocket config (DR samples around these values)
        self.dry_mass = dry_mass
        self.fuel_mass = fuel_mass
        self.length = length
        self.max_thrust = max_thrust
        self.burn_rate = burn_rate
        self.max_gimbal_angle = max_gimbal_angle
        self.gimbal_rate_limit = gimbal_rate_limit

        # Initial conditions
        self.init_altitude_range = init_altitude_range
        self.init_velocity_range = init_velocity_range
        self.init_horizontal_range = init_horizontal_range
        self.init_tilt_range = init_tilt_range
        self.init_fuel_range = init_fuel_range

        # Landing thresholds
        self.max_landing_speed = max_landing_speed
        self.max_landing_tilt = max_landing_tilt

        # Termination
        self.max_tilt = max_tilt
        if max_altitude == 100.0 and init_altitude_range[1] > 80.0:
            self.max_altitude = init_altitude_range[1] * 1.5
        else:
            self.max_altitude = max_altitude

        # Domain randomization
        self.domain_randomization = domain_randomization

        self.render_mode = render_mode

        # Action space: [throttle, gimbal_outer, gimbal_inner]
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )

        # Observation space: [pos(3), vel(3), quat(4), omega(3)] = 13 dims
        # Bounds scale with init range to avoid clipping at high altitudes/speeds
        alt_bound = max(150.0, self.max_altitude * 1.2)
        vel_bound = max(50.0, math.sqrt(2 * 9.81 * self.max_altitude) * 1.2)
        obs_low = np.array(
            [
                -100, -10, -100,             # position (gps_x, baro_alt, gps_z)
                -vel_bound, -vel_bound, -vel_bound,  # velocity
                -1, -1, -1, -1,              # quaternion (from attitude estimator)
                -10, -10, -10,               # angular velocity
            ],
            dtype=np.float32,
        )
        obs_high = np.array(
            [
                100, alt_bound, 100,             # position (alt_bound scales with init range)
                vel_bound, vel_bound, vel_bound, # velocity
                1, 1, 1, 1,                      # quaternion
                10, 10, 10,                      # angular velocity
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high)

        # Initialized in reset()
        self._kernel: Optional[Kernel] = None
        self._sensors: Optional[Sensors] = None
        self._actuators: Optional[Actuators] = None
        self._step_count: int = 0
        self._time: float = 0.0
        self._ep_params: Optional[dict] = None  # Current episode's DR parameters

    def _sample_episode_params(self) -> dict:
        """
        Sample physical parameters for this episode.

        If domain_randomization is None, returns nominal values.
        Otherwise, scales each parameter by a uniform random factor.
        All sampled values are stored in info for logging.
        """
        dr = self.domain_randomization
        rng = self.np_random

        if dr is None:
            return {
                "dry_mass": self.dry_mass,
                "fuel_mass": self.fuel_mass,
                "length": self.length,
                "max_thrust": self.max_thrust,
                "burn_rate": self.burn_rate,
                "gimbal_rate": self.gimbal_rate_limit,
                "gimbal_tau": 0.05,
                "gravity": 9.81,
                "air_density": 1.225,
                "wind": WorldVec3(0.0, 0.0, 0.0),
                "gps_noise_std": 0.0,
                "baro_noise_std": 0.0,
                "velocity_noise_std": 0.0,
                "orientation_noise_std": 0.0,
                "gyro_noise_std": 0.0,
                "gyro_bias": 0.0,
            }

        # Wind: random horizontal direction, sampled magnitude
        wind_speed = rng.uniform(*dr.wind_speed_range)
        wind_angle = rng.uniform(0, 2 * math.pi)
        wind = WorldVec3(
            wind_speed * math.cos(wind_angle),
            0.0,
            wind_speed * math.sin(wind_angle),
        )

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

    def _make_rocket(self, init_state: RocketState, fuel_fraction: float,
                     ep: Optional[dict] = None) -> Rocket:
        """Create rocket with fuel model, using episode parameters if provided."""
        dry_mass = ep["dry_mass"] if ep else self.dry_mass
        fuel_mass = ep["fuel_mass"] if ep else self.fuel_mass
        length = ep["length"] if ep else self.length
        burn_rate = ep["burn_rate"] if ep else self.burn_rate

        mass_model = MassModel(components=[
            MassComponent("dry", dry_mass, RocketVec3(length / 2, 0, 0)),
            FuelComponent(
                "fuel",
                mass=fuel_mass * fuel_fraction,
                position=RocketVec3(length * 0.3, 0, 0),
                full_mass=fuel_mass,
                burn_rate=burn_rate,
            ),
        ])

        props = mass_model.to_rocket_props(
            length=length,
            thrust_application_offset=RocketVec3(0, 0, 0),
        )

        return Rocket(props=props, state=init_state, mass_model=mass_model)

    def _make_environment(self, ep: Optional[dict] = None) -> Environment:
        """Create world environment, using episode parameters if provided."""
        if ep:
            return Environment(
                gravity=WorldVec3(0.0, -ep["gravity"], 0.0),
                wind=ep["wind"],
                air_density=ep["air_density"],
            )
        return Environment(
            gravity=WorldVec3(0.0, -9.81, 0.0),
            wind=WorldVec3(0.0, 0.0, 0.0),
            air_density=1.225,
        )

    def _make_actuators(self, rocket: Rocket, ep: Optional[dict] = None) -> Actuators:
        """Create actuator bundle, using episode parameters if provided."""
        gimbal_rate = ep["gimbal_rate"] if ep else self.gimbal_rate_limit
        gimbal_tau = ep["gimbal_tau"] if ep else 0.05
        max_thrust = ep["max_thrust"] if ep else self.max_thrust

        gimbal = GimbalActuator(
            rocket=rocket,
            max_outer_angle=self.max_gimbal_angle,
            max_inner_angle=self.max_gimbal_angle,
            max_rate=gimbal_rate,
            tau=gimbal_tau,
        )
        throttle = ThrottleActuator(
            rocket=rocket,
            max_thrust_rate=max_thrust * 2,
            min_thrust=0.0,
        )
        return Actuators(gimbal=gimbal, throttle=throttle)

    def _make_sensors(self, kernel: Kernel, ep: Optional[dict] = None) -> Sensors:
        """
        Create sensor bundle with per-episode noise levels.

        When domain randomization is active, each episode gets different
        sensor noise characteristics — simulating hardware variation.
        """
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
                "x": GaussianNoise(std=ep["velocity_noise_std"]),
                "y": GaussianNoise(std=ep["velocity_noise_std"]),
                "z": GaussianNoise(std=ep["velocity_noise_std"]),
            }

        orientation_errors = None
        if ep["orientation_noise_std"] > 0:
            orientation_errors = {
                "roll": GaussianNoise(std=ep["orientation_noise_std"]),
                "pitch": GaussianNoise(std=ep["orientation_noise_std"]),
                "yaw": GaussianNoise(std=ep["orientation_noise_std"]),
            }

        gyro_errors = None
        if ep["gyro_noise_std"] > 0 or ep["gyro_bias"] != 0:
            gyro_errors = {}
            for ch in ("x", "y", "z"):
                errors = []
                if ep["gyro_bias"] != 0:
                    # Per-axis bias: constant for the episode, sampled in [-bias, +bias]
                    errors.append(Bias(offset=self.np_random.uniform(
                        -abs(ep["gyro_bias"]), abs(ep["gyro_bias"])
                    )))
                if ep["gyro_noise_std"] > 0:
                    errors.append(GaussianNoise(std=ep["gyro_noise_std"]))
                if len(errors) == 1:
                    gyro_errors[ch] = errors[0]
                elif len(errors) > 1:
                    gyro_errors[ch] = CompositeError(errors=errors)

        return Sensors(
            kernel,
            gps_errors=gps_errors,
            velocity_errors=velocity_errors,
            orientation_errors=orientation_errors,
            gyro_errors=gyro_errors,
            baro_errors=baro_errors,
        )

    def _get_obs(self) -> np.ndarray:
        """
        Extract observation from sensors (realistic sensor suite).

        13-dim observation:
            [gps_x, baro_alt, gps_z,     # position (3)
             vel_x, vel_y, vel_z,         # velocity (3)
             qw, qx, qy, qz,              # orientation from AHRS (4)
             omega_x, omega_y, omega_z]   # angular velocity (3)
        """
        gps = self._sensors.GPS.read()  # Horizontal position (x, z)
        alt = self._sensors.Barometer.read()  # Altitude (y)
        vel = self._sensors.VelocitySensor.read()
        orient = self._sensors.OrientationSensor.read()  # Ground truth + noise
        omega = self._sensors.GyroSensor.read()

        qw, qx, qy, qz = orient.as_array()

        return np.array(
            [
                gps.x, alt, gps.z,
                vel.x, vel.y, vel.z,
                qw, qx, qy, qz,
                omega.x, omega.y, omega.z,
            ],
            dtype=np.float32,
        )

    def _get_tilt_angle(self) -> float:
        """Compute tilt angle from vertical (radians)."""
        orient = self._kernel.rocket.state.orientation
        nose_body = RocketVec3(1.0, 0.0, 0.0)
        nose_world = nose_body.to_world(orient)
        cos_angle = max(-1.0, min(1.0, nose_world.y))
        return math.acos(cos_angle)

    def _compute_reward(self, action: np.ndarray) -> Tuple[float, dict]:
        """
        Compute reward with ADDITIVE terminal bonuses (not multiplicative).

        Terminal rewards:
        - Landed successfully: 50 + velocity_bonus + tilt_bonus + position_bonus
        - Crashed: -100

        Flight shaping (per step):
        - Small penalties for altitude, distance, velocity, tilt
        - NO fuel penalty (let the agent learn efficiency naturally)
        """
        state = self._kernel.rocket.state
        ground = self._kernel.ground_contact
        tilt = self._get_tilt_angle()
        vel_mag = state.velocity.norm
        alt = state.position.y
        horiz_dist = math.hypot(state.position.x, state.position.z)

        reward = 0.0
        info = {}

        if ground.on_ground:
            if ground.crashed or ground.tipped_over:
                # Crash penalty (fixed, not scaled)
                reward = -100.0
                info["outcome"] = "crash"
            else:
                # Successful landing - ADDITIVE bonuses
                reward = 50.0  # Base landing reward

                # Velocity bonus: 30 points for soft landing (vel < 0.5), linear decay
                vel_bonus = 30.0 * max(0.0, 1.0 - vel_mag / 3.0)
                reward += vel_bonus

                # Tilt bonus: 15 points for upright (tilt < 0.05), linear decay
                tilt_bonus = 15.0 * max(0.0, 1.0 - tilt / 0.26)
                reward += tilt_bonus

                # Position bonus: 5 points for on-target, linear decay
                pos_bonus = 5.0 * max(0.0, 1.0 - horiz_dist / 10.0)
                reward += pos_bonus

                info["outcome"] = "landed"
                info["landing_vel"] = vel_mag
                info["landing_tilt"] = math.degrees(tilt)
                info["landing_dist"] = horiz_dist
                info["vel_bonus"] = vel_bonus
                info["tilt_bonus"] = tilt_bonus
                info["pos_bonus"] = pos_bonus
        else:
            # Shaping reward during flight - REBALANCED
            # Goal: guide toward target while descending

            # Altitude penalty: encourage descent
            alt_penalty = -0.005 * alt

            # Horizontal distance penalty: encourage moving toward target
            horiz_penalty = -0.005 * horiz_dist

            # Velocity penalty: discourage excessive speed
            vel_penalty = -0.01 * vel_mag

            # Tilt penalty: encourage staying upright
            tilt_penalty = -0.01 * tilt

            # NO fuel/throttle penalty - removed to encourage proper braking

            reward = alt_penalty + horiz_penalty + vel_penalty + tilt_penalty

        return reward, info

    def _check_termination(self) -> Tuple[bool, bool]:
        """Check if episode should terminate."""
        state = self._kernel.rocket.state
        ground = self._kernel.ground_contact

        truncated = self._step_count >= self.max_episode_steps

        terminated = False

        # Landed (success or crash)
        if ground.on_ground:
            terminated = True

        # Excessive tilt mid-flight
        if self._get_tilt_angle() > self.max_tilt:
            terminated = True

        # Out of fuel
        if self._kernel.rocket.mass_model.fuel_fraction <= 0:
            terminated = True

        # Out of bounds (too high)
        if state.position.y > self.max_altitude:
            terminated = True

        return terminated, truncated

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Allow options to override initial conditions (for curriculum)
        init_altitude_range = self.init_altitude_range
        init_velocity_range = self.init_velocity_range
        init_horizontal_range = self.init_horizontal_range
        init_tilt_range = self.init_tilt_range
        init_fuel_range = self.init_fuel_range

        if options:
            init_altitude_range = options.get("altitude_range", init_altitude_range)
            init_velocity_range = options.get("velocity_range", init_velocity_range)
            init_horizontal_range = options.get("horizontal_range", init_horizontal_range)
            init_tilt_range = options.get("tilt_range", init_tilt_range)
            init_fuel_range = options.get("fuel_range", init_fuel_range)

        # Sample initial conditions
        init_alt = self.np_random.uniform(*init_altitude_range)
        init_vy = self.np_random.uniform(*init_velocity_range)
        init_vx = self.np_random.uniform(*init_horizontal_range) * 0.1
        init_vz = self.np_random.uniform(*init_horizontal_range) * 0.1
        init_x = self.np_random.uniform(*init_horizontal_range)
        init_z = self.np_random.uniform(*init_horizontal_range)
        init_tilt_y = self.np_random.uniform(*init_tilt_range)
        init_tilt_z = self.np_random.uniform(*init_tilt_range)
        init_fuel = self.np_random.uniform(*init_fuel_range)

        # Perturbed orientation
        base_rot = get_upright_orientation()._as_scipy_rotation()
        perturb_y = R.from_euler("y", init_tilt_y)
        perturb_z = R.from_euler("z", init_tilt_z)
        combined = perturb_z * perturb_y * base_rot
        x, y, z, w = combined.as_quat()
        init_orient = Orientation.from_wxyz(w, x, y, z)

        init_state = RocketState(
            position=WorldVec3(init_x, init_alt, init_z),
            velocity=WorldVec3(init_vx, init_vy, init_vz),
            orientation=init_orient,
            angular_velocity_body=RocketVec3(0.0, 0.0, 0.0),
            thrust_body=RocketVec3(0.0, 0.0, 0.0),
        )

        # Sample episode parameters (domain randomization)
        ep = self._sample_episode_params()
        self._ep_params = ep

        rocket = self._make_rocket(init_state, init_fuel, ep)
        environment = self._make_environment(ep)

        self._kernel = Kernel(rocket, environment)
        self._sensors = self._make_sensors(self._kernel, ep)
        self._actuators = self._make_actuators(rocket, ep)

        self._step_count = 0
        self._time = 0.0

        obs = self._get_obs()
        info = {
            "time": self._time,
            "init_altitude": init_alt,
            "init_fuel": init_fuel,
        }
        # Log DR params for analysis (only when DR is active)
        if self.domain_randomization is not None:
            info["dr_params"] = {
                k: (v if not isinstance(v, WorldVec3) else [v.x, v.y, v.z])
                for k, v in ep.items()
            }

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment."""
        # Scale action — use episode max_thrust if DR is active
        ep_thrust = self._ep_params["max_thrust"] if self._ep_params else self.max_thrust
        throttle = float(action[0]) * ep_thrust
        gimbal_outer = float(action[1]) * self.max_gimbal_angle
        gimbal_inner = float(action[2]) * self.max_gimbal_angle

        # Apply commands
        self._actuators._throttle.apply(ThrottleCommand(thrust_N=throttle))
        self._actuators._gimbal.apply(
            GimbalCommand(
                desired_outer_angle=gimbal_outer,
                desired_inner_angle=gimbal_inner,
            )
        )

        # Advance simulation
        self._sensors.set_time(self._time)
        self._actuators.set_time(self._time)
        self._actuators.step(self.dt)
        self._kernel.step(self.dt)

        # Update attitude estimator (fuses accel + gyro + mag)
        self._sensors.update_attitude_estimator(self.dt)

        self._time += self.dt
        self._step_count += 1

        # Get outputs
        obs = self._get_obs()
        reward, reward_info = self._compute_reward(action)
        terminated, truncated = self._check_termination()

        pos = self._kernel.rocket.state.position
        info = {
            "time": self._time,
            "altitude": pos.y,
            "horiz_dist": math.hypot(pos.x, pos.z),
            "velocity": self._kernel.rocket.state.velocity.norm,
            "tilt_deg": math.degrees(self._get_tilt_angle()),
            "fuel": self._kernel.rocket.mass_model.fuel_fraction,
            "on_ground": self._kernel.ground_contact.on_ground,
            **reward_info,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
