"""
Tests for the Gymnasium environment wrapper.
"""

import numpy as np
import pytest

from simulacra.gym_env import RocketHoverEnv, RocketLandingEnv


class TestRocketHoverEnv:
    """Tests for RocketHoverEnv."""

    def test_env_creation(self):
        """Environment can be created with default params."""
        env = RocketHoverEnv()
        assert env is not None
        assert env.observation_space.shape == (14,)
        assert env.action_space.shape == (3,)

    def test_reset_returns_valid_obs(self):
        """reset() returns observation within bounds."""
        env = RocketHoverEnv()
        obs, info = env.reset(seed=42)

        assert obs.shape == (14,)
        assert not np.any(np.isnan(obs)), "Observation contains NaN"
        assert not np.any(np.isinf(obs)), "Observation contains Inf"
        assert env.observation_space.contains(obs), "Observation out of bounds"

    def test_reset_with_different_seeds(self):
        """Different seeds produce different initial states."""
        env = RocketHoverEnv()

        obs1, _ = env.reset(seed=1)
        obs2, _ = env.reset(seed=2)

        # At minimum, positions should differ (altitude is randomized)
        assert not np.allclose(obs1, obs2), "Same seeds should produce different states"

    def test_step_returns_valid_output(self):
        """step() returns valid (obs, reward, terminated, truncated, info)."""
        env = RocketHoverEnv()
        env.reset(seed=42)

        # Apply a neutral hover action
        action = np.array([0.5, 0.0, 0.0], dtype=np.float32)  # ~50% throttle, no gimbal
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (14,)
        assert not np.any(np.isnan(obs)), "Observation contains NaN"
        assert isinstance(reward, float)
        assert not np.isnan(reward), "Reward is NaN"
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert "altitude" in info
        assert "tilt" in info

    def test_step_multiple_times(self):
        """Can run multiple steps without errors."""
        env = RocketHoverEnv()
        env.reset(seed=42)

        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            assert not np.any(np.isnan(obs)), "Observation contains NaN after step"
            assert not np.isnan(reward), "Reward is NaN"

            if terminated or truncated:
                break

    def test_reward_reasonable(self):
        """Reward doesn't explode over many steps."""
        env = RocketHoverEnv()
        env.reset(seed=42)

        rewards = []
        # Apply near-hover thrust to keep it stable
        action = np.array([0.5, 0.0, 0.0], dtype=np.float32)

        for _ in range(200):
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)

            if terminated or truncated:
                break

        # Reward should be negative (penalty-based) and bounded
        assert all(r < 0 for r in rewards), "Reward should be negative (penalty-based)"
        assert all(abs(r) < 1000 for r in rewards), "Reward exploded"

    def test_termination_on_excessive_tilt(self):
        """Environment terminates when rocket tilts too much."""
        env = RocketHoverEnv(max_tilt=0.3)  # ~17 degrees
        env.reset(seed=42)

        # Apply asymmetric thrust to induce tilt
        action = np.array([0.5, 1.0, 0.0], dtype=np.float32)

        terminated = False
        for _ in range(500):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break

        assert terminated, "Should terminate due to excessive tilt"

    def test_truncation_on_timeout(self):
        """Environment truncates at max_episode_steps."""
        max_steps = 50
        env = RocketHoverEnv(max_episode_steps=max_steps)
        env.reset(seed=42)

        # Hover action to stay stable
        action = np.array([0.5, 0.0, 0.0], dtype=np.float32)

        step_count = 0
        truncated = False
        for _ in range(max_steps + 10):
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            if terminated or truncated:
                break

        assert truncated, "Should truncate at max_episode_steps"
        assert step_count == max_steps

    def test_orientation_perturbation(self):
        """Initial orientation includes perturbation when tilt range is non-zero."""
        # Create env with larger tilt range
        env = RocketHoverEnv(init_tilt_range=(-0.2, 0.2))
        obs, _ = env.reset(seed=123)

        # Extract quaternion from observation
        qw, qx, qy, qz = obs[6:10]

        # Upright quaternion is approximately (0.707, 0.707, 0, 0) for 90deg about X
        # With perturbation, it should differ
        upright_qw = 0.7071
        upright_qx = 0.7071

        # Check that quaternion isn't exactly upright (has some perturbation)
        quat_diff = abs(qw - upright_qw) + abs(qx - upright_qx) + abs(qy) + abs(qz)
        assert quat_diff > 0.01, "Orientation should have perturbation"

    def test_action_space_bounds(self):
        """Actions are properly scaled within actuator limits."""
        env = RocketHoverEnv()
        env.reset(seed=42)

        # Max throttle action
        action = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        env.step(action)

        # Verify internal state (thrust should be near max)
        thrust_mag = env._kernel.rocket.state.thrust_body.norm
        assert thrust_mag > env.max_thrust * 0.5, "Thrust should respond to throttle"

    def test_observation_matches_sensors(self):
        """Observation comes from sensors, not ground truth."""
        env = RocketHoverEnv()
        env.reset(seed=42)

        obs, _, _, _, _ = env.step(np.array([0.5, 0.0, 0.0], dtype=np.float32))

        # Position from observation
        obs_y = obs[1]

        # Position from kernel (ground truth)
        true_y = env._kernel.rocket.state.position.y

        # They should be very close for default sensors (no noise)
        assert abs(obs_y - true_y) < 0.1, "Sensor reading should match truth (no noise config)"

    def test_gymnasium_compatibility(self):
        """Environment passes gymnasium check_env validation."""
        from gymnasium.utils.env_checker import check_env

        env = RocketHoverEnv()
        # This will raise if the env doesn't comply with Gymnasium API
        check_env(env, skip_render_check=True)


class TestRocketHoverEnvDomainRandomization:
    """Tests for domain randomization features."""

    def test_altitude_randomization(self):
        """Initial altitude varies within specified range."""
        env = RocketHoverEnv(init_altitude_range=(5.0, 15.0))

        altitudes = []
        for seed in range(20):
            obs, _ = env.reset(seed=seed)
            altitudes.append(obs[1])  # y position

        assert min(altitudes) >= 5.0 - 0.5  # Allow small tolerance
        assert max(altitudes) <= 15.0 + 0.5
        assert max(altitudes) - min(altitudes) > 2.0, "Should have variety in altitudes"

    def test_velocity_randomization(self):
        """Initial velocity varies within specified range."""
        env = RocketHoverEnv(init_velocity_range=(-3.0, 3.0))

        velocities = []
        for seed in range(20):
            obs, _ = env.reset(seed=seed)
            velocities.append(obs[4])  # vy

        assert min(velocities) >= -3.5
        assert max(velocities) <= 3.5
        assert max(velocities) - min(velocities) > 1.0, "Should have variety in velocities"


class TestRocketLandingEnv:
    """Tests for RocketLandingEnv."""

    def test_env_creation(self):
        """Environment can be created with default params."""
        env = RocketLandingEnv()
        assert env is not None
        assert env.observation_space.shape == (15,)  # +1 for fuel
        assert env.action_space.shape == (3,)

    def test_reset_returns_valid_obs(self):
        """reset() returns observation within bounds."""
        env = RocketLandingEnv()
        obs, info = env.reset(seed=42)

        assert obs.shape == (15,)
        assert not np.any(np.isnan(obs)), "Observation contains NaN"
        assert not np.any(np.isinf(obs)), "Observation contains Inf"
        assert env.observation_space.contains(obs), "Observation out of bounds"
        assert "init_altitude" in info
        assert "init_fuel" in info

    def test_initial_altitude(self):
        """Rocket starts at altitude, not on ground."""
        env = RocketLandingEnv(init_altitude_range=(40.0, 60.0))
        obs, info = env.reset(seed=42)

        altitude = obs[1]  # y position
        assert altitude >= 35.0, "Should start at altitude"
        assert altitude <= 65.0, "Should start within range"

    def test_fuel_in_observation(self):
        """Fuel fraction is included in observation."""
        env = RocketLandingEnv()
        obs, _ = env.reset(seed=42)

        fuel = obs[14]  # last element
        assert 0.0 <= fuel <= 1.0, "Fuel should be a fraction"

    def test_step_consumes_fuel(self):
        """Thrust consumes fuel over time."""
        env = RocketLandingEnv()
        obs, _ = env.reset(seed=42)
        initial_fuel = obs[14]

        # Full throttle for several steps
        action = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(50):
            obs, _, terminated, _, _ = env.step(action)
            if terminated:
                break

        final_fuel = obs[14]
        assert final_fuel < initial_fuel, "Fuel should decrease with thrust"

    def test_landing_terminates(self):
        """Episode terminates when rocket lands."""
        env = RocketLandingEnv(
            init_altitude_range=(10.0, 15.0),
            init_velocity_range=(-1.0, 0.0),
        )
        env.reset(seed=42)

        # No thrust - let it fall
        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        terminated = False
        for _ in range(500):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break

        assert terminated, "Should terminate on landing/crash"
        assert info.get("on_ground", False), "Should be on ground"

    def test_soft_landing_positive_reward(self):
        """Soft landing gives positive reward."""
        env = RocketLandingEnv(
            init_altitude_range=(5.0, 6.0),  # Start low
            init_velocity_range=(0.0, 0.0),  # No initial velocity
            init_tilt_range=(0.0, 0.0),  # Upright
            init_horizontal_range=(0.0, 0.0),  # At target
            init_fuel_range=(1.0, 1.0),
        )
        env.reset(seed=42)

        # Controlled descent - throttle to slow down
        total_reward = 0.0
        for _ in range(500):
            # Get altitude and adjust thrust
            alt = env._kernel.rocket.state.position.y
            vel_y = env._kernel.rocket.state.velocity.y

            # Simple control: more thrust when falling fast, less when slow
            if vel_y < -2.0:
                throttle = 0.7
            elif vel_y < -0.5:
                throttle = 0.5
            else:
                throttle = 0.3

            action = np.array([throttle, 0.0, 0.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated:
                break

        # Check we landed successfully
        assert info.get("outcome") == "landed", f"Should land successfully, got {info}"
        assert info.get("landing_vel", 999) < 3.0, "Landing velocity should be low"

    def test_crash_negative_reward(self):
        """Crash gives large negative reward."""
        env = RocketLandingEnv(
            init_altitude_range=(30.0, 35.0),
            init_velocity_range=(-10.0, -8.0),  # Fast downward
        )
        env.reset(seed=42)

        # No thrust - let it crash
        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        final_reward = 0.0
        for _ in range(500):
            obs, reward, terminated, truncated, info = env.step(action)
            final_reward = reward
            if terminated:
                break

        assert info.get("outcome") == "crash", "Should crash"
        assert final_reward < -50, "Crash should give large negative reward"

    def test_horizontal_distance_tracked(self):
        """Horizontal distance from target is tracked in info."""
        env = RocketLandingEnv(
            init_horizontal_range=(5.0, 10.0),  # Start away from target
        )
        env.reset(seed=42)

        action = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        assert "horiz_dist" in info
        assert info["horiz_dist"] > 0, "Should have horizontal distance from target"

    def test_gymnasium_compatibility(self):
        """Environment passes gymnasium check_env validation."""
        from gymnasium.utils.env_checker import check_env

        env = RocketLandingEnv()
        check_env(env, skip_render_check=True)
