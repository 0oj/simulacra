"""
Tests for the Gymnasium environment wrapper.
"""

import numpy as np
import pytest

from simulacra.gym_env import RocketHoverEnv


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
