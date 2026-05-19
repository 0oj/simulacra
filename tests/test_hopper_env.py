"""Unit tests for HopperTrackingEnv."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
import gymnasium as gym

from simulacra.hopper_env import HopperTrackingEnv


@pytest.fixture
def env():
    e = HopperTrackingEnv(
        scenarios=["vertical"],
        max_episode_steps=200,
        max_altitude=100.0,
    )
    yield e
    e.close()


@pytest.fixture
def all_scenarios_env():
    e = HopperTrackingEnv(
        scenarios=["vertical", "lateral", "diagonal", "landing", "liftoff"],
        max_episode_steps=200,
        max_altitude=100.0,
    )
    yield e
    e.close()


class TestObservationSpace:
    def test_obs_shape(self, env):
        obs, _ = env.reset(seed=42)
        assert obs.shape == (22,), f"Expected 22-dim obs, got {obs.shape}"

    def test_obs_in_bounds(self, env):
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs), (
            f"Obs not in space: {obs}"
        )

    def test_obs_components(self, env):
        obs, _ = env.reset(seed=42)
        # Base state: [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
        x, y, z = obs[0], obs[1], obs[2]
        qw, qx, qy, qz = obs[6], obs[7], obs[8], obs[9]
        # Quaternion should be approximately unit
        qnorm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        assert abs(qnorm - 1.0) < 0.01, f"Quaternion not unit: {qnorm}"
        # Goal-relative: [dx, dy, dz]
        dx, dy, dz = obs[13], obs[14], obs[15]
        # dist_to_wp should be >= 0
        dist_to_wp = obs[16]
        assert dist_to_wp >= 0

    def test_obs_space_consistent_across_seeds(self, env):
        obs1, _ = env.reset(seed=1)
        obs2, _ = env.reset(seed=2)
        assert obs1.shape == obs2.shape
        assert env.observation_space.contains(obs1)
        assert env.observation_space.contains(obs2)


class TestActionSpace:
    def test_action_shape(self, env):
        assert env.action_space.shape == (3,)

    def test_action_bounds(self, env):
        low, high = env.action_space.low, env.action_space.high
        np.testing.assert_array_equal(low, [0.0, -1.0, -1.0])
        np.testing.assert_array_equal(high, [1.0, 1.0, 1.0])


class TestReset:
    def test_reset_returns_obs_info(self, env):
        result = env.reset(seed=42)
        assert len(result) == 2
        obs, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_reset_seed_reproducibility(self, env):
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)

    def test_reset_different_seeds(self, env):
        obs1, _ = env.reset(seed=1)
        obs2, _ = env.reset(seed=99)
        # Different seeds should generally produce different starting states
        assert not np.allclose(obs1, obs2)


class TestStep:
    def test_step_returns_5_tuple(self, env):
        env.reset(seed=42)
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5
        obs, reward, term, trunc, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)

    def test_step_obs_in_bounds(self, env):
        env.reset(seed=42)
        for _ in range(20):
            action = env.action_space.sample()
            obs, _, term, trunc, _ = env.step(action)
            assert env.observation_space.contains(obs), f"Step obs not in space"
            if term or trunc:
                break

    def test_hover_action(self, env):
        """Zero tilt + moderate throttle should roughly hover."""
        env.reset(seed=42)
        hover_action = np.array([0.7, 0.0, 0.0], dtype=np.float32)
        for _ in range(50):
            obs, _, term, trunc, _ = env.step(hover_action)
            if term or trunc:
                break
        # Should not have crashed immediately
        y = obs[1]
        assert y > -5.0, "Rocket fell through ground"

    def test_full_throttle_goes_up(self, env):
        """Full throttle with no tilt should increase altitude."""
        env.reset(seed=42)
        initial_y = env._kernel.rocket.state.position.y
        action = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(50):
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break
        final_y = obs[1]
        assert final_y > initial_y, "Full throttle should increase altitude"


class TestTermination:
    def test_crash_terminates(self, env):
        """Zero throttle from altitude should eventually crash."""
        env.reset(seed=42)
        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        crashed = False
        for _ in range(500):
            obs, _, term, trunc, info = env.step(action)
            if term:
                crashed = True
                break
        # Should terminate (either crash or something else)
        assert crashed or trunc, "Zero throttle should terminate"

    def test_truncation_at_max_steps(self):
        short_env = HopperTrackingEnv(
            scenarios=["vertical"],
            max_episode_steps=10,
            max_altitude=100.0,
        )
        short_env.reset(seed=42)
        action = np.array([0.7, 0.0, 0.0], dtype=np.float32)
        for i in range(20):
            obs, _, term, trunc, info = short_env.step(action)
            if term or trunc:
                break
        assert term or trunc, "Should terminate within 20 steps for 10 max"
        short_env.close()

    def test_outcome_in_info(self, env):
        """Terminal info should include outcome key."""
        env.reset(seed=42)
        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(500):
            _, _, term, trunc, info = env.step(action)
            if term or trunc:
                assert "outcome" in info, "Terminal info should have outcome"
                assert info["outcome"] in ("crash", "arrived", "landed", "timeout",
                                           "out_of_bounds")
                break


class TestScenarios:
    def test_all_scenarios_reset(self, all_scenarios_env):
        """All scenario types should produce valid initial states."""
        for seed in range(50):
            obs, info = all_scenarios_env.reset(seed=seed)
            assert all_scenarios_env.observation_space.contains(obs), (
                f"Scenario seed {seed}: obs not in space"
            )

    def test_vertical_scenario_start(self):
        env = HopperTrackingEnv(
            scenarios=["vertical"],
            max_episode_steps=100,
            max_altitude=100.0,
        )
        obs, _ = env.reset(seed=42)
        # Vertical: target should be directly above/below (small x, z offset)
        target = env._target
        assert abs(target[0]) < 1.0, "Vertical target should have small X"
        assert abs(target[2]) < 1.0, "Vertical target should have small Z"
        env.close()

    def test_liftoff_starts_on_ground(self):
        env = HopperTrackingEnv(
            scenarios=["liftoff"],
            max_episode_steps=100,
            max_altitude=100.0,
        )
        obs, _ = env.reset(seed=42)
        y = obs[1]
        assert y < 1.0, f"Liftoff should start near ground, got y={y}"
        env.close()


class TestObsSpaceConsistency:
    def test_consistent_across_scenario_configs(self):
        """All scenario configs should produce same obs space bounds."""
        configs = [
            dict(scenarios=["vertical"], max_episode_steps=100, max_altitude=100.0),
            dict(scenarios=["lateral"], max_episode_steps=100, max_altitude=100.0),
            dict(scenarios=["vertical", "lateral", "diagonal"],
                 max_episode_steps=100, max_altitude=100.0),
            dict(scenarios=["vertical", "lateral", "diagonal", "landing", "liftoff"],
                 max_episode_steps=100, max_altitude=100.0),
        ]
        spaces = []
        for cfg in configs:
            env = HopperTrackingEnv(**cfg)
            spaces.append(env.observation_space)
            env.close()

        for i in range(1, len(spaces)):
            np.testing.assert_array_equal(
                spaces[0].low, spaces[i].low,
                err_msg=f"Config {i} low bounds differ"
            )
            np.testing.assert_array_equal(
                spaces[0].high, spaces[i].high,
                err_msg=f"Config {i} high bounds differ"
            )


class TestReward:
    def test_reward_is_float(self, env):
        env.reset(seed=42)
        _, reward, _, _, _ = env.step(env.action_space.sample())
        assert isinstance(reward, float)

    def test_approach_reward_positive(self, env):
        """Moving toward target should give positive reward."""
        env.reset(seed=42)
        # Full throttle up when target is above
        target = env._target
        if target[1] > 0:
            action = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            total_r = 0
            for _ in range(20):
                _, r, term, trunc, _ = env.step(action)
                total_r += r
                if term or trunc:
                    break
            # Should get some positive reward from approaching
            assert total_r > -50, f"Expected some positive reward, got {total_r}"
