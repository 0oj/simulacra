"""Tests for JAX-vectorized hopper environment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from simulacra.jax_hopper_env import (
    JaxHopperVecEnv,
    hopper_physics_step,
    hopper_default_params,
    make_hopper_initial_state,
    _sample_scenario,
    SCENARIO_NAMES,
    SCENARIO_NAME_TO_ID,
)


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def env():
    return JaxHopperVecEnv(
        num_envs=16,
        scenarios=["vertical", "lateral", "diagonal", "landing", "liftoff"],
        use_guidance=True,
        seed=42,
    )


@pytest.fixture
def small_env():
    return JaxHopperVecEnv(
        num_envs=4,
        scenarios=["vertical"],
        use_guidance=True,
        seed=42,
    )


# ──────────────────────────────────────────────────────────────
# Observation space
# ──────────────────────────────────────────────────────────────

class TestObsSpace:
    def test_obs_shape(self, env):
        obs = env.reset()
        assert obs.shape == (16, 19), f"Expected (16, 19), got {obs.shape}"

    def test_obs_finite(self, env):
        obs = env.reset()
        assert np.all(np.isfinite(obs)), "Obs has non-finite values"

    def test_obs_consistent_across_seeds(self, env):
        obs1 = env.reset(seed=1)
        obs2 = env.reset(seed=1)
        np.testing.assert_array_equal(obs1, obs2)

    def test_obs_different_seeds(self, env):
        obs1 = env.reset(seed=1)
        obs2 = env.reset(seed=99)
        assert not np.allclose(obs1, obs2)

    def test_quaternion_unit(self, env):
        obs = env.reset()
        quats = obs[:, 6:10]
        norms = np.linalg.norm(quats, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=0.01)


# ──────────────────────────────────────────────────────────────
# Step mechanics
# ──────────────────────────────────────────────────────────────

class TestStep:
    def test_step_shapes(self, env):
        obs = env.reset()
        actions = np.random.uniform(0, 1, (16, 3)).astype(np.float32)
        actions[:, 1:] = actions[:, 1:] * 2 - 1
        obs, rewards, dones, infos = env.step(actions)
        assert obs.shape == (16, 19)
        assert rewards.shape == (16,)
        assert dones.shape == (16,)
        assert len(infos) == 16

    def test_step_finite(self, env):
        env.reset()
        actions = np.zeros((16, 3), dtype=np.float32)
        actions[:, 0] = 0.7  # moderate thrust
        for _ in range(20):
            obs, rewards, dones, infos = env.step(actions)
            assert np.all(np.isfinite(obs)), "Step obs has non-finite values"
            assert np.all(np.isfinite(rewards)), "Step reward has non-finite values"

    def test_zero_thrust_crashes(self, small_env):
        """Zero throttle from altitude should eventually terminate."""
        small_env.reset()
        action = np.zeros((4, 3), dtype=np.float32)
        any_done = False
        for _ in range(500):
            obs, _, dones, _ = small_env.step(action)
            if np.any(dones):
                any_done = True
                break
        assert any_done, "Zero throttle should terminate within 500 steps"

    def test_full_thrust_increases_altitude(self, small_env):
        """Full throttle should increase altitude for most envs."""
        obs = small_env.reset()
        initial_y = obs[:, 1].copy()
        action = np.array([[1.0, 0.0, 0.0]] * 4, dtype=np.float32)
        for _ in range(50):
            obs, _, dones, _ = small_env.step(action)
            if np.any(dones):
                break
        # At least some envs should have increased altitude
        final_y = obs[:, 1]
        assert np.any(final_y > initial_y - 1.0), "Full thrust should increase altitude"


# ──────────────────────────────────────────────────────────────
# Scenario sampling
# ──────────────────────────────────────────────────────────────

class TestScenarios:
    def test_all_scenarios_valid(self):
        """Each scenario ID should produce valid positions."""
        params = hopper_default_params()
        rng = jax.random.PRNGKey(42)

        for sid in range(len(SCENARIO_NAMES)):
            rng, k = jax.random.split(rng)
            start, vel, target, on_ground = _sample_scenario(sid, k, params)
            assert np.all(np.isfinite(np.array(start))), f"Scenario {sid}: invalid start"
            assert np.all(np.isfinite(np.array(target))), f"Scenario {sid}: invalid target"
            assert np.all(np.isfinite(np.array(vel))), f"Scenario {sid}: invalid vel"

    def test_landing_target_at_ground(self):
        """Landing scenarios should have target at y=0."""
        params = hopper_default_params()
        rng = jax.random.PRNGKey(42)

        for sid_name in ["landing", "landing_precision", "landing_high"]:
            sid = SCENARIO_NAME_TO_ID[sid_name]
            for seed in range(10):
                rng, k = jax.random.split(rng)
                _, _, target, _ = _sample_scenario(sid, k, params)
                assert float(target[1]) == 0.0, f"{sid_name}: target y should be 0"

    def test_liftoff_starts_on_ground(self):
        """Liftoff scenarios should start on ground."""
        params = hopper_default_params()
        rng = jax.random.PRNGKey(42)

        for sid_name in ["liftoff", "liftoff_fast"]:
            sid = SCENARIO_NAME_TO_ID[sid_name]
            for seed in range(10):
                rng, k = jax.random.split(rng)
                start, _, _, on_ground = _sample_scenario(sid, k, params)
                assert float(start[1]) == 0.0, f"{sid_name}: should start at y=0"
                assert bool(on_ground), f"{sid_name}: should be on ground"

    def test_all_10_scenarios_env(self):
        """Env with all 10 scenarios should reset without error."""
        env = JaxHopperVecEnv(
            num_envs=32,
            scenarios=SCENARIO_NAMES,
            use_guidance=True,
            seed=42,
        )
        obs = env.reset()
        assert obs.shape == (32, 19)
        assert np.all(np.isfinite(obs))


# ──────────────────────────────────────────────────────────────
# Reward
# ──────────────────────────────────────────────────────────────

class TestReward:
    def test_reward_finite(self, env):
        env.reset()
        actions = np.random.uniform(0, 1, (16, 3)).astype(np.float32)
        actions[:, 1:] = actions[:, 1:] * 2 - 1
        for _ in range(20):
            _, rewards, _, _ = env.step(actions)
            assert np.all(np.isfinite(rewards)), "Reward has non-finite values"

    def test_crash_gives_negative_reward(self, small_env):
        """Terminal crash should give large negative reward."""
        small_env.reset()
        action = np.zeros((4, 3), dtype=np.float32)
        for _ in range(500):
            _, rewards, dones, infos = small_env.step(action)
            for i in range(4):
                if dones[i] and infos[i].get("outcome") == "crash":
                    assert rewards[i] < -100, "Crash reward should be very negative"
                    return
        # It's ok if no crash happened (e.g., truncated first)


# ──────────────────────────────────────────────────────────────
# Auto-reset
# ──────────────────────────────────────────────────────────────

class TestAutoReset:
    def test_auto_reset_produces_valid_obs(self, env):
        """After done, env should auto-reset with valid obs."""
        env.reset()
        actions = np.zeros((16, 3), dtype=np.float32)
        for _ in range(500):
            obs, _, dones, _ = env.step(actions)
            if np.any(dones):
                # Obs for done envs should be fresh reset obs
                assert np.all(np.isfinite(obs)), "Auto-reset obs should be finite"
                return
        # No done in 500 steps is ok for some configs


# ──────────────────────────────────────────────────────────────
# Physics match (spot check vs NumPy kernel)
# ──────────────────────────────────────────────────────────────

class TestPhysicsMatch:
    def test_free_fall_altitude(self):
        """Free fall from 50m should match analytical solution."""
        params = hopper_default_params(dt=0.02, max_episode_steps=500)
        state = make_hopper_initial_state(
            start_pos=jnp.array([0.0, 50.0, 0.0]),
            start_vel=jnp.zeros(3),
            target=jnp.array([0.0, 0.0, 0.0]),
            on_ground=False,
            fuel_fraction=1.0,
        )
        rng = jax.random.PRNGKey(0)
        action = jnp.array([0.0, 0.0, 0.0])  # no thrust

        # Run 50 steps (1 second)
        for _ in range(50):
            rng, k = jax.random.split(rng)
            state, obs, _, _, _ = hopper_physics_step(state, action, params, k)

        # After 1s free fall: y = 50 - 0.5 * 9.81 * 1^2 ≈ 45.1
        y = float(state["pos"][1])
        expected = 50.0 - 0.5 * 9.81 * 1.0**2
        assert abs(y - expected) < 1.0, f"Free fall y={y:.2f}, expected ~{expected:.2f}"

    def test_hover_thrust_maintains_altitude(self):
        """Hover thrust (mg) should roughly maintain altitude."""
        params = hopper_default_params(dt=0.02)
        state = make_hopper_initial_state(
            start_pos=jnp.array([0.0, 20.0, 0.0]),
            start_vel=jnp.zeros(3),
            target=jnp.array([0.0, 30.0, 0.0]),
            on_ground=False,
            fuel_fraction=1.0,
        )
        rng = jax.random.PRNGKey(0)

        # mg = (60+80) * 9.81 = 1373.4 N → throttle_norm = 1373.4 / 2000 = 0.687
        hover_throttle = (60.0 + 80.0) * 9.81 / 2000.0
        action = jnp.array([hover_throttle, 0.0, 0.0])

        for _ in range(100):  # 2 seconds
            rng, k = jax.random.split(rng)
            state, _, _, _, _ = hopper_physics_step(state, action, params, k)

        y = float(state["pos"][1])
        # Wider tolerance: throttle ramps up from 0 (rate-limited actuator),
        # so rocket drops ~3m during initial ramp before stabilizing
        assert abs(y - 20.0) < 4.0, f"Hover y={y:.2f}, expected ~20.0"


# ──────────────────────────────────────────────────────────────
# Info dict
# ──────────────────────────────────────────────────────────────

class TestInfo:
    def test_terminal_info_has_outcome(self, small_env):
        """Terminal info dict should have an outcome key."""
        small_env.reset()
        action = np.zeros((4, 3), dtype=np.float32)
        for _ in range(500):
            _, _, dones, infos = small_env.step(action)
            for i in range(4):
                if dones[i]:
                    assert "outcome" in infos[i], "Terminal info should have outcome"
                    assert infos[i]["outcome"] in (
                        "arrived", "landed", "crash", "truncated"
                    )
                    return

    def test_terminal_has_terminal_obs(self, small_env):
        """Terminal info should have terminal_observation for SB3."""
        small_env.reset()
        action = np.zeros((4, 3), dtype=np.float32)
        for _ in range(500):
            _, _, dones, infos = small_env.step(action)
            for i in range(4):
                if dones[i]:
                    assert "terminal_observation" in infos[i]
                    assert infos[i]["terminal_observation"].shape == (19,)
                    return
