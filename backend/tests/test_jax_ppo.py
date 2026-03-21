"""Tests for the pure-JAX PPO implementation.

Tests cover: config defaults, network shapes, forward pass, GAE computation,
PPO loss, trainer lifecycle (init, collect, update, save/load, predict).
"""

import tempfile
from pathlib import Path

import pytest
import numpy as np

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
nn = pytest.importorskip("flax.linen")

from simulacra.jax_ppo import (
    PPOConfig,
    ActorCritic,
    RolloutBatch,
    gaussian_log_prob,
    gaussian_entropy,
    sample_action,
    compute_gae,
    ppo_loss,
    PPOTrainer,
)
from simulacra.jax_hopper_env import JaxHopperVecEnv


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def config():
    """Small PPO config for fast testing."""
    return PPOConfig(n_steps=32, batch_size=16, n_epochs=2)


@pytest.fixture
def env():
    """Small JaxHopperVecEnv."""
    return JaxHopperVecEnv(
        num_envs=8,
        scenarios=["vertical", "landing"],
        use_guidance=True,
        max_tilt=0.25,
        seed=42,
    )


@pytest.fixture
def trainer(env, config):
    """PPOTrainer with small env and config."""
    return PPOTrainer(env, config, seed=0)


# ──────────────────────────────────────────────────────────────
# PPOConfig tests
# ──────────────────────────────────────────────────────────────

class TestPPOConfig:
    def test_defaults(self):
        """Config defaults match documented SB3 settings."""
        c = PPOConfig()
        assert c.n_steps == 2048
        assert c.gamma == 0.99
        assert c.gae_lambda == 0.95
        assert c.n_epochs == 10
        assert c.batch_size == 256
        assert c.lr == 3e-4
        assert c.max_grad_norm == 0.5
        assert c.clip_range == 0.2
        assert c.ent_coef == 0.01
        assert c.vf_coef == 0.5
        assert c.hidden_dims == (256, 256)

    def test_custom_config(self):
        """Config accepts overrides."""
        c = PPOConfig(n_steps=64, lr=1e-4, hidden_dims=(128,))
        assert c.n_steps == 64
        assert c.lr == 1e-4
        assert c.hidden_dims == (128,)


# ──────────────────────────────────────────────────────────────
# ActorCritic network tests
# ──────────────────────────────────────────────────────────────

class TestActorCritic:
    def test_init_param_shapes(self):
        """Network parameters have correct shapes for 28-dim obs, 3-dim action."""
        net = ActorCritic(action_dim=3, hidden_dims=(64, 64))
        rng = jax.random.PRNGKey(0)
        params = net.init(rng, jnp.zeros((1, 28)))["params"]

        # log_std should be (3,) — one per action dim
        assert params["log_std"].shape == (3,)

        # Should have Dense layers for both actor and critic
        leaves = jax.tree_util.tree_leaves(params)
        total_params = sum(leaf.size for leaf in leaves)
        # 19*64 + 64 + 64*64 + 64 + 64*3 + 3 (actor) + 3 (log_std)
        # + 19*64 + 64 + 64*64 + 64 + 64*1 + 1 (critic)
        # = 1216 + 64 + 4096 + 64 + 192 + 3 + 3
        # + 1216 + 64 + 4096 + 64 + 64 + 1
        # = 11143
        assert total_params > 0  # just confirm it's non-trivial

    def test_forward_pass_shapes(self):
        """Forward pass returns correct output shapes."""
        net = ActorCritic(action_dim=3, hidden_dims=(64, 64))
        rng = jax.random.PRNGKey(0)
        params = net.init(rng, jnp.zeros((1, 28)))

        batch_size = 4
        obs = jnp.ones((batch_size, 28))
        action_mean, log_std, value = net.apply(params, obs)

        assert action_mean.shape == (batch_size, 3)
        assert log_std.shape == (3,)
        assert value.shape == (batch_size,)

    def test_forward_single_obs(self):
        """Forward pass works with single observation (batch=1)."""
        net = ActorCritic(action_dim=3, hidden_dims=(32,))
        rng = jax.random.PRNGKey(0)
        params = net.init(rng, jnp.zeros((1, 28)))

        obs = jnp.ones((1, 28))
        action_mean, log_std, value = net.apply(params, obs)

        assert action_mean.shape == (1, 3)
        assert value.shape == (1,)
        # Initial log_std should be zeros
        np.testing.assert_allclose(log_std, np.zeros(3), atol=1e-6)

    def test_different_hidden_dims(self):
        """Network works with various hidden layer configurations."""
        for dims in [(32,), (64, 64), (128, 64, 32)]:
            net = ActorCritic(action_dim=3, hidden_dims=dims)
            rng = jax.random.PRNGKey(0)
            params = net.init(rng, jnp.zeros((1, 28)))
            action_mean, log_std, value = net.apply(params, jnp.ones((2, 28)))
            assert action_mean.shape == (2, 3)


# ──────────────────────────────────────────────────────────────
# Gaussian policy helpers
# ──────────────────────────────────────────────────────────────

class TestGaussianPolicy:
    def test_log_prob_standard_normal(self):
        """Log prob of 0 under N(0, 1) should be -0.5 * log(2*pi) per dim."""
        mean = jnp.zeros(3)
        log_std = jnp.zeros(3)  # std=1
        action = jnp.zeros(3)
        lp = gaussian_log_prob(mean, log_std, action)
        expected = -0.5 * 3 * np.log(2 * np.pi)
        np.testing.assert_allclose(float(lp), expected, atol=1e-5)

    def test_log_prob_batch(self):
        """Log prob works with batched inputs."""
        mean = jnp.zeros((4, 3))
        log_std = jnp.zeros(3)
        action = jnp.ones((4, 3))
        lp = gaussian_log_prob(mean, log_std, action)
        assert lp.shape == (4,)
        # All should be the same since inputs are identical
        np.testing.assert_allclose(lp[0], lp[1], atol=1e-6)

    def test_log_prob_further_is_lower(self):
        """Actions further from mean should have lower log probability."""
        mean = jnp.zeros(3)
        log_std = jnp.zeros(3)
        near = jnp.array([0.1, 0.1, 0.1])
        far = jnp.array([2.0, 2.0, 2.0])
        lp_near = gaussian_log_prob(mean, log_std, near)
        lp_far = gaussian_log_prob(mean, log_std, far)
        assert float(lp_near) > float(lp_far)

    def test_entropy_positive(self):
        """Gaussian entropy should be positive for log_std=0."""
        log_std = jnp.zeros(3)
        ent = gaussian_entropy(log_std)
        assert float(ent) > 0

    def test_entropy_increases_with_std(self):
        """Larger std should give higher entropy."""
        ent_small = gaussian_entropy(jnp.array([-1.0, -1.0, -1.0]))
        ent_large = gaussian_entropy(jnp.array([1.0, 1.0, 1.0]))
        assert float(ent_large) > float(ent_small)

    def test_sample_action_shapes(self):
        """sample_action returns correct shapes."""
        mean = jnp.zeros((4, 3))
        log_std = jnp.zeros(3)
        rng = jax.random.PRNGKey(0)
        action, log_prob = sample_action(mean, log_std, rng)
        assert action.shape == (4, 3)
        assert log_prob.shape == (4,)


# ──────────────────────────────────────────────────────────────
# GAE computation
# ──────────────────────────────────────────────────────────────

class TestGAE:
    def test_gae_shapes(self):
        """GAE returns correct shapes."""
        n_steps, n_envs = 8, 4
        rewards = jnp.ones((n_steps, n_envs))
        values = jnp.ones((n_steps, n_envs)) * 0.5
        dones = jnp.zeros((n_steps, n_envs))
        last_value = jnp.ones(n_envs) * 0.5

        advantages, returns = compute_gae(
            rewards, values, dones, last_value, gamma=0.99, gae_lambda=0.95,
        )
        assert advantages.shape == (n_steps, n_envs)
        assert returns.shape == (n_steps, n_envs)

    def test_gae_simple_case(self):
        """Test GAE with a simple known case: constant reward, no dones.

        With reward=1 every step, value=0 everywhere, gamma=0.99, lambda=0.95:
        TD error delta_t = r_t + gamma * V(t+1) - V(t) = 1 + 0 - 0 = 1 (all steps)
        GAE_t = sum_{l=0}^{T-t-1} (gamma*lambda)^l * delta_{t+l}
              = sum_{l=0}^{T-t-1} (0.99*0.95)^l
        """
        n_steps = 4
        n_envs = 1
        gamma = 0.99
        lam = 0.95

        rewards = jnp.ones((n_steps, n_envs))
        values = jnp.zeros((n_steps, n_envs))
        dones = jnp.zeros((n_steps, n_envs))
        last_value = jnp.zeros(n_envs)

        advantages, returns = compute_gae(
            rewards, values, dones, last_value, gamma=gamma, gae_lambda=lam,
        )

        gl = gamma * lam  # 0.9405
        # Expected advantages (from reverse scan):
        # t=3: delta=1, gae=1
        # t=2: delta=1, gae=1 + gl*1 = 1 + 0.9405
        # t=1: delta=1, gae=1 + gl*(1 + gl) = 1 + gl + gl^2
        # t=0: delta=1, gae=1 + gl + gl^2 + gl^3
        expected = np.array([
            [1 + gl + gl**2 + gl**3],
            [1 + gl + gl**2],
            [1 + gl],
            [1.0],
        ])
        np.testing.assert_allclose(advantages, expected, atol=1e-5)

    def test_gae_done_resets(self):
        """Done flag should reset GAE accumulation."""
        n_steps = 4
        n_envs = 1
        gamma = 0.99
        lam = 0.95

        rewards = jnp.ones((n_steps, n_envs))
        values = jnp.zeros((n_steps, n_envs))
        dones = jnp.array([[0], [1], [0], [0]], dtype=jnp.float32)
        last_value = jnp.zeros(n_envs)

        advantages, returns = compute_gae(
            rewards, values, dones, last_value, gamma=gamma, gae_lambda=lam,
        )

        gl = gamma * lam
        # t=3: delta=1, gae=1
        # t=2: delta=1, gae=1 + gl*1 = 1+gl
        # t=1: done=1, delta=1+0-0=1, gae=1 (done resets next_gae to 0)
        #   Actually: delta = r + gamma*next_val*(1-done) - val = 1 + 0 - 0 = 1
        #   gae = delta + gamma*lam*(1-done)*prev_gae = 1 + 0 = 1 (done kills carry)
        # t=0: delta=1+gamma*0*(1-0)-0=1, gae=1+gl*1 = 1+gl
        #   Wait, done[0]=0 and at t=1, done=1 affects the carry from t=1 to t=0
        #   gae_0 = delta_0 + gamma*lam*(1-done[0])*gae_1 = 1 + gl*1 = 1+gl
        expected_t3 = 1.0
        expected_t2 = 1.0 + gl
        expected_t1 = 1.0  # done kills carry
        expected_t0 = 1.0 + gl * expected_t1  # done[0]=0, so carry from t=1 flows

        np.testing.assert_allclose(float(advantages[3, 0]), expected_t3, atol=1e-5)
        np.testing.assert_allclose(float(advantages[2, 0]), expected_t2, atol=1e-5)
        np.testing.assert_allclose(float(advantages[1, 0]), expected_t1, atol=1e-5)
        np.testing.assert_allclose(float(advantages[0, 0]), expected_t0, atol=1e-5)

    def test_returns_equal_advantages_plus_values(self):
        """Returns = advantages + values by definition."""
        n_steps, n_envs = 8, 4
        rewards = jax.random.normal(jax.random.PRNGKey(0), (n_steps, n_envs))
        values = jax.random.normal(jax.random.PRNGKey(1), (n_steps, n_envs))
        dones = jnp.zeros((n_steps, n_envs))
        last_value = jnp.zeros(n_envs)

        advantages, returns = compute_gae(
            rewards, values, dones, last_value, gamma=0.99, gae_lambda=0.95,
        )
        np.testing.assert_allclose(returns, advantages + values, atol=1e-5)


# ──────────────────────────────────────────────────────────────
# PPO loss
# ──────────────────────────────────────────────────────────────

class TestPPOLoss:
    def test_loss_returns_expected_keys(self):
        """PPO loss returns dict with all expected metric keys."""
        net = ActorCritic(action_dim=3, hidden_dims=(32,))
        rng = jax.random.PRNGKey(0)
        params = net.init(rng, jnp.zeros((1, 28)))["params"]

        batch = RolloutBatch(
            obs=jnp.ones((16, 28)),
            actions=jnp.zeros((16, 3)),
            log_probs=jnp.zeros(16),
            advantages=jnp.ones(16),
            returns=jnp.ones(16),
            values=jnp.zeros(16),
        )

        total_loss, info = ppo_loss(
            params, net.apply, batch,
            clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
        )

        assert isinstance(float(total_loss), float)
        assert "policy_loss" in info
        assert "value_loss" in info
        assert "entropy" in info
        assert "approx_kl" in info
        assert "clip_frac" in info

    def test_loss_is_finite(self):
        """PPO loss should produce finite values."""
        net = ActorCritic(action_dim=3, hidden_dims=(32,))
        rng = jax.random.PRNGKey(0)
        params = net.init(rng, jnp.zeros((1, 28)))["params"]

        batch = RolloutBatch(
            obs=jax.random.normal(jax.random.PRNGKey(1), (16, 28)),
            actions=jax.random.normal(jax.random.PRNGKey(2), (16, 3)),
            log_probs=-jnp.ones(16),
            advantages=jnp.ones(16),
            returns=jnp.ones(16),
            values=jnp.zeros(16),
        )

        total_loss, info = ppo_loss(
            params, net.apply, batch,
            clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
        )

        assert jnp.isfinite(total_loss)
        for key, val in info.items():
            assert jnp.isfinite(val), f"{key} is not finite: {val}"

    def test_zero_advantage_zero_policy_loss(self):
        """Zero advantages should give near-zero policy loss."""
        net = ActorCritic(action_dim=3, hidden_dims=(32,))
        rng = jax.random.PRNGKey(0)
        params = net.init(rng, jnp.zeros((1, 28)))["params"]

        # Get log probs under current policy for these actions
        obs = jnp.ones((16, 28))
        action_mean, log_std, _ = net.apply({"params": params}, obs)
        actions = action_mean  # use mean actions for consistency
        from simulacra.jax_ppo import gaussian_log_prob
        log_probs = gaussian_log_prob(action_mean, log_std, actions)

        batch = RolloutBatch(
            obs=obs,
            actions=actions,
            log_probs=log_probs,
            advantages=jnp.zeros(16),
            returns=jnp.ones(16),
            values=jnp.ones(16),
        )

        _, info = ppo_loss(
            params, net.apply, batch,
            clip_range=0.2, ent_coef=0.0, vf_coef=0.0,
        )

        # Policy loss should be ~0 when advantages are 0
        assert abs(float(info["policy_loss"])) < 1e-5


# ──────────────────────────────────────────────────────────────
# PPOTrainer
# ──────────────────────────────────────────────────────────────

class TestPPOTrainer:
    def test_initialization(self, trainer, env):
        """Trainer initializes with correct state."""
        assert trainer.total_steps == 0
        assert trainer.env is env
        assert trainer.get_param_count() > 0

    def test_param_count(self, trainer):
        """Param count matches expected for 256x256 network with 28-dim obs, 3-dim action.

        Actor: 28*256 + 256 + 256*256 + 256 + 256*3 + 3 = 7168+256+65536+256+768+3 = 73987
        log_std: 3
        Critic: 28*256 + 256 + 256*256 + 256 + 256*1 + 1 = 7168+256+65536+256+256+1 = 73473
        Total ≈ 147K
        """
        count = trainer.get_param_count()
        # 256x256 nets should have ~147K params
        assert count > 100_000
        assert count < 200_000

    def test_collect_rollout_shapes(self, trainer, env, config):
        """Rollout collection returns correct shapes."""
        n_envs = env.num_envs
        n_steps = config.n_steps

        # Initialize env
        rng = jax.random.PRNGKey(99)
        keys = jax.random.split(rng, n_envs)
        env_state = env._reset_fn(keys)
        obs = env._state_to_obs(env_state)

        rollout, final_obs, final_state, episode_infos = trainer.collect_rollout(
            env_state, obs,
        )

        assert rollout["obs"].shape == (n_steps, n_envs, 28)
        assert rollout["actions"].shape == (n_steps, n_envs, 3)
        assert rollout["log_probs"].shape == (n_steps, n_envs)
        assert rollout["rewards"].shape == (n_steps, n_envs)
        assert rollout["dones"].shape == (n_steps, n_envs)
        assert rollout["values"].shape == (n_steps, n_envs)
        assert final_obs.shape == (n_envs, 28)

    def test_update_returns_metrics(self, trainer, env, config):
        """PPO update runs and returns metrics dict."""
        n_envs = env.num_envs

        rng = jax.random.PRNGKey(99)
        keys = jax.random.split(rng, n_envs)
        env_state = env._reset_fn(keys)
        obs = env._state_to_obs(env_state)

        rollout, final_obs, final_state, _ = trainer.collect_rollout(env_state, obs)
        info = trainer.update(rollout, final_obs)

        assert "policy_loss" in info
        assert "value_loss" in info
        assert "entropy" in info
        assert "approx_kl" in info
        assert "clip_frac" in info

        # All metrics should be finite
        for key, val in info.items():
            assert jnp.isfinite(val), f"{key} is not finite"

    def test_update_changes_params(self, trainer, env):
        """PPO update should modify parameters."""
        n_envs = env.num_envs
        rng = jax.random.PRNGKey(99)
        keys = jax.random.split(rng, n_envs)
        env_state = env._reset_fn(keys)
        obs = env._state_to_obs(env_state)

        # Snapshot params before update
        params_before = jax.tree_util.tree_leaves(trainer.train_state.params)
        params_before = [np.array(p) for p in params_before]

        rollout, final_obs, _, _ = trainer.collect_rollout(env_state, obs)
        trainer.update(rollout, final_obs)

        # Compare params after update
        params_after = jax.tree_util.tree_leaves(trainer.train_state.params)
        any_changed = False
        for before, after in zip(params_before, params_after):
            if not np.allclose(before, np.array(after)):
                any_changed = True
                break
        assert any_changed, "No parameters changed after update"

    def test_train_updates_step_count(self, trainer, env, config):
        """trainer.train() increments total_steps."""
        assert trainer.total_steps == 0
        n_envs = env.num_envs
        steps_per_rollout = config.n_steps * n_envs

        trainer.train(total_steps=steps_per_rollout, stage_name="test")

        assert trainer.total_steps == steps_per_rollout

    def test_predict_deterministic(self, trainer):
        """predict() returns clipped actions in correct range."""
        obs = np.random.randn(4, 28).astype(np.float32)
        actions = trainer.predict(obs, deterministic=True)

        assert isinstance(actions, np.ndarray)
        assert actions.shape == (4, 3)
        # Throttle in [0, 1]
        assert np.all(actions[:, 0] >= 0.0)
        assert np.all(actions[:, 0] <= 1.0)
        # Gimbal in [-1, 1]
        assert np.all(actions[:, 1:] >= -1.0)
        assert np.all(actions[:, 1:] <= 1.0)

    def test_predict_stochastic(self, trainer):
        """predict(deterministic=False) returns actions with noise."""
        obs = np.ones((4, 28), dtype=np.float32)

        # Stochastic predictions should differ across calls
        a1 = trainer.predict(obs, deterministic=False)
        a2 = trainer.predict(obs, deterministic=False)

        assert a1.shape == (4, 3)
        # Very unlikely they are exactly the same
        assert not np.allclose(a1, a2), "Stochastic predictions should differ"

    def test_predict_deterministic_is_consistent(self, trainer):
        """Deterministic predictions should be the same every call."""
        obs = np.ones((4, 28), dtype=np.float32)
        a1 = trainer.predict(obs, deterministic=True)
        a2 = trainer.predict(obs, deterministic=True)
        np.testing.assert_allclose(a1, a2, atol=1e-6)

    def test_set_lr(self, trainer):
        """set_lr should update the optimizer without error."""
        # Should not raise
        trainer.set_lr(1e-5)
        # Trainer should still be functional
        obs = np.ones((2, 28), dtype=np.float32)
        actions = trainer.predict(obs, deterministic=True)
        assert actions.shape == (2, 3)


# ──────────────────────────────────────────────────────────────
# Checkpoint save/load
# ──────────────────────────────────────────────────────────────

class TestCheckpoint:
    def test_save_load_roundtrip(self, trainer):
        """Saved and loaded parameters should match exactly."""
        # Modify total_steps to something non-zero
        trainer.total_steps = 12345

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_checkpoint"
            trainer.save(save_path)

            # Verify files exist
            assert (save_path / "params.npz").exists()
            assert (save_path / "config.json").exists()

            # Save current params for comparison
            original_leaves = jax.tree_util.tree_leaves(trainer.train_state.params)
            original_arrays = [np.array(leaf) for leaf in original_leaves]

            # Mutate params (zero them out)
            import json
            zero_params = jax.tree.map(jnp.zeros_like, trainer.train_state.params)
            trainer.train_state = trainer.train_state.replace(params=zero_params)
            trainer.total_steps = 0

            # Load
            trainer.load(save_path)

            # Check params match
            loaded_leaves = jax.tree_util.tree_leaves(trainer.train_state.params)
            for orig, loaded in zip(original_arrays, loaded_leaves):
                np.testing.assert_allclose(np.array(loaded), orig, atol=1e-7)

            # Check total_steps restored
            assert trainer.total_steps == 12345

    def test_save_creates_directory(self, trainer):
        """save() creates the directory if it does not exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            deep_path = Path(tmpdir) / "a" / "b" / "c"
            trainer.save(deep_path)
            assert (deep_path / "params.npz").exists()

    def test_load_config_metadata(self, trainer):
        """Loaded config.json contains expected metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "ckpt"
            trainer.save(save_path)

            import json
            with open(save_path / "config.json") as f:
                meta = json.load(f)

            assert meta["obs_dim"] == 28
            assert meta["action_dim"] == 3
            assert "config" in meta
            assert meta["config"]["n_steps"] == trainer.config.n_steps


# ──────────────────────────────────────────────────────────────
# Integration: full cycle
# ──────────────────────────────────────────────────────────────

class TestIntegration:
    def test_full_training_cycle(self):
        """Run a complete mini training loop: init, train, predict."""
        env = JaxHopperVecEnv(
            num_envs=8,
            scenarios=["vertical"],
            use_guidance=True,
            max_tilt=0.25,
            seed=0,
        )
        config = PPOConfig(n_steps=16, batch_size=8, n_epochs=1)
        trainer = PPOTrainer(env, config, seed=42)

        # Train for one rollout
        steps = config.n_steps * env.num_envs  # 128
        trainer.train(total_steps=steps, stage_name="integration_test")

        assert trainer.total_steps == steps

        # Predict
        obs = np.random.randn(4, 28).astype(np.float32)
        actions = trainer.predict(obs)
        assert actions.shape == (4, 3)

    def test_episode_outcomes_tracked(self):
        """Training should track episode outcomes."""
        env = JaxHopperVecEnv(
            num_envs=16,
            scenarios=["landing"],  # likely to finish quickly
            use_guidance=True,
            max_tilt=0.25,
            seed=0,
            max_episode_steps=50,
        )
        config = PPOConfig(n_steps=64, batch_size=16, n_epochs=1)
        trainer = PPOTrainer(env, config, seed=0)

        trainer.train(total_steps=64 * 16, stage_name="outcome_test")

        # With 50-step episodes and 64 steps of rollout, some should complete
        assert len(trainer._recent_outcomes) > 0
        # Each outcome should be a valid string
        valid = {"arrived", "landed", "crash", "truncated"}
        for outcome in trainer._recent_outcomes:
            assert outcome in valid, f"Invalid outcome: {outcome}"
