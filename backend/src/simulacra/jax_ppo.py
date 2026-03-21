"""
Pure-JAX PPO implementation for Simulacra.

Keeps the entire training loop on GPU: rollout collection, GAE computation,
minibatch shuffling, gradient computation, and parameter updates all happen
in JAX arrays without leaving the device.

The only CPU touches are for logging and checkpointing.

Architecture:
    ActorCritic (flax.linen)
      ├── Actor: MLP 256×256 → action_dim (mean) + action_dim (log_std param)
      └── Critic: MLP 256×256 → 1

Usage:
    from simulacra.jax_ppo import PPOTrainer, PPOConfig, ActorCritic
    from simulacra.jax_hopper_env import JaxHopperVecEnv

    env = JaxHopperVecEnv(num_envs=1024, ...)
    config = PPOConfig()
    trainer = PPOTrainer(env, config, seed=0)
    trainer.train(total_steps=10_000_000)
"""

from __future__ import annotations

import time
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import optax
from flax.training.train_state import TrainState


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

@dataclass
class PPOConfig:
    """PPO hyperparameters. Defaults match SB3 settings from train_v5_jax.py."""

    # Rollout
    n_steps: int = 2048           # steps per env per rollout
    gamma: float = 0.99           # discount factor
    gae_lambda: float = 0.95      # GAE lambda

    # Optimization
    n_epochs: int = 10            # PPO epochs per update
    batch_size: int = 256         # minibatch size
    lr: float = 3e-4              # learning rate
    max_grad_norm: float = 0.5    # gradient clipping

    # PPO losses
    clip_range: float = 0.2       # PPO clip epsilon
    ent_coef: float = 0.01        # entropy bonus coefficient
    vf_coef: float = 0.5          # value function loss coefficient

    # Network
    hidden_dims: tuple[int, ...] = (256, 256)

    # Logging
    log_interval_sec: float = 30.0  # seconds between log prints


# ──────────────────────────────────────────────────────────────
# Network: Actor-Critic MLP
# ──────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """MLP actor-critic with separate value head and Gaussian policy.

    Actor outputs action means; log_std is a learned parameter vector
    (not state-dependent), matching SB3's default behavior.
    """

    action_dim: int
    hidden_dims: tuple[int, ...] = (256, 256)

    @nn.compact
    def __call__(self, x):
        # Actor trunk
        a = x
        for dim in self.hidden_dims:
            a = nn.Dense(dim, kernel_init=nn.initializers.orthogonal(math.sqrt(2)))(a)
            a = nn.tanh(a)
        action_mean = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.orthogonal(0.01),
        )(a)

        # Learned log_std (state-independent, like SB3)
        log_std = self.param(
            "log_std",
            nn.initializers.zeros,
            (self.action_dim,),
        )

        # Critic trunk (separate network, no shared layers)
        v = x
        for dim in self.hidden_dims:
            v = nn.Dense(dim, kernel_init=nn.initializers.orthogonal(math.sqrt(2)))(v)
            v = nn.tanh(v)
        value = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(1.0),
        )(v)

        return action_mean, log_std, value.squeeze(-1)


# ──────────────────────────────────────────────────────────────
# Gaussian policy helpers
# ──────────────────────────────────────────────────────────────

def gaussian_log_prob(mean, log_std, action):
    """Log probability of action under diagonal Gaussian N(mean, exp(log_std)^2).

    Args:
        mean: (*, action_dim) action means
        log_std: (action_dim,) log standard deviations
        action: (*, action_dim) actions

    Returns:
        (*, ) log probabilities (summed over action dims)
    """
    std = jnp.exp(log_std)
    var = std ** 2
    log_prob = -0.5 * (
        ((action - mean) ** 2) / var
        + 2.0 * log_std
        + jnp.log(2.0 * jnp.pi)
    )
    return jnp.sum(log_prob, axis=-1)


def gaussian_entropy(log_std):
    """Entropy of diagonal Gaussian.

    Args:
        log_std: (action_dim,) log standard deviations

    Returns:
        scalar entropy
    """
    return 0.5 * jnp.sum(1.0 + jnp.log(2.0 * jnp.pi) + 2.0 * log_std)


def sample_action(mean, log_std, rng_key):
    """Sample from diagonal Gaussian and return (action, log_prob).

    Args:
        mean: (action_dim,) or (batch, action_dim)
        log_std: (action_dim,)
        rng_key: JAX PRNG key

    Returns:
        action, log_prob
    """
    std = jnp.exp(log_std)
    noise = jax.random.normal(rng_key, shape=mean.shape)
    action = mean + std * noise
    log_prob = gaussian_log_prob(mean, log_std, action)
    return action, log_prob


# ──────────────────────────────────────────────────────────────
# Rollout storage
# ──────────────────────────────────────────────────────────────

class RolloutBatch(NamedTuple):
    """Flattened rollout data for PPO updates."""
    obs: jnp.ndarray           # (batch, obs_dim)
    actions: jnp.ndarray       # (batch, action_dim)
    log_probs: jnp.ndarray     # (batch,)
    advantages: jnp.ndarray    # (batch,)
    returns: jnp.ndarray       # (batch,)
    values: jnp.ndarray        # (batch,)


# ──────────────────────────────────────────────────────────────
# GAE computation
# ──────────────────────────────────────────────────────────────

def compute_gae(rewards, values, dones, last_value, gamma, gae_lambda):
    """Compute Generalized Advantage Estimation.

    Uses reverse scan for efficient computation on GPU.

    Args:
        rewards: (n_steps, n_envs) rewards
        values: (n_steps, n_envs) value predictions
        dones: (n_steps, n_envs) done flags
        last_value: (n_envs,) value of last observation
        gamma: discount factor
        gae_lambda: GAE lambda

    Returns:
        advantages: (n_steps, n_envs)
        returns: (n_steps, n_envs)
    """
    n_steps = rewards.shape[0]

    def _scan_fn(gae, t):
        """Scan from last step to first."""
        idx = n_steps - 1 - t
        done = dones[idx]
        reward = rewards[idx]
        value = values[idx]

        # Next value (t+1)
        next_value = jnp.where(
            idx == n_steps - 1,
            last_value,
            values[idx + 1],
        )

        # TD error
        delta = reward + gamma * next_value * (1.0 - done) - value

        # GAE recursive formula
        gae = delta + gamma * gae_lambda * (1.0 - done) * gae
        return gae, gae

    # Scan from last timestep to first
    _, advantages_reversed = jax.lax.scan(
        _scan_fn,
        jnp.zeros_like(last_value),  # initial GAE = 0
        jnp.arange(n_steps),
    )

    # Reverse back to chronological order
    advantages = jnp.flip(advantages_reversed, axis=0)
    returns = advantages + values

    return advantages, returns


# ──────────────────────────────────────────────────────────────
# PPO loss function
# ──────────────────────────────────────────────────────────────

def ppo_loss(params, apply_fn, batch, clip_range, ent_coef, vf_coef):
    """Compute PPO clipped objective + value loss + entropy bonus.

    Args:
        params: network parameters
        apply_fn: network forward function
        batch: RolloutBatch with minibatch data
        clip_range: PPO clipping epsilon
        ent_coef: entropy coefficient
        vf_coef: value function coefficient

    Returns:
        total_loss, info_dict
    """
    action_mean, log_std, values = apply_fn({"params": params}, batch.obs)

    # New log probs under current policy
    new_log_probs = gaussian_log_prob(action_mean, log_std, batch.actions)

    # Policy ratio
    ratio = jnp.exp(new_log_probs - batch.log_probs)

    # Clipped surrogate objective
    pg_loss1 = -batch.advantages * ratio
    pg_loss2 = -batch.advantages * jnp.clip(ratio, 1.0 - clip_range, 1.0 + clip_range)
    policy_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))

    # Value loss (clipped, matching SB3)
    value_pred_clipped = batch.values + jnp.clip(
        values - batch.values, -clip_range, clip_range,
    )
    vf_loss1 = (values - batch.returns) ** 2
    vf_loss2 = (value_pred_clipped - batch.returns) ** 2
    value_loss = 0.5 * jnp.mean(jnp.maximum(vf_loss1, vf_loss2))

    # Entropy bonus
    entropy = gaussian_entropy(log_std)
    entropy_loss = -entropy

    # Total loss
    total_loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

    info = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "approx_kl": jnp.mean((ratio - 1.0) - jnp.log(ratio)),
        "clip_frac": jnp.mean(jnp.abs(ratio - 1.0) > clip_range),
    }

    return total_loss, info


# ──────────────────────────────────────────────────────────────
# PPO Trainer
# ──────────────────────────────────────────────────────────────

class PPOTrainer:
    """Pure-JAX PPO trainer for JaxHopperVecEnv.

    The entire training loop stays on GPU/TPU. Only logging and
    checkpointing touch the CPU.

    Args:
        env: JaxHopperVecEnv instance.
        config: PPOConfig with hyperparameters.
        seed: Random seed.
    """

    def __init__(self, env, config: PPOConfig, seed: int = 0):
        self.env = env
        self.config = config
        self.rng = jax.random.PRNGKey(seed)

        obs_dim = env._obs_dim  # 28 (22 base + 6 safety)
        action_dim = 3

        # Initialize network
        self.network = ActorCritic(
            action_dim=action_dim,
            hidden_dims=config.hidden_dims,
        )
        self.rng, init_key = jax.random.split(self.rng)
        dummy_obs = jnp.zeros((1, obs_dim))
        params = self.network.init(init_key, dummy_obs)["params"]

        # Optimizer: Adam with gradient clipping
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.lr),
        )

        # TrainState bundles params + optimizer state
        self.train_state = TrainState.create(
            apply_fn=self.network.apply,
            params=params,
            tx=self.optimizer,
        )

        # Pre-compile the core functions
        self._compile()

        # Tracking
        self.total_steps = 0
        self._recent_outcomes: list[str] = []

    def set_lr(self, lr: float):
        """Update learning rate by rebuilding optimizer and train state."""
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.adam(lr),
        )
        self.train_state = TrainState.create(
            apply_fn=self.network.apply,
            params=self.train_state.params,
            tx=self.optimizer,
        )
        # Recompile with new optimizer
        self._compile()

    def _compile(self):
        """JIT-compile the core training functions."""
        config = self.config

        @jax.jit
        def _get_action_and_value(params, obs, rng_key):
            """Forward pass: get action, log_prob, value for a batch of obs."""
            action_mean, log_std, value = self.network.apply(
                {"params": params}, obs,
            )
            action, log_prob = sample_action(action_mean, log_std, rng_key)
            return action, log_prob, value

        @jax.jit
        def _get_value(params, obs):
            """Get value prediction only (for bootstrapping)."""
            _, _, value = self.network.apply({"params": params}, obs)
            return value

        @jax.jit
        def _get_deterministic_action(params, obs):
            """Get deterministic action (mean) for evaluation."""
            action_mean, _, _ = self.network.apply({"params": params}, obs)
            return action_mean

        @jax.jit
        def _compute_gae_jit(rewards, values, dones, last_value):
            return compute_gae(
                rewards, values, dones, last_value,
                config.gamma, config.gae_lambda,
            )

        @jax.jit
        def _update_step(train_state, batch):
            """Single gradient update step."""
            grad_fn = jax.grad(ppo_loss, has_aux=True)
            grads, info = grad_fn(
                train_state.params,
                train_state.apply_fn,
                batch,
                config.clip_range,
                config.ent_coef,
                config.vf_coef,
            )
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, info

        # ── Full update: all epochs × minibatches in one XLA dispatch ──
        # Uses jax.lax.scan with pre-reshaped batch arrays (no dynamic_slice).
        total_samples = config.n_steps * self.env.num_envs
        n_batches = total_samples // config.batch_size
        bs = config.batch_size
        od = self.env._obs_dim
        ad = 3
        n_ep = config.n_epochs

        grad_fn = jax.grad(ppo_loss, has_aux=True)

        @jax.jit
        def _full_update(train_state, flat_obs, flat_actions, flat_log_probs,
                         flat_advantages, flat_returns, flat_values, rng_key):
            """JIT-compiled PPO update: n_epochs × n_batches in one XLA call."""

            def _batch_step(ts, batch_data):
                """Process one minibatch."""
                obs_b, act_b, lp_b, adv_b, ret_b, val_b = batch_data
                batch = RolloutBatch(
                    obs=obs_b, actions=act_b, log_probs=lp_b,
                    advantages=adv_b, returns=ret_b, values=val_b,
                )
                grads, _info = grad_fn(
                    ts.params, ts.apply_fn, batch,
                    config.clip_range, config.ent_coef, config.vf_coef,
                )
                ts = ts.apply_gradients(grads=grads)
                return ts, None

            def _epoch_step(carry, _):
                """One epoch: shuffle + scan over batches."""
                ts, rk = carry
                rk, shuffle_key = jax.random.split(rk)
                indices = jax.random.permutation(shuffle_key, total_samples)

                # Reshape shuffled data into (n_batches, batch_size, ...)
                batched = (
                    flat_obs[indices].reshape(n_batches, bs, od),
                    flat_actions[indices].reshape(n_batches, bs, ad),
                    flat_log_probs[indices].reshape(n_batches, bs),
                    flat_advantages[indices].reshape(n_batches, bs),
                    flat_returns[indices].reshape(n_batches, bs),
                    flat_values[indices].reshape(n_batches, bs),
                )

                ts, _ = jax.lax.scan(_batch_step, ts, batched)
                return (ts, rk), None

            (train_state, rng_key), _ = jax.lax.scan(
                _epoch_step, (train_state, rng_key), None, length=n_ep,
            )

            # Post-update metrics (single forward pass for logging)
            _, info = ppo_loss(
                train_state.params, train_state.apply_fn,
                RolloutBatch(
                    obs=flat_obs[:bs],
                    actions=flat_actions[:bs],
                    log_probs=flat_log_probs[:bs],
                    advantages=flat_advantages[:bs],
                    returns=flat_returns[:bs],
                    values=flat_values[:bs],
                ),
                config.clip_range, config.ent_coef, config.vf_coef,
            )

            return train_state, info, rng_key

        self._get_action_and_value = _get_action_and_value
        self._get_value = _get_value
        self._get_deterministic_action = _get_deterministic_action
        self._compute_gae = _compute_gae_jit
        self._update_step = _update_step
        self._full_update = _full_update

    def collect_rollout(self, env_state, obs):
        """Collect a rollout of n_steps from all parallel envs.

        This is a Python loop (not jax.lax.scan) because the env step
        involves auto-reset logic that mutates env._state. The per-step
        computation (action sampling + env step) is still JIT-compiled.

        Args:
            env_state: current env state dict (JAX arrays, batched)
            obs: current observations (n_envs, obs_dim) JAX array

        Returns:
            rollout_obs: (n_steps, n_envs, obs_dim)
            rollout_actions: (n_steps, n_envs, action_dim)
            rollout_log_probs: (n_steps, n_envs)
            rollout_rewards: (n_steps, n_envs)
            rollout_dones: (n_steps, n_envs)
            rollout_values: (n_steps, n_envs)
            final_obs: (n_envs, obs_dim)
            final_state: env state after rollout
            episode_infos: list of per-episode outcome dicts
        """
        n_steps = self.config.n_steps
        n_envs = self.env.num_envs

        all_obs = []
        all_actions = []
        all_log_probs = []
        all_rewards = []
        all_dones = []
        all_values = []
        episode_infos = []

        for step in range(n_steps):
            # Get action and value from policy
            self.rng, action_key = jax.random.split(self.rng)
            action, log_prob, value = self._get_action_and_value(
                self.train_state.params, obs, action_key,
            )

            # Store RAW action + log_prob for PPO (must be consistent)
            all_obs.append(obs)
            all_actions.append(action)
            all_log_probs.append(log_prob)
            all_values.append(value)

            # Clip action for env step only (PPO uses raw actions above)
            # throttle [0, 1], tilt_x [-1, 1], tilt_z [-1, 1]
            action_clipped = jnp.concatenate([
                jnp.clip(action[:, :1], 0.0, 1.0),
                jnp.clip(action[:, 1:], -1.0, 1.0),
            ], axis=-1)

            # Env step (uses JIT-compiled vmapped physics)
            self.rng, step_key = jax.random.split(self.rng)
            step_keys = jax.random.split(step_key, n_envs)
            new_state, new_obs, rewards, dones, info = self.env._step_fn(
                env_state, action_clipped, step_keys,
            )

            # Auto-reset done envs
            self.rng, reset_key = jax.random.split(self.rng)
            reset_keys = jax.random.split(reset_key, n_envs)
            fresh_state = self.env._reset_fn(reset_keys)

            # Merge: done envs get fresh state
            env_state = jax.tree.map(
                lambda fresh, cont: jnp.where(
                    dones[:, None] if fresh.ndim > 1 else dones,
                    fresh, cont,
                ),
                fresh_state, new_state,
            )

            # For done envs, the obs should be the RESET obs (for next step)
            # But we record the pre-reset obs for GAE computation
            reset_obs = self.env._state_to_obs(fresh_state)
            obs = jnp.where(dones[:, None], reset_obs, new_obs)

            all_rewards.append(rewards)
            all_dones.append(dones)

            # Track episode outcomes (move to CPU lazily for logging)
            dones_np = np.asarray(dones)
            if np.any(dones_np):
                arrived_np = np.asarray(info["arrived"])
                landed_np = np.asarray(info["landed"])
                crashed_np = np.asarray(info["crashed"])
                for i in np.where(dones_np)[0]:
                    if arrived_np[i]:
                        episode_infos.append("arrived")
                    elif landed_np[i]:
                        episode_infos.append("landed")
                    elif crashed_np[i]:
                        episode_infos.append("crash")
                    else:
                        episode_infos.append("truncated")

        # Stack into (n_steps, n_envs, ...)
        rollout = {
            "obs": jnp.stack(all_obs),
            "actions": jnp.stack(all_actions),
            "log_probs": jnp.stack(all_log_probs),
            "rewards": jnp.stack(all_rewards),
            "dones": jnp.stack(all_dones),
            "values": jnp.stack(all_values),
        }

        return rollout, obs, env_state, episode_infos

    def update(self, rollout, last_obs):
        """Run PPO update on collected rollout.

        Args:
            rollout: dict with obs, actions, log_probs, rewards, dones, values
                     each shaped (n_steps, n_envs, ...)
            last_obs: (n_envs, obs_dim) final observation for bootstrapping

        Returns:
            info: dict with loss metrics
        """
        config = self.config
        n_steps = config.n_steps
        n_envs = self.env.num_envs

        # Bootstrap value for last observation
        last_value = self._get_value(self.train_state.params, last_obs)

        # Compute GAE
        advantages, returns = self._compute_gae(
            rollout["rewards"],
            rollout["values"],
            rollout["dones"].astype(jnp.float32),
            last_value,
        )

        # Normalize advantages
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

        # Flatten (n_steps, n_envs, ...) -> (n_steps * n_envs, ...)
        total_samples = n_steps * n_envs
        flat_obs = rollout["obs"].reshape(total_samples, -1)
        flat_actions = rollout["actions"].reshape(total_samples, -1)
        flat_log_probs = rollout["log_probs"].reshape(total_samples)
        flat_advantages = advantages.reshape(total_samples)
        flat_returns = returns.reshape(total_samples)
        flat_values = rollout["values"].reshape(total_samples)

        # All epochs × minibatches in one JIT call
        self.rng, update_key = jax.random.split(self.rng)
        self.train_state, info, _ = self._full_update(
            self.train_state,
            flat_obs, flat_actions, flat_log_probs,
            flat_advantages, flat_returns, flat_values,
            update_key,
        )
        return info

    def train(
        self,
        total_steps: int,
        save_dir: Optional[Path] = None,
        save_freq: int = 10_000_000,
        eval_fn: Optional[Any] = None,
        eval_freq: int = 10_000_000,
        log_entries: Optional[list] = None,
        stage_name: str = "",
        shutdown_flag: Optional[list] = None,
        lr_start: Optional[float] = None,
        lr_end: Optional[float] = None,
    ):
        """Main training loop.

        Args:
            total_steps: total environment steps to train for.
            save_dir: directory for checkpoints (optional).
            save_freq: steps between checkpoint saves.
            eval_fn: callable(trainer) -> None for periodic evaluation.
            eval_freq: steps between evaluations.
            log_entries: list to append log dicts to (for persistence).
            stage_name: name for logging.
            shutdown_flag: list[bool] — set [True] to request graceful stop.
            lr_start: initial learning rate for cosine schedule (None = constant).
            lr_end: final learning rate for cosine schedule.
        """
        config = self.config
        n_envs = self.env.num_envs
        steps_per_rollout = config.n_steps * n_envs

        if shutdown_flag is None:
            shutdown_flag = [False]
        if log_entries is None:
            log_entries = []

        # LR schedule — rebuild optimizer with cosine decay
        if lr_start is not None and lr_end is not None:
            n_updates = max(1, total_steps // steps_per_rollout)
            # Each PPO update does n_epochs passes, but optax counts apply_gradients calls.
            # PPO does n_epochs * n_minibatches gradient steps per update.
            n_minibatches = max(1, (config.n_steps * n_envs) // config.batch_size)
            total_grad_steps = n_updates * config.n_epochs * n_minibatches
            lr_schedule = optax.cosine_decay_schedule(
                init_value=lr_start,
                decay_steps=total_grad_steps,
                alpha=lr_end / lr_start,  # alpha = end/start ratio
            )
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adam(lr_schedule),
            )
            self.train_state = TrainState.create(
                apply_fn=self.network.apply,
                params=self.train_state.params,
                tx=self.optimizer,
            )
            self._compile()
            print(f"  LR schedule: {lr_start} → {lr_end} (cosine over {total_grad_steps:,} grad steps)")

        # Initialize env
        self.rng, reset_key = jax.random.split(self.rng)
        reset_keys = jax.random.split(reset_key, n_envs)
        env_state = self.env._reset_fn(reset_keys)
        obs = self.env._state_to_obs(env_state)

        train_start = time.time()
        last_log_time = time.time()
        last_save_steps = 0
        last_eval_steps = 0
        step_count_this_stage = 0
        all_outcomes: list[str] = []

        n_updates = max(1, total_steps // steps_per_rollout)

        for update_idx in range(n_updates):
            if shutdown_flag[0]:
                if save_dir is not None:
                    self.save(save_dir / "checkpoints" / "interrupted")
                print("  [!] Shutdown, saving and exiting.")
                break

            # ── Collect rollout ──
            rollout, obs, env_state, episode_infos = self.collect_rollout(
                env_state, obs,
            )
            all_outcomes.extend(episode_infos)
            self._recent_outcomes.extend(episode_infos)

            # ── PPO update ──
            update_info = self.update(rollout, obs)

            step_count_this_stage += steps_per_rollout
            self.total_steps += steps_per_rollout

            # ── Logging ──
            now = time.time()
            if now - last_log_time >= config.log_interval_sec:
                last_log_time = now
                elapsed_min = (now - train_start) / 60.0
                total_elapsed_sec = now - train_start
                sps = step_count_this_stage / max(1.0, total_elapsed_sec)

                # Outcome rates from recent episodes
                recent = self._recent_outcomes[-2000:]
                n_recent = len(recent)
                if n_recent > 0:
                    ok_rate = sum(1 for o in recent if o in ("arrived", "landed")) / n_recent
                    crash_rate = sum(1 for o in recent if o == "crash") / n_recent
                else:
                    ok_rate = 0.0
                    crash_rate = 0.0

                entry = {
                    "t_min": round(elapsed_min, 1),
                    "steps": self.total_steps,
                    "stage": stage_name,
                    "ok_rate": round(ok_rate, 3),
                    "crash_rate": round(crash_rate, 3),
                    "sps": round(sps),
                    "policy_loss": float(update_info["policy_loss"]),
                    "value_loss": float(update_info["value_loss"]),
                    "entropy": float(update_info["entropy"]),
                    "approx_kl": float(update_info["approx_kl"]),
                }
                log_entries.append(entry)

                print(
                    f"  [{elapsed_min:.0f}m] {self.total_steps:,} steps"
                    f" | ok={100*ok_rate:.0f}%"
                    f" | crash={100*crash_rate:.0f}%"
                    f" | {sps:.0f} sps"
                )

            # ── Checkpointing ──
            if save_dir is not None and (self.total_steps - last_save_steps) >= save_freq:
                last_save_steps = self.total_steps
                self.save(save_dir / "checkpoints" / f"step_{self.total_steps}")

            # ── Evaluation ──
            if eval_fn is not None and (self.total_steps - last_eval_steps) >= eval_freq:
                last_eval_steps = self.total_steps
                eval_fn(self)

        # Final timing report
        total_elapsed = time.time() - train_start
        final_sps = step_count_this_stage / max(1.0, total_elapsed)
        print(f"  Stage done: {step_count_this_stage:,} steps in "
              f"{total_elapsed / 60:.1f} min ({final_sps:.0f} sps)")

        return env_state, obs

    def predict(self, obs, deterministic: bool = True):
        """Get actions for observations (for evaluation).

        Args:
            obs: (n, obs_dim) numpy or jax array

        Returns:
            actions: (n, action_dim) numpy array
        """
        obs_jax = jnp.array(obs)
        if deterministic:
            actions = self._get_deterministic_action(
                self.train_state.params, obs_jax,
            )
        else:
            self.rng, key = jax.random.split(self.rng)
            actions, _ = self._get_action_and_value(
                self.train_state.params, obs_jax, key,
            )[:2]  # drop value

        # Clip to action bounds
        actions = jnp.concatenate([
            jnp.clip(actions[:, :1], 0.0, 1.0),
            jnp.clip(actions[:, 1:], -1.0, 1.0),
        ], axis=-1)

        return np.asarray(actions)

    def save(self, path: Path):
        """Save model parameters and config.

        Saves:
            {path}/params.npz — flattened parameter arrays
            {path}/config.json — PPO config + metadata
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Flatten params to numpy arrays
        flat_params = jax.tree_util.tree_leaves_with_path(self.train_state.params)
        param_dict = {}
        for key_path, leaf in flat_params:
            key_str = "/".join(str(k) for k in key_path)
            param_dict[key_str] = np.asarray(leaf)
        np.savez(path / "params.npz", **param_dict)

        # Save tree structure for reconstruction
        tree_def = jax.tree_util.tree_structure(self.train_state.params)

        # Config and metadata
        meta = {
            "total_steps": self.total_steps,
            "config": {
                "n_steps": self.config.n_steps,
                "gamma": self.config.gamma,
                "gae_lambda": self.config.gae_lambda,
                "n_epochs": self.config.n_epochs,
                "batch_size": self.config.batch_size,
                "lr": self.config.lr,
                "max_grad_norm": self.config.max_grad_norm,
                "clip_range": self.config.clip_range,
                "ent_coef": self.config.ent_coef,
                "vf_coef": self.config.vf_coef,
                "hidden_dims": list(self.config.hidden_dims),
            },
            "obs_dim": self.env._obs_dim,
            "action_dim": 3,
        }
        with open(path / "config.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  Saved checkpoint: {path}")

    def load(self, path: Path):
        """Load model parameters from a checkpoint.

        Args:
            path: directory containing params.npz and config.json
        """
        path = Path(path)
        data = np.load(path / "params.npz")

        # Get the parameter tree structure from current params
        tree_structure = jax.tree_util.tree_structure(self.train_state.params)
        flat_leaves = jax.tree_util.tree_leaves(self.train_state.params)

        # Load leaves in same order (sorted by key)
        loaded_arrays = {k: jnp.array(data[k]) for k in sorted(data.files)}
        loaded_leaves = list(loaded_arrays.values())

        if len(loaded_leaves) != len(flat_leaves):
            raise ValueError(
                f"Checkpoint has {len(loaded_leaves)} params, "
                f"model expects {len(flat_leaves)}"
            )

        # Reconstruct parameter tree
        new_params = jax.tree_util.tree_unflatten(tree_structure, loaded_leaves)

        # Update train state with loaded params
        self.train_state = self.train_state.replace(params=new_params)

        # Load metadata
        meta_path = path / "config.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self.total_steps = meta.get("total_steps", 0)
            print(f"  Loaded checkpoint from {path} ({self.total_steps:,} steps)")
        else:
            print(f"  Loaded checkpoint from {path}")

        # Recompile with updated params in train_state
        self._compile()

    def load_sb3_weights(self, sb3_path: Path):
        """Load weights from an SB3 PPO model checkpoint (.zip).

        Maps SB3's MlpPolicy parameters to our flax ActorCritic:
          SB3 actor:  mlp_extractor.policy_net.{0,2} + action_net + log_std
          SB3 critic: mlp_extractor.value_net.{0,2} + value_net
          Flax:       Dense_0-2 (actor) + Dense_3-5 (critic) + log_std

        Note: PyTorch Linear stores weight as (out, in); flax Dense
        stores kernel as (in, out). Weights are transposed automatically.
        """
        import zipfile
        import io

        try:
            import torch
        except ImportError:
            raise ImportError("torch is required to load SB3 weights")

        sb3_path = Path(sb3_path)

        with zipfile.ZipFile(sb3_path, "r") as zf:
            with zf.open("policy.pth") as f:
                sd = torch.load(
                    io.BytesIO(f.read()),
                    map_location="cpu",
                    weights_only=True,
                )

        def to_jnp(tensor):
            return jnp.array(tensor.numpy())

        # SB3 → flax mapping (transpose weights: PyTorch (out,in) → flax (in,out))
        new_params = {
            # Actor hidden layers
            "Dense_0": {
                "kernel": to_jnp(sd["mlp_extractor.policy_net.0.weight"].T),
                "bias": to_jnp(sd["mlp_extractor.policy_net.0.bias"]),
            },
            "Dense_1": {
                "kernel": to_jnp(sd["mlp_extractor.policy_net.2.weight"].T),
                "bias": to_jnp(sd["mlp_extractor.policy_net.2.bias"]),
            },
            # Actor output
            "Dense_2": {
                "kernel": to_jnp(sd["action_net.weight"].T),
                "bias": to_jnp(sd["action_net.bias"]),
            },
            # Learned log_std
            "log_std": to_jnp(sd["log_std"]),
            # Critic hidden layers
            "Dense_3": {
                "kernel": to_jnp(sd["mlp_extractor.value_net.0.weight"].T),
                "bias": to_jnp(sd["mlp_extractor.value_net.0.bias"]),
            },
            "Dense_4": {
                "kernel": to_jnp(sd["mlp_extractor.value_net.2.weight"].T),
                "bias": to_jnp(sd["mlp_extractor.value_net.2.bias"]),
            },
            # Critic output
            "Dense_5": {
                "kernel": to_jnp(sd["value_net.weight"].T),
                "bias": to_jnp(sd["value_net.bias"]),
            },
        }

        # Verify shapes match
        old_params = self.train_state.params
        for key in new_params:
            if isinstance(new_params[key], dict):
                for subkey in new_params[key]:
                    expected = old_params[key][subkey].shape
                    got = new_params[key][subkey].shape
                    if expected != got:
                        raise ValueError(
                            f"Shape mismatch {key}/{subkey}: "
                            f"expected {expected}, got {got}"
                        )
            else:
                expected = old_params[key].shape
                got = new_params[key].shape
                if expected != got:
                    raise ValueError(
                        f"Shape mismatch {key}: expected {expected}, got {got}"
                    )

        self.train_state = self.train_state.replace(params=new_params)
        self._compile()
        print(f"  Loaded SB3 weights from {sb3_path}")

    def get_param_count(self):
        """Count total trainable parameters."""
        leaves = jax.tree_util.tree_leaves(self.train_state.params)
        return sum(leaf.size for leaf in leaves)


# ──────────────────────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────────────────────

def _smoke_test():
    """Verify the PPO trainer initializes and runs one update."""
    from simulacra.jax_hopper_env import JaxHopperVecEnv

    print("PPO smoke test...")
    env = JaxHopperVecEnv(
        num_envs=64,
        scenarios=["vertical", "landing"],
        use_guidance=True,
        max_tilt=0.25,
        seed=42,
    )
    config = PPOConfig(n_steps=64, batch_size=32, n_epochs=2)
    trainer = PPOTrainer(env, config, seed=0)

    print(f"  Network params: {trainer.get_param_count():,}")
    print(f"  Obs dim: {env._obs_dim}, Action dim: 3")

    # Run one training step
    t0 = time.time()
    trainer.train(
        total_steps=config.n_steps * env.num_envs,
        stage_name="smoke_test",
    )
    elapsed = time.time() - t0
    sps = config.n_steps * env.num_envs / elapsed
    print(f"  One update: {elapsed:.2f}s ({sps:.0f} sps)")
    print("  PASS")


if __name__ == "__main__":
    _smoke_test()
