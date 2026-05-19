"""
JAX-compatible RAJS (Random Annealing Jump Start) for PPO training.

Uses a simple PD guide policy to bootstrap exploration in early training.
The guide flies the first H steps of each episode (H ~ U(0, H_max * beta)),
then hands off to the RL policy. Beta anneals from 1 to 0 as success rate
crosses a threshold.

Architecture: RAJS lives in the trainer, not the env. The guide policy is a
pure JAX function that reads from the 28-dim observation vector. Action
substitution happens in collect_rollout — the rollout buffer always stores
the RL policy's (action, log_prob), maintaining PPO's self-consistency.

Based on: "Rocket Landing Control with Random Annealing Jump Start
Reinforcement Learning" (LandSpace, 2024)

Usage:
    from simulacra.rajs_jax import RAJSPPOTrainer, RAJSConfig
    from simulacra.jax_ppo import PPOConfig

    trainer = RAJSPPOTrainer(env, ppo_config, rajs_config, seed=0)
    trainer.train(total_steps=500_000_000, ...)
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .jax_ppo import PPOTrainer, PPOConfig


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

@dataclass
class RAJSConfig:
    """RAJS hyperparameters."""

    h_max: int = 200              # Max guide horizon in env steps (200 * 0.02s = 4s)
    anneal_threshold: float = 0.3  # Success rate to trigger annealing
    anneal_step: float = 1 / 1500  # Beta reduction per trigger
    window_size: int = 2000        # Rolling window for success rate

    # Guide PD gains
    guide_g_comp: float = 0.69     # Gravity compensation (~mg / max_thrust)
    guide_kp_alt: float = 0.03     # Altitude proportional gain
    guide_kd_alt: float = 0.08     # Altitude derivative (velocity) gain
    guide_kp_lat: float = 0.05     # Lateral proportional gain
    guide_kd_lat: float = 0.15     # Lateral derivative gain
    guide_liftoff_boost: float = 0.15  # Extra throttle for liftoff mode


# ──────────────────────────────────────────────────────────────
# Pure JAX guide policy
# ──────────────────────────────────────────────────────────────

def make_guide_policy(config: RAJSConfig):
    """Create a JIT-compiled, batched guide policy from config.

    The guide is a simple PD controller operating on the 28-dim obs:
      obs[1] = altitude (y)
      obs[3:6] = velocity (vx, vy, vz)
      obs[13:16] = goal_rel (dx, dy, dz)
      obs[21] = is_liftoff mode flag

    Returns actions in [throttle, tilt_x, tilt_z] matching the RL action space.
    """
    g_comp = config.guide_g_comp
    kp_alt = config.guide_kp_alt
    kd_alt = config.guide_kd_alt
    kp_lat = config.guide_kp_lat
    kd_lat = config.guide_kd_lat
    liftoff_boost = config.guide_liftoff_boost

    def _guide_single(obs):
        """Guide policy for a single env."""
        # Goal-relative position
        dx = obs[13]
        dy = obs[14]
        dz = obs[15]

        # Current velocity
        vx = obs[3]
        vy = obs[4]
        vz = obs[5]

        # Flight mode
        is_liftoff = obs[21]

        # Vertical: gravity compensation + P altitude + D velocity
        throttle = g_comp + kp_alt * dy - kd_alt * vy + liftoff_boost * is_liftoff
        throttle = jnp.clip(throttle, 0.05, 0.95)

        # Lateral: tilt proportional to error, damped by velocity
        tilt_x = kp_lat * dx - kd_lat * vx
        tilt_z = kp_lat * dz - kd_lat * vz
        tilt_x = jnp.clip(tilt_x, -0.8, 0.8)
        tilt_z = jnp.clip(tilt_z, -0.8, 0.8)

        return jnp.array([throttle, tilt_x, tilt_z])

    # Batched + JIT
    return jax.jit(jax.vmap(_guide_single))


# ──────────────────────────────────────────────────────────────
# RAJS PPO Trainer
# ──────────────────────────────────────────────────────────────

class RAJSPPOTrainer(PPOTrainer):
    """PPO trainer with RAJS guide policy for bootstrapped exploration.

    Overrides collect_rollout to substitute guide actions for the first
    H steps of each episode. The rollout buffer stores the RL policy's
    actions (not the guide's), maintaining PPO correctness.

    Args:
        env: JaxHopperVecEnv instance
        config: PPOConfig with PPO hyperparameters
        rajs_config: RAJSConfig with RAJS hyperparameters
        seed: Random seed
    """

    def __init__(self, env, config: PPOConfig, rajs_config: RAJSConfig, seed: int = 0):
        super().__init__(env, config, seed)

        self.rajs_config = rajs_config
        self.beta = 1.0

        # Per-env guide horizons (resampled on episode reset)
        n_envs = env.num_envs
        self._guide_horizons = jnp.zeros(n_envs, dtype=jnp.int32)

        # JIT-compiled batched guide policy
        self._guide_fn = make_guide_policy(rajs_config)

        # RAJS tracking
        self._rajs_guide_steps = 0
        self._rajs_total_steps = 0

    def _extra_log_info(self) -> tuple[dict, str]:
        """Add RAJS metrics to log entries and print output."""
        fields = {
            "rajs_beta": round(self.beta, 4),
            "rajs_guide_frac": round(self.guide_fraction, 4),
            "rajs_h_eff": round(self.effective_h_max, 1),
        }
        msg = f" | β={self.beta:.3f}"
        if self.beta > 0:
            msg += f" h={self.effective_h_max:.0f}"
        return fields, msg

    def collect_rollout(self, env_state, obs):
        """Collect rollout with RAJS guide action substitution.

        During guide phase (step_count < guide_horizon for each env),
        the guide's action is sent to the env, but the RL policy's
        action/log_prob/value are stored in the buffer.
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

        guide_steps_this_rollout = 0

        for step in range(n_steps):
            # Get action and value from RL policy
            self.rng, action_key = jax.random.split(self.rng)
            action, log_prob, value = self._get_action_and_value(
                self.train_state.params, obs, action_key,
            )

            # Store RL policy's outputs (always — PPO consistency)
            all_obs.append(obs)
            all_actions.append(action)
            all_log_probs.append(log_prob)
            all_values.append(value)

            # Clip RL action for env step
            action_clipped = jnp.concatenate([
                jnp.clip(action[:, :1], 0.0, 1.0),
                jnp.clip(action[:, 1:], -1.0, 1.0),
            ], axis=-1)

            # ── RAJS action substitution ──
            if self.beta > 0:
                guide_action = self._guide_fn(obs)
                # Clip guide action to same bounds
                guide_clipped = jnp.concatenate([
                    jnp.clip(guide_action[:, :1], 0.0, 1.0),
                    jnp.clip(guide_action[:, 1:], -1.0, 1.0),
                ], axis=-1)

                # Use guide if within guide horizon
                use_guide = env_state["step_count"] < self._guide_horizons  # (n_envs,)
                action_for_env = jnp.where(use_guide[:, None], guide_clipped, action_clipped)

                guide_steps_this_rollout += int(jnp.sum(use_guide))
            else:
                action_for_env = action_clipped

            # Env step
            self.rng, step_key = jax.random.split(self.rng)
            step_keys = jax.random.split(step_key, n_envs)
            new_state, new_obs, rewards, dones, info = self.env._step_fn(
                env_state, action_for_env, step_keys,
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

            # ── RAJS horizon resampling on reset ──
            if self.beta > 0:
                self.rng, rajs_key = jax.random.split(self.rng)
                raw = jax.random.uniform(rajs_key, shape=(n_envs,))
                new_horizons = jnp.int32(raw * self.rajs_config.h_max * self.beta)
                self._guide_horizons = jnp.where(dones, new_horizons, self._guide_horizons)

            # Obs: done envs get reset obs
            reset_obs = self.env._state_to_obs(fresh_state)
            obs = jnp.where(dones[:, None], reset_obs, new_obs)

            all_rewards.append(rewards)
            all_dones.append(dones)

            # Track episode outcomes
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

        # ── RAJS beta annealing (once per rollout) ──
        if self.beta > 0:
            recent = self._recent_outcomes[-self.rajs_config.window_size:]
            if len(recent) >= 50:
                success_rate = sum(1 for o in recent if o in ("arrived", "landed")) / len(recent)
                if success_rate >= self.rajs_config.anneal_threshold:
                    self.beta = max(0.0, self.beta - self.rajs_config.anneal_step)

        # Update RAJS tracking
        total_this_rollout = n_steps * n_envs
        self._rajs_guide_steps += guide_steps_this_rollout
        self._rajs_total_steps += total_this_rollout

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

    @property
    def guide_fraction(self) -> float:
        """Fraction of env steps using guide policy in recent rollouts."""
        if self._rajs_total_steps == 0:
            return 0.0
        return self._rajs_guide_steps / self._rajs_total_steps

    @property
    def effective_h_max(self) -> float:
        """Current effective max guide horizon (h_max * beta)."""
        return self.rajs_config.h_max * self.beta
