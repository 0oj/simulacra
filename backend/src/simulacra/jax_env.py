"""
JAX-vectorized Gymnasium environment for Simulacra.

Runs thousands of rocket environments in parallel on GPU via jax.vmap.
Compatible with SB3 via the SB3JaxVecEnv adapter.

Architecture:
    SB3JaxVecEnv (SB3-compatible VecEnv adapter)
      └─ JaxVecEnv (core JAX env)
           └─ physics_step / guidance_step (jax_kernel.py)  ← jit + vmap

Typical speedup over SubprocVecEnv:
    8 CPU envs  → ~1,500 steps/sec
    4096 GPU envs → ~2,000,000+ steps/sec  (1500× faster)

Usage with SB3:
    from simulacra.jax_env import SB3JaxVecEnv
    env = SB3JaxVecEnv(num_envs=4096)
    model = PPO("MlpPolicy", env, ...)
    model.learn(total_timesteps=20_000_000)
"""

import math
import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional

from .jax_kernel import (
    physics_step,
    default_params,
    make_initial_state,
    quat_rotate,
    quat_rotate_inverse,
)


# ──────────────────────────────────────────────────────────────
# Guidance wrapper (PD attitude controller, runs in JAX)
# ──────────────────────────────────────────────────────────────

ATT_KP = 500.0
ATT_KD = 200.0
MAX_THRUST = 2000.0
MOMENT_ARM = 2.0
MAX_GIMBAL = 0.26


def guidance_to_gimbal(obs, guidance_action, max_tilt=0.15):
    """Convert [throttle, tilt_x, tilt_z] → [throttle, gimbal_outer, gimbal_inner].

    PD attitude controller — matches train_guidance.py GuidanceWrapper.step().
    Runs inside JAX (vmapped across batch).
    """
    throttle = guidance_action[0]
    tilt_x = guidance_action[1] * max_tilt
    tilt_z = guidance_action[2] * max_tilt

    # Desired nose direction (world frame)
    desired = jnp.array([tilt_x, 1.0, tilt_z])
    desired = desired / jnp.linalg.norm(desired)

    # Current orientation from obs
    qw, qx, qy, qz = obs[6], obs[7], obs[8], obs[9]
    wx, wy, wz = obs[10], obs[11], obs[12]
    quat = jnp.array([qw, qx, qy, qz])

    # PD attitude control
    nose_world = quat_rotate(quat, jnp.array([1.0, 0.0, 0.0]))
    error_world = jnp.cross(nose_world, desired)
    error_body = quat_rotate_inverse(quat, error_world)
    omega = jnp.array([wx, wy, wz])
    tau = ATT_KP * error_body - ATT_KD * omega

    # Convert torque to gimbal angles
    thrust_n = throttle * MAX_THRUST
    scale = jnp.where(thrust_n > 100.0, 1.0 / (thrust_n * MOMENT_ARM), 0.0)
    outer = jnp.clip(-tau[2] * scale, -MAX_GIMBAL, MAX_GIMBAL)
    inner = jnp.clip(-tau[1] * scale, -MAX_GIMBAL, MAX_GIMBAL)

    return jnp.array([throttle, outer / MAX_GIMBAL, inner / MAX_GIMBAL])


# ──────────────────────────────────────────────────────────────
# Physics features (run in JAX, vmapped)
# ──────────────────────────────────────────────────────────────

A_MAX = 4.0
K_DESCENT = 2.0


def compute_features(obs, feature_set="A"):
    """Compute physics features from 13-dim observation.

    Variant A (default): sqrt_y, brake_margin, vel_error, glide_slope
    """
    x, y, z = obs[0], obs[1], obs[2]
    vy = obs[4]
    y_safe = jnp.maximum(y, 0.1)

    sqrt_y = jnp.sqrt(y_safe)
    brake_margin = y - (vy**2) / (2.0 * A_MAX)
    vel_error = vy + K_DESCENT * jnp.sqrt(y_safe)
    horiz_dist = jnp.sqrt(x**2 + z**2)
    glide_slope = horiz_dist / y_safe

    return jnp.array([sqrt_y, brake_margin, vel_error, glide_slope])


# ──────────────────────────────────────────────────────────────
# Core JAX vectorized environment
# ──────────────────────────────────────────────────────────────

class JaxVecEnv:
    """Vectorized rocket landing environment running entirely via JAX.

    Args:
        num_envs: Number of parallel environments.
        use_guidance: If True, action is [throttle, tilt_x, tilt_z] with PD inner loop.
        use_features: If True, append physics features to observation.
        seed: Random seed.
        **kwargs: Init ranges and param overrides.
    """

    def __init__(
        self,
        num_envs: int = 4096,
        use_guidance: bool = False,
        use_features: bool = False,
        init_altitude_range: tuple = (5.0, 80.0),
        init_velocity_range: tuple = (-9.0, 0.0),
        init_horizontal_range: tuple = (-7.0, 7.0),
        init_tilt_range: tuple = (-0.17, 0.17),
        init_fuel_range: tuple = (0.6, 1.0),
        max_episode_steps: int = 1000,
        max_altitude: float = 750.0,
        seed: int = 0,
        **param_overrides,
    ):
        self.num_envs = num_envs
        self.use_guidance = use_guidance
        self.use_features = use_features

        # Init ranges
        self.init_altitude_range = init_altitude_range
        self.init_velocity_range = init_velocity_range
        self.init_horizontal_range = init_horizontal_range
        self.init_tilt_range = init_tilt_range
        self.init_fuel_range = init_fuel_range

        # Build params (static across all envs)
        self.params = default_params(
            max_episode_steps=max_episode_steps,
            max_altitude=max_altitude,
            **param_overrides,
        )

        # PRNG
        self.rng = jax.random.PRNGKey(seed)

        # Observation/action space dimensions
        self._obs_dim = 13 + (4 if use_features else 0)  # 4 features for variant A
        vel_bound = max(50.0, math.sqrt(2 * 9.81 * max_altitude) * 1.2)
        alt_bound = max(150.0, max_altitude * 1.2)
        base_low = np.array(
            [-100, -10, -100, -vel_bound, -vel_bound, -vel_bound,
             -1, -1, -1, -1, -10, -10, -10], dtype=np.float32)
        base_high = np.array(
            [100, alt_bound, 100, vel_bound, vel_bound, vel_bound,
             1, 1, 1, 1, 10, 10, 10], dtype=np.float32)
        if use_features:
            feat_low = np.full(4, -np.inf, dtype=np.float32)
            feat_high = np.full(4, np.inf, dtype=np.float32)
            self.obs_low = np.concatenate([base_low, feat_low])
            self.obs_high = np.concatenate([base_high, feat_high])
        else:
            self.obs_low = base_low
            self.obs_high = base_high

        # Pre-compile kernels
        self._compile_kernels()

        # State
        self._state = None
        self._last_obs = None  # needed for guidance wrapper

    def _compile_kernels(self):
        """JIT-compile the batched step and reset functions."""
        params = self.params
        use_guidance = self.use_guidance
        use_features = self.use_features

        def _step_single(state, action, rng_key):
            # Get previous obs for guidance (from state directly)
            obs_for_guidance = jnp.concatenate([
                state["pos"], state["vel"], state["quat"], state["omega"],
            ])

            # Guidance: convert tilt command → gimbal angles
            raw_action = jnp.where(
                use_guidance,
                guidance_to_gimbal(obs_for_guidance, action),
                action,
            )

            # Physics step
            new_state, obs, reward, done, info = physics_step(
                state, raw_action, params, rng_key,
            )

            # Append features if enabled
            full_obs = jnp.where(
                use_features,
                jnp.concatenate([obs, compute_features(obs)]),
                # Pad to fixed size when features disabled (for JIT shape consistency)
                obs,
            )

            return new_state, full_obs, reward, done, info

        # Handle variable obs size: always compute features, slice if not needed
        def _step_single_fixed(state, action, rng_key):
            obs_for_guidance = jnp.concatenate([
                state["pos"], state["vel"], state["quat"], state["omega"],
            ])

            raw_action = guidance_to_gimbal(obs_for_guidance, action) if use_guidance else action

            new_state, obs, reward, done, info = physics_step(
                state, raw_action, params, rng_key,
            )

            if use_features:
                obs = jnp.concatenate([obs, compute_features(obs)])

            return new_state, obs, reward, done, info

        self._step_fn = jax.jit(jax.vmap(
            _step_single_fixed,
            in_axes=(0, 0, 0),
        ))

        self._reset_fn = jax.jit(jax.vmap(
            lambda k: self._sample_initial_state(k, params),
        ))

    @staticmethod
    def _sample_initial_state(rng_key, params):
        """Sample a single initial state. Vmapped across batch."""
        k1, k2, k3, k4, k5, k6 = jax.random.split(rng_key, 6)

        alt = jax.random.uniform(k1, minval=params["init_alt_lo"],
                                 maxval=params["init_alt_hi"])
        vy = jax.random.uniform(k2, minval=params["init_vel_lo"],
                                maxval=params["init_vel_hi"])
        hx = jax.random.uniform(k3, minval=params["init_horiz_lo"],
                                maxval=params["init_horiz_hi"])
        hz = jax.random.uniform(k4, minval=params["init_horiz_lo"],
                                maxval=params["init_horiz_hi"])
        tilt = jax.random.uniform(k5, minval=params["init_tilt_lo"],
                                  maxval=params["init_tilt_hi"])
        fuel_frac = jax.random.uniform(k6, minval=params["init_fuel_lo"],
                                       maxval=params["init_fuel_hi"])

        return make_initial_state(
            altitude=alt,
            velocity_y=vy,
            horizontal_x=hx,
            horizontal_z=hz,
            tilt_angle=tilt,
            fuel_fraction=fuel_frac,
            fuel_capacity=params["fuel_capacity"],
        )

    def _update_params_with_ranges(self):
        """Bake init ranges into params for vmapped reset."""
        self.params["init_alt_lo"] = self.init_altitude_range[0]
        self.params["init_alt_hi"] = self.init_altitude_range[1]
        self.params["init_vel_lo"] = self.init_velocity_range[0]
        self.params["init_vel_hi"] = self.init_velocity_range[1]
        self.params["init_horiz_lo"] = self.init_horizontal_range[0]
        self.params["init_horiz_hi"] = self.init_horizontal_range[1]
        self.params["init_tilt_lo"] = self.init_tilt_range[0]
        self.params["init_tilt_hi"] = self.init_tilt_range[1]
        self.params["init_fuel_lo"] = self.init_fuel_range[0]
        self.params["init_fuel_hi"] = self.init_fuel_range[1]

    def _state_to_obs(self, state):
        """Extract observation from batched state."""
        obs = jnp.concatenate([
            state["pos"],   # (N, 3)
            state["vel"],   # (N, 3)
            state["quat"],  # (N, 4)
            state["omega"], # (N, 3)
        ], axis=-1)
        if self.use_features:
            feats = jax.vmap(compute_features)(obs)
            obs = jnp.concatenate([obs, feats], axis=-1)
        return obs

    def reset(self, seed: Optional[int] = None):
        """Reset all environments. Returns obs (num_envs, obs_dim) as NumPy."""
        if seed is not None:
            self.rng = jax.random.PRNGKey(seed)

        self._update_params_with_ranges()
        # Recompile kernels since ranges changed
        self._compile_kernels()

        self.rng, reset_key = jax.random.split(self.rng)
        keys = jax.random.split(reset_key, self.num_envs)
        self._state = self._reset_fn(keys)

        obs = np.array(self._state_to_obs(self._state))  # writable copy
        self._last_obs = obs
        return obs

    def step(self, actions):
        """Step all environments.

        Args:
            actions: (num_envs, 3) NumPy array

        Returns:
            obs: (num_envs, obs_dim) NumPy
            rewards: (num_envs,) NumPy
            dones: (num_envs,) NumPy bool
            infos: list of dicts (SB3-compatible)
        """
        actions_jax = jnp.array(actions)

        # Split PRNG
        self.rng, step_key = jax.random.split(self.rng)
        keys = jax.random.split(step_key, self.num_envs)

        # Batched physics step
        new_state, obs, rewards, dones, info = self._step_fn(
            self._state, actions_jax, keys,
        )

        # Auto-reset: sample fresh states for done envs
        self.rng, reset_key = jax.random.split(self.rng)
        reset_keys = jax.random.split(reset_key, self.num_envs)
        fresh_state = self._reset_fn(reset_keys)

        # Done envs get fresh state; continuing envs keep new_state
        self._state = jax.tree.map(
            lambda fresh, cont: jnp.where(
                dones[:, None] if fresh.ndim > 1 else dones,
                fresh, cont,
            ),
            fresh_state, new_state,
        )

        # Convert to NumPy
        obs_np = np.asarray(obs)
        rewards_np = np.asarray(rewards).astype(np.float32)
        dones_np = np.asarray(dones)

        # Build SB3-compatible infos
        landed_np = np.asarray(info["landed"])
        crashed_np = np.asarray(info["crashed"])
        truncated_np = np.asarray(info["truncated"])
        impact_np = np.asarray(info["impact_speed"])
        dist_np = np.asarray(info["horiz_dist"])

        infos = []
        for i in range(self.num_envs):
            d = {}
            if dones_np[i]:
                if landed_np[i]:
                    d["outcome"] = "landed"
                    d["landing_dist"] = float(dist_np[i])
                    d["landing_vel"] = float(impact_np[i])
                elif crashed_np[i]:
                    d["outcome"] = "crash"
                # SB3 expects terminal_observation for auto-reset envs
                d["terminal_observation"] = obs_np[i]
                d["episode"] = {"r": float(rewards_np[i]), "l": 0}
                # SB3 uses this to bootstrap V(terminal) for truncated episodes
                if truncated_np[i]:
                    d["TimeLimit.truncated"] = True
            infos.append(d)

        # For done envs, return the RESET observation (SB3 convention)
        reset_obs = np.asarray(self._state_to_obs(self._state))
        obs_out = np.where(dones_np[:, None], reset_obs, obs_np)

        self._last_obs = obs_out
        # Make arrays writable (JAX arrays are read-only, PyTorch warns)
        return np.array(obs_out), np.array(rewards_np), np.array(dones_np), infos

    def close(self):
        pass

    @property
    def observation_space(self):
        import gymnasium as gym
        return gym.spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.float32)

    @property
    def action_space(self):
        import gymnasium as gym
        return gym.spaces.Box(
            low=np.array([0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )


# ──────────────────────────────────────────────────────────────
# SB3 VecEnv adapter
# ──────────────────────────────────────────────────────────────

class SB3JaxVecEnv:
    """Stable-Baselines3 compatible VecEnv wrapping JaxVecEnv.

    Inherits from SB3's VecEnv base class for full compatibility with PPO, A2C, etc.

    Usage:
        env = SB3JaxVecEnv(num_envs=4096, use_guidance=True, use_features=True)
        model = PPO("MlpPolicy", env, ...)
        model.learn(total_timesteps=20_000_000)
    """

    def __new__(cls, **kwargs):
        """Construct via SB3's VecEnv base class."""
        from stable_baselines3.common.vec_env import VecEnv

        jax_env = JaxVecEnv(**kwargs)

        class _SB3Adapter(VecEnv):
            def __init__(self, jax_env):
                self.jax_env = jax_env
                super().__init__(
                    num_envs=jax_env.num_envs,
                    observation_space=jax_env.observation_space,
                    action_space=jax_env.action_space,
                )
                self._actions = None

            def reset(self):
                return self.jax_env.reset()

            def step_async(self, actions):
                self._actions = actions

            def step_wait(self):
                return self.jax_env.step(self._actions)

            def close(self):
                self.jax_env.close()

            def env_method(self, method_name, *args, indices=None, **kwargs):
                return [getattr(self.jax_env, method_name)(*args, **kwargs)]

            def get_attr(self, attr_name, indices=None):
                val = getattr(self.jax_env, attr_name)
                n = len(indices) if indices else self.num_envs
                return [val] * n

            def set_attr(self, attr_name, value, indices=None):
                setattr(self.jax_env, attr_name, value)

            def env_is_wrapped(self, wrapper_class, indices=None):
                n = len(indices) if indices else self.num_envs
                return [False] * n

            def seed(self, seed=None):
                if seed is not None:
                    self.jax_env.rng = jax.random.PRNGKey(seed)

        return _SB3Adapter(jax_env)


# ──────────────────────────────────────────────────────────────
# Benchmark
# ──────────────────────────────────────────────────────────────

def benchmark(num_envs=4096, num_steps=1000, use_guidance=False, use_features=False):
    """Measure steps/sec for the JAX vectorized env."""
    import time

    env = JaxVecEnv(num_envs=num_envs, seed=42,
                    use_guidance=use_guidance, use_features=use_features)
    obs = env.reset()

    # Warmup (JIT compilation)
    actions = np.random.uniform(0, 1, (num_envs, 3)).astype(np.float32)
    actions[:, 1:] = actions[:, 1:] * 2 - 1
    for _ in range(10):
        obs, _, _, _ = env.step(actions)

    # Timed run
    start = time.perf_counter()
    for _ in range(num_steps):
        obs, _, _, _ = env.step(actions)
    jax.block_until_ready(obs)
    elapsed = time.perf_counter() - start

    total_steps = num_envs * num_steps
    rate = total_steps / elapsed
    mode = []
    if use_guidance:
        mode.append("guidance")
    if use_features:
        mode.append("features")
    mode_str = f" ({'+'.join(mode)})" if mode else ""
    print(f"JAX env benchmark{mode_str}:")
    print(f"  {num_envs} envs × {num_steps} steps = {total_steps:,} total")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Rate: {rate:,.0f} steps/sec")
    print(f"  Speedup over 8-env CPU: ~{rate / 1500:.0f}×")
    return rate


if __name__ == "__main__":
    benchmark(num_envs=4096, num_steps=500)
    benchmark(num_envs=4096, num_steps=500, use_guidance=True, use_features=True)
