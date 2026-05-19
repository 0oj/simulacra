"""
SAC (Soft Actor-Critic) for throttle-only rocket landing.

SAC advantages over PPO for this task:
- Off-policy: replay buffer prevents forgetting good experiences
- Entropy-regularized: naturally balances exploration vs exploitation
- More stable: doesn't oscillate like PPO with curriculum
- Better sample efficiency for continuous control

Usage:
    cd backend && uv run python scripts/train_sac_throttle.py
"""

import sys
import os
import signal
import subprocess
import json
import math
import time
from pathlib import Path
from datetime import datetime
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from simulacra.gym_env import RocketLandingEnvV2

SAVE_DIR = Path("runs/sac_throttle")
_shutdown = False


def _signal_handler(signum, frame):
    global _shutdown
    _shutdown = True
    print("\n[!] Shutdown requested...")


def notify(title, msg):
    try:
        subprocess.run([os.path.expanduser("~/.claude/scripts/pushover"), msg, f"Simulacra: {title}"],
                       capture_output=True, timeout=10)
    except Exception:
        pass
    print(f"[NOTIFY] {title}: {msg}")


# ── Wrappers (same as train_landing_rl.py) ───────────────

class ThrottleOnlyWrapper(gym.Wrapper):
    def __init__(self, env, gimbal_gain=0.5, gimbal_clamp=0.3):
        super().__init__(env)
        self.gimbal_gain = gimbal_gain
        self.gimbal_clamp = gimbal_clamp
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
        )

    def step(self, action):
        throttle = float(action[0])
        obs = self._last_obs
        wy, wz = float(obs[11]), float(obs[12])
        go = np.clip(-self.gimbal_gain * wz, -self.gimbal_clamp, self.gimbal_clamp)
        gi = np.clip(-self.gimbal_gain * wy, -self.gimbal_clamp, self.gimbal_clamp)
        full = np.array([throttle, go, gi], dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(full)
        self._last_obs = obs
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info


class LandingRewardWrapper(gym.Wrapper):
    """Physics-based reward: penalize excess speed, reward safe descent."""

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = self._reward(obs, info, terminated, truncated)
        return obs, reward, terminated, truncated, info

    def _reward(self, obs, info, terminated, truncated):
        alt = float(obs[1])
        vy = float(obs[4])
        vx, vz = float(obs[3]), float(obs[5])
        qw = float(obs[6])
        wx, wy, wz = float(obs[10]), float(obs[11]), float(obs[12])

        tilt = 2.0 * math.acos(min(1.0, abs(qw)))
        omega = math.sqrt(wx**2 + wy**2 + wz**2)
        v_horiz = math.sqrt(vx**2 + vz**2)

        outcome = info.get("outcome")

        if outcome == "landed":
            vel = info.get("landing_vel", 3.0)
            lt = math.radians(info.get("landing_tilt", 15.0))
            return 200.0 + 50.0 * max(0, 1 - vel / 3) + 20.0 * max(0, 1 - lt / 0.26)

        if outcome == "crash":
            return -200.0

        if terminated and outcome is None:
            return -100.0

        if truncated:
            return -50.0 - 0.5 * alt

        # Dense shaping
        max_decel = 4.0
        max_safe_vy = math.sqrt(2 * max_decel * max(0.1, alt))
        excess = max(0.0, abs(vy) - max_safe_vy)

        descent_reward = 0.0
        if vy < 0:
            descent_reward = 0.02 * min(abs(vy), max_safe_vy)

        if alt < 2.0:
            if -1.5 < vy < 0:
                descent_reward += 0.1
            elif vy < -3.0:
                descent_reward -= 0.5

        return (
            0.02
            + descent_reward
            - 0.5 * excess ** 2
            - 0.10 * tilt ** 2
            - 0.03 * omega
            - 0.01 * v_horiz
        )


def make_env(seed, **env_kwargs):
    def _init():
        from stable_baselines3.common.monitor import Monitor
        env = RocketLandingEnvV2(**env_kwargs)
        env = ThrottleOnlyWrapper(env)
        env = LandingRewardWrapper(env)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def evaluate(model, n=200, **env_kwargs):
    env = ThrottleOnlyWrapper(RocketLandingEnvV2(**env_kwargs))
    landed = 0
    for ep in range(n):
        obs, _ = env.reset(seed=ep + 5000)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, t, tr, info = env.step(action)
            done = t or tr
        if info.get("outcome") == "landed":
            landed += 1
    env.close()
    return landed / n


def main():
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    (SAVE_DIR / "checkpoints").mkdir(exist_ok=True)

    total_steps = 3_000_000

    print("=" * 60)
    print("SAC Throttle-Only Landing")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)

    # Train directly on L3 (full difficulty) — SAC handles it better
    env_kwargs = dict(
        init_altitude_range=(5.0, 60.0),
        init_velocity_range=(-8.0, 0.0),
        init_horizontal_range=(-5.0, 5.0),
        init_tilt_range=(-0.15, 0.15),
        init_fuel_range=(0.6, 1.0),
    )

    from stable_baselines3.common.vec_env import VecNormalize

    train_env = DummyVecEnv([make_env(0, **env_kwargs)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True,
                             clip_obs=10.0, clip_reward=10.0, gamma=0.99)

    eval_env = DummyVecEnv([make_env(999, **env_kwargs)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            clip_obs=10.0, training=False)

    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=500_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef="auto",
        learning_starts=10_000,
        verbose=0,
        tensorboard_log=str(SAVE_DIR / "tensorboard"),
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    class NotifyCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.best_rate = 0
            self.start = time.time()
            self.last_log = time.time()

        def _on_step(self):
            if _shutdown:
                self.model.save(str(SAVE_DIR / "checkpoints" / "interrupted"))
                notify("SAC Paused", f"Step {self.num_timesteps:,}")
                return False
            if time.time() - self.last_log > 120:
                self.last_log = time.time()
                elapsed = (time.time() - self.start) / 60
                print(f"  [{elapsed:.0f}m] steps={self.num_timesteps:,}")
            return True

    callbacks = [
        NotifyCallback(),
        EvalCallback(
            eval_env,
            best_model_save_path=str(SAVE_DIR / "best_model"),
            log_path=str(SAVE_DIR / "eval_logs"),
            eval_freq=5_000,
            n_eval_episodes=20,
            deterministic=True,
            verbose=0,
        ),
        CheckpointCallback(
            save_freq=100_000,
            save_path=str(SAVE_DIR / "checkpoints"),
            name_prefix="sac",
        ),
    ]

    notify("SAC started", f"{total_steps:,} steps, L3 difficulty")

    try:
        model.learn(total_timesteps=total_steps, callback=callbacks, progress_bar=True)
    except KeyboardInterrupt:
        pass

    model.save(SAVE_DIR / "final_model")

    # Final eval
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    for name, kw in [
        ("L0", dict(init_altitude_range=(3., 8.), init_velocity_range=(-2., 0.),
                     init_horizontal_range=(-0.5, 0.5), init_tilt_range=(-0.02, 0.02))),
        ("L2", dict(init_altitude_range=(8., 30.), init_velocity_range=(-6., 0.),
                     init_horizontal_range=(-2., 2.), init_tilt_range=(-0.1, 0.1))),
        ("L3", dict(init_altitude_range=(10., 60.), init_velocity_range=(-8., 0.),
                     init_horizontal_range=(-5., 5.), init_tilt_range=(-0.15, 0.15))),
    ]:
        rate = evaluate(model, 200, **kw)
        print(f"  {name}: {100*rate:.0f}%")

    notify("SAC done", "Check results")
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
