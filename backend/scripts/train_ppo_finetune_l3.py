"""
Fine-tune the best PPO checkpoint on L3 only (no curriculum).

Takes the 60% L3 model and trains it exclusively on hard scenarios.
No curriculum — just raw L3 difficulty.

Usage:
    cd backend && uv run python scripts/train_ppo_finetune_l3.py
"""

import sys
import os
import signal
import subprocess
import math
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from simulacra.gym_env import RocketLandingEnvV2

SAVE_DIR = Path("runs/ppo_finetune_l3")
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
            0.02 + descent_reward - 0.5 * excess ** 2
            - 0.10 * tilt ** 2 - 0.03 * omega - 0.01 * v_horiz
        )


def make_env(seed):
    def _init():
        from stable_baselines3.common.monitor import Monitor
        env = RocketLandingEnvV2(
            init_altitude_range=(5.0, 60.0),
            init_velocity_range=(-8.0, 0.0),
            init_horizontal_range=(-5.0, 5.0),
            init_tilt_range=(-0.15, 0.15),
            init_fuel_range=(0.6, 1.0),
        )
        env = ThrottleOnlyWrapper(env)
        env = LandingRewardWrapper(env)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def evaluate(model, n=200):
    env = ThrottleOnlyWrapper(RocketLandingEnvV2(
        init_altitude_range=(10., 60.), init_velocity_range=(-8., 0.),
        init_horizontal_range=(-5., 5.), init_tilt_range=(-0.15, 0.15),
    ))
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
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    (SAVE_DIR / "checkpoints").mkdir(exist_ok=True)

    # Load best model from throttle_ppo
    best_path = Path("runs/throttle_ppo/best_model/best_model.zip")
    if not best_path.exists():
        print("No best model found! Run train_landing_rl.py first.")
        return

    print("=" * 60)
    print("PPO Fine-tune on L3 (no curriculum)")
    print(f"Loading best model from {best_path}")
    print("=" * 60)

    n_envs = 4
    train_envs = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    eval_env = DummyVecEnv([make_env(999)])

    model = PPO.load(
        str(best_path),
        env=train_envs,
        tensorboard_log=str(SAVE_DIR / "tensorboard"),
        # Lower learning rate for fine-tuning
        learning_rate=1e-4,
        ent_coef=0.005,  # Less exploration — start from good policy
        clip_range=0.1,  # Smaller updates — preserve what works
    )

    # Baseline eval
    rate = evaluate(model)
    print(f"Baseline L3: {100*rate:.0f}%")
    notify("PPO finetune started", f"Baseline: {100*rate:.0f}% L3")

    class ShutdownCB(BaseCallback):
        def _on_step(self):
            if _shutdown:
                self.model.save(str(SAVE_DIR / "checkpoints" / "interrupted"))
                return False
            return True

    callbacks = [
        ShutdownCB(),
        EvalCallback(
            eval_env,
            best_model_save_path=str(SAVE_DIR / "best_model"),
            log_path=str(SAVE_DIR / "eval_logs"),
            eval_freq=10_000 // n_envs,
            n_eval_episodes=20,
            deterministic=True,
            verbose=0,
        ),
        CheckpointCallback(
            save_freq=100_000 // n_envs,
            save_path=str(SAVE_DIR / "checkpoints"),
            name_prefix="ppo_ft",
        ),
    ]

    total_steps = 3_000_000

    try:
        model.learn(total_timesteps=total_steps, callback=callbacks, progress_bar=True)
    except KeyboardInterrupt:
        pass

    model.save(SAVE_DIR / "final_model")

    rate = evaluate(model)
    print(f"\nFinal L3: {100*rate:.0f}%")
    notify("PPO finetune done", f"L3: {100*rate:.0f}%")

    train_envs.close()
    eval_env.close()


if __name__ == "__main__":
    main()
