"""
Hierarchical guidance PPO: RL learns WHERE to point, PD controller handles HOW.

Architecture (two-layer control):
  Outer loop (RL, 50Hz):  outputs [throttle, tilt_x, tilt_z]
    - throttle: thrust magnitude (0-1)
    - tilt_x, tilt_z: desired nose direction offset from vertical (-1 to +1)
      scaled by max_tilt (default 0.15 rad ≈ 8.6°)

  Inner loop (PD, 50Hz):  tracks the commanded nose direction
    - Computes attitude error between current and desired nose direction
    - Converts to gimbal angles via torque-based PD law
    - Handles actuator dynamics, rate limits, lag

This separation is how real rockets work: guidance computes desired attitude,
attitude controller tracks it. It's also highly interpretable — you can
directly read "the agent wants to tilt 5° toward the target" from the action.

Usage:
    cd backend && uv run python scripts/train_guidance.py
"""

import sys
import os
import signal
import subprocess
import math
import time
from pathlib import Path
from datetime import datetime
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from simulacra.gym_env import RocketLandingEnvV2

SAVE_DIR = Path("runs/guidance_ppo")
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


def _rotate_body_to_world(qw, qx, qy, qz, v):
    qv = np.array([qx, qy, qz])
    vn = np.array(v, dtype=float)
    c1 = np.cross(qv, vn)
    c2 = np.cross(qv, c1)
    return vn + 2.0 * qw * c1 + 2.0 * c2


def _rotate_world_to_body(qw, qx, qy, qz, v):
    qv = np.array([-qx, -qy, -qz])
    vn = np.array(v, dtype=float)
    c1 = np.cross(qv, vn)
    c2 = np.cross(qv, c1)
    return vn + 2.0 * qw * c1 + 2.0 * c2


class GuidanceWrapper(gym.Wrapper):
    """
    3D action space: [throttle, tilt_x, tilt_z]

    The RL agent commands a desired nose direction. The inner PD controller
    tracks it via torque-based gimbal control.

    Action semantics:
      throttle ∈ [0, 1]: thrust fraction
      tilt_x ∈ [-1, 1]:  desired nose X component (scaled by max_tilt)
      tilt_z ∈ [-1, 1]:  desired nose Z component (scaled by max_tilt)

    Desired nose direction in world frame:
      desired = normalize([tilt_x * max_tilt, 1, tilt_z * max_tilt])

    When tilt_x = tilt_z = 0: desired = [0, 1, 0] = vertical (hover).
    When tilt_x = 1: desired tilts max_tilt rad toward +X (steer toward -X).
    """

    ATT_KP = 500.0
    ATT_KD = 200.0
    MAX_THRUST = 2000.0
    MOMENT_ARM = 2.0
    MAX_GIMBAL = 0.26

    def __init__(self, env, max_tilt=0.15):
        super().__init__(env)
        self.max_tilt = max_tilt  # max commanded tilt in radians (~8.6°)

        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )

    def step(self, action):
        throttle = float(action[0])
        tilt_x = float(action[1]) * self.max_tilt
        tilt_z = float(action[2]) * self.max_tilt
        obs = self._last_obs

        # Desired nose direction: slightly off-vertical per RL command
        desired = np.array([tilt_x, 1.0, tilt_z])
        desired = desired / np.linalg.norm(desired)

        # PD attitude controller tracking the desired direction
        qw, qx, qy, qz = [float(obs[i]) for i in range(6, 10)]
        wx, wy, wz = [float(obs[i]) for i in range(10, 13)]

        nose_world = _rotate_body_to_world(qw, qx, qy, qz, [1, 0, 0])
        error_world = np.cross(nose_world, desired)
        error_body = _rotate_world_to_body(qw, qx, qy, qz, error_world)
        omega = np.array([wx, wy, wz])
        tau = self.ATT_KP * error_body - self.ATT_KD * omega

        thrust_n = throttle * self.MAX_THRUST
        if thrust_n > 100:
            scale = 1.0 / (thrust_n * self.MOMENT_ARM)
            outer = float(np.clip(-tau[2] * scale, -self.MAX_GIMBAL, self.MAX_GIMBAL))
            inner = float(np.clip(-tau[1] * scale, -self.MAX_GIMBAL, self.MAX_GIMBAL))
        else:
            outer = 0.0
            inner = 0.0

        outer_norm = outer / self.MAX_GIMBAL
        inner_norm = inner / self.MAX_GIMBAL

        full = np.array([throttle, outer_norm, inner_norm], dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(full)
        self._last_obs = obs
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info


class GuidanceReward(gym.Wrapper):
    """
    Reward shaped for guidance learning.

    Compared to throttle-only reward:
    - Stronger horizontal position signal (the main thing to learn)
    - Potential-based shaping for horizontal approach
    - Tilt penalty is lighter (we WANT the agent to tilt to steer)
    """

    def __init__(self, env):
        super().__init__(env)
        self._prev_horiz_dist = None
        self._prev_alt = None

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = self._reward(obs, info, terminated, truncated)
        return obs, reward, terminated, truncated, info

    def _reward(self, obs, info, terminated, truncated):
        alt = float(obs[1])
        vy = float(obs[4])
        vx, vz = float(obs[3]), float(obs[5])
        wx, wy, wz = float(obs[10]), float(obs[11]), float(obs[12])

        nose = _rotate_body_to_world(
            float(obs[6]), float(obs[7]), float(obs[8]), float(obs[9]),
            [1, 0, 0],
        )
        tilt = math.acos(max(-1.0, min(1.0, nose[1])))
        omega = math.sqrt(wx**2 + wy**2 + wz**2)
        v_horiz = math.sqrt(vx**2 + vz**2)
        horiz_dist = math.sqrt(float(obs[0])**2 + float(obs[2])**2)

        outcome = info.get("outcome")

        if outcome == "landed":
            vel = info.get("landing_vel", 3.0)
            lt = math.radians(info.get("landing_tilt", 15.0))
            dist = info.get("landing_dist", 10.0)
            return (200.0
                    + 50.0 * max(0, 1 - vel / 3)
                    + 20.0 * max(0, 1 - lt / 0.26)
                    + 80.0 * max(0, 1 - dist / 3.0))  # strong position bonus, tight radius

        if outcome == "crash":
            return -200.0

        if terminated and outcome is None:
            return -100.0

        if truncated:
            return -50.0 - 0.5 * alt

        # Dense shaping
        max_decel = 8.0  # ~mid-fuel TWR, not worst-case full-fuel
        max_safe_vy = math.sqrt(2 * max_decel * max(0.1, alt))
        excess = max(0.0, abs(vy) - max_safe_vy)

        descent_reward = 0.0
        if vy < 0:
            descent_reward = 0.03 * min(abs(vy), max_safe_vy)
        if alt < 2.0:
            if -1.5 < vy < 0:
                descent_reward += 0.15
            elif vy < -3.0:
                descent_reward -= 0.5

        # Potential-based shaping: reward descent progress, not absolute altitude.
        # Old: -0.005 * alt (accumulates to -2500 at 500m, drowning terminal bonus).
        # New: 0.005 * (prev_alt - alt) — identical gradient, no massive negative baseline.
        alt_shaping = 0.0
        if self._prev_alt is not None:
            alt_shaping = 0.005 * (self._prev_alt - alt)
        self._prev_alt = alt

        # Potential-based shaping: reward for reducing horizontal distance
        approach_reward = 0.0
        if self._prev_horiz_dist is not None:
            approach_reward = 0.15 * (self._prev_horiz_dist - horiz_dist)
        self._prev_horiz_dist = horiz_dist

        return (
            alt_shaping
            + descent_reward
            - 0.5 * excess ** 2
            - 0.05 * tilt ** 2      # lighter tilt penalty (tilting is useful now)
            - 0.03 * omega
            - 0.01 * v_horiz        # lighter — horizontal velocity is sometimes needed
            - 0.02 * horiz_dist     # strong distance penalty
            + approach_reward
        )

    def reset(self, **kwargs):
        self._prev_horiz_dist = None
        self._prev_alt = None
        return self.env.reset(**kwargs)


def make_env(seed, **env_kwargs):
    def _init():
        from stable_baselines3.common.monitor import Monitor
        env = RocketLandingEnvV2(**env_kwargs)
        env = GuidanceWrapper(env)
        env = GuidanceReward(env)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def evaluate(model, n=200, **env_kwargs):
    env = GuidanceWrapper(RocketLandingEnvV2(**env_kwargs))
    landed, total_dist = 0, 0.0
    for ep in range(n):
        obs, _ = env.reset(seed=ep + 5000)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, t, tr, info = env.step(action)
            done = t or tr
        if info.get("outcome") == "landed":
            landed += 1
            total_dist += info.get("landing_dist", 0)
    env.close()
    rate = landed / n
    avg_dist = total_dist / max(1, landed)
    print(f"    {landed}/{n} landed ({100*rate:.0f}%), avg dist={avg_dist:.2f}m")
    return rate, avg_dist


def main():
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    (SAVE_DIR / "checkpoints").mkdir(exist_ok=True)

    total_steps = 5_000_000
    n_envs = 4

    print("=" * 60)
    print("Guidance PPO — hierarchical control")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Action: [throttle, tilt_x, tilt_z]")
    print(f"Inner loop: torque-based PD attitude controller")
    print("=" * 60)

    env_kwargs = dict(
        init_altitude_range=(5.0, 60.0),
        init_velocity_range=(-8.0, 0.0),
        init_horizontal_range=(-5.0, 5.0),
        init_tilt_range=(-0.15, 0.15),
        init_fuel_range=(0.6, 1.0),
    )

    train_envs = SubprocVecEnv([make_env(i, **env_kwargs) for i in range(n_envs)])
    eval_env = DummyVecEnv([make_env(999, **env_kwargs)])

    # Resume from latest checkpoint if available
    checkpoints = sorted(SAVE_DIR.glob("checkpoints/guidance_*_steps.zip"))
    resume_path = checkpoints[-1] if checkpoints else None
    if resume_path and resume_path.exists():
        print(f"  Resuming from {resume_path}")
        model = PPO.load(resume_path, env=train_envs,
                         tensorboard_log=str(SAVE_DIR / "tensorboard"))
    else:
        model = PPO(
            "MlpPolicy",
            train_envs,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0,
            tensorboard_log=str(SAVE_DIR / "tensorboard"),
            policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
        )

    class ProgressCB(BaseCallback):
        def __init__(self):
            super().__init__()
            self.start = time.time()
            self.last_log = time.time()
            self.episodes = 0
            self.landings = 0
            self.recent = deque(maxlen=200)
            self.recent_dist = deque(maxlen=200)

        def _on_step(self):
            if _shutdown:
                self.model.save(str(SAVE_DIR / "checkpoints" / "interrupted"))
                notify("Paused", f"Step {self.num_timesteps:,}")
                return False

            for info in self.locals.get("infos", []):
                if "episode" in info:
                    self.episodes += 1
                    landed = info.get("outcome") == "landed"
                    self.recent.append(1 if landed else 0)
                    if landed:
                        self.landings += 1
                        self.recent_dist.append(info.get("landing_dist", 99))
                        if self.landings == 1:
                            notify("First landing!", f"Step {self.num_timesteps:,}")

            if time.time() - self.last_log > 60:
                self.last_log = time.time()
                rate = sum(self.recent) / max(1, len(self.recent))
                avg_dist = np.mean(self.recent_dist) if self.recent_dist else 0
                elapsed = (time.time() - self.start) / 60
                print(f"  [{elapsed:.0f}m] steps={self.num_timesteps:,}, "
                      f"landings={self.landings}, rate={100*rate:.0f}%, "
                      f"avg_dist={avg_dist:.2f}m")
            return True

    callbacks = [
        ProgressCB(),
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
            name_prefix="guidance",
        ),
    ]

    notify("guidance PPO started", f"{total_steps:,} steps, hierarchical control")

    try:
        model.learn(total_timesteps=total_steps, callback=callbacks, progress_bar=True)
    except KeyboardInterrupt:
        pass

    model.save(SAVE_DIR / "final_model")

    # Final eval
    print(f"\n{'='*60}")
    print("FINAL EVALUATION")
    print(f"{'='*60}")
    for name, kw in [
        ("L0 (easy)", dict(init_altitude_range=(3., 8.), init_velocity_range=(-2., 0.),
                     init_horizontal_range=(-0.5, 0.5), init_tilt_range=(-0.02, 0.02))),
        ("L3 (hard)", dict(init_altitude_range=(10., 60.), init_velocity_range=(-8., 0.),
                     init_horizontal_range=(-5., 5.), init_tilt_range=(-0.15, 0.15))),
    ]:
        print(f"  {name}:")
        evaluate(model, 200, **kw)

    notify("guidance PPO done", "Check results")
    train_envs.close()
    eval_env.close()


if __name__ == "__main__":
    main()
