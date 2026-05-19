"""
Gimbal residual PPO: learn horizontal steering corrections on top of PD attitude controller.

Architecture (from residual RL literature):
  - Frozen throttle policy (from train_throttle_v2 best model)
  - PD attitude controller as gimbal baseline (torque-based, proven stable)
  - RL learns 2D gimbal residuals [outer, inner] to steer toward landing target

Key techniques (Silver et al. 2018, DAWN 2025, Ankile et al. 2024):
  1. Zero-initialize residual output layer → starts as pure base controller
  2. L2 penalty on residual magnitude → keeps corrections small
  3. Potential-based shaping for horizontal approach → per-step gradient signal
  4. Increased position reward → the signal this agent needs to learn
  5. PPO not SAC → avoids entropy miscalibration with small residual space

Usage:
    cd backend && uv run python scripts/train_gimbal_residual_v2.py
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

SAVE_DIR = Path("runs/gimbal_residual_v2")
THROTTLE_MODEL_PATH = Path("runs/throttle_v2/best_model/best_model.zip")
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


class GimbalResidualWrapper(gym.Wrapper):
    """
    2D action space: [gimbal_residual_outer, gimbal_residual_inner]

    Throttle: from frozen PPO model (deterministic).
    Gimbal baseline: torque-based PD attitude controller (LandingController-style).
    Gimbal actual: baseline + residual * bound.

    The residual lets the RL agent tilt the rocket slightly off-vertical
    to steer toward the landing target — something the pure PD controller
    (which always targets vertical) cannot do.
    """

    ATT_KP = 500.0
    ATT_KD = 200.0
    MAX_THRUST = 2000.0
    MOMENT_ARM = 2.0
    MAX_GIMBAL = 0.26

    def __init__(self, env, throttle_model, residual_bound=0.3):
        super().__init__(env)
        self.throttle_model = throttle_model
        self.residual_bound = residual_bound

        # 2D action: gimbal residuals only
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

    def step(self, action):
        res_outer = float(action[0]) * self.residual_bound
        res_inner = float(action[1]) * self.residual_bound
        obs = self._last_obs

        # Throttle from frozen model
        throttle_action, _ = self.throttle_model.predict(obs, deterministic=True)
        throttle = float(throttle_action[0])

        # PD attitude baseline
        qw, qx, qy, qz = [float(obs[i]) for i in range(6, 10)]
        wx, wy, wz = [float(obs[i]) for i in range(10, 13)]

        nose_world = _rotate_body_to_world(qw, qx, qy, qz, [1, 0, 0])
        desired = np.array([0, 1, 0])
        error_world = np.cross(nose_world, desired)
        error_body = _rotate_world_to_body(qw, qx, qy, qz, error_world)
        omega = np.array([wx, wy, wz])
        tau = self.ATT_KP * error_body - self.ATT_KD * omega

        thrust_n = throttle * self.MAX_THRUST
        if thrust_n > 100:
            scale = 1.0 / (thrust_n * self.MOMENT_ARM)
            outer_base = float(np.clip(-tau[2] * scale, -self.MAX_GIMBAL, self.MAX_GIMBAL))
            inner_base = float(np.clip(-tau[1] * scale, -self.MAX_GIMBAL, self.MAX_GIMBAL))
        else:
            outer_base = 0.0
            inner_base = 0.0

        # Baseline + residual, normalized to [-1, 1]
        outer_norm = np.clip(outer_base / self.MAX_GIMBAL + res_outer, -1.0, 1.0)
        inner_norm = np.clip(inner_base / self.MAX_GIMBAL + res_inner, -1.0, 1.0)

        full = np.array([throttle, outer_norm, inner_norm], dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(full)
        self._last_obs = obs
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info


class HorizontalTargetingReward(gym.Wrapper):
    """
    Reward function focused on horizontal targeting.

    Key differences from throttle-only reward:
    - 4x stronger horizontal position penalties and bonuses
    - Potential-based shaping: reward for reducing horiz distance each step
    - L2 penalty on residual magnitude (keeps corrections small)
    - Landing still rewarded, but position accuracy is the main signal
    """

    def __init__(self, env, residual_l2_coeff=0.5):
        super().__init__(env)
        self.residual_l2_coeff = residual_l2_coeff
        self._prev_horiz_dist = None

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = self._reward(obs, action, info, terminated, truncated)
        return obs, reward, terminated, truncated, info

    def _reward(self, obs, action, info, terminated, truncated):
        alt = float(obs[1])
        vy = float(obs[4])
        vx, vz = float(obs[3]), float(obs[5])
        wx, wy, wz = float(obs[10]), float(obs[11]), float(obs[12])

        # Tilt from nose direction
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
                    # 4x position bonus with tighter radius
                    + 80.0 * max(0, 1 - dist / 5.0))

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
            descent_reward = 0.03 * min(abs(vy), max_safe_vy)
        if alt < 2.0:
            if -1.5 < vy < 0:
                descent_reward += 0.15
            elif vy < -3.0:
                descent_reward -= 0.5

        # Potential-based shaping: reward for getting closer to target
        approach_reward = 0.0
        if self._prev_horiz_dist is not None:
            approach_reward = 0.1 * (self._prev_horiz_dist - horiz_dist)
        self._prev_horiz_dist = horiz_dist

        # L2 penalty on residual magnitude
        res_penalty = self.residual_l2_coeff * (float(action[0])**2 + float(action[1])**2)

        return (
            -0.005 * alt
            + descent_reward
            - 0.5 * excess ** 2
            - 0.10 * tilt ** 2
            - 0.03 * omega
            - 0.02 * v_horiz
            # 4x horizontal distance penalty
            - 0.02 * horiz_dist
            + approach_reward
            - res_penalty
        )

    def reset(self, **kwargs):
        self._prev_horiz_dist = None
        return self.env.reset(**kwargs)


def make_env(seed, throttle_model_path, residual_bound=0.3, **env_kwargs):
    def _init():
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor

        throttle_model = PPO.load(str(throttle_model_path))
        env = RocketLandingEnvV2(**env_kwargs)
        env = GimbalResidualWrapper(env, throttle_model, residual_bound=residual_bound)
        env = HorizontalTargetingReward(env)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def zero_init_action_net(model):
    """Zero-initialize the residual output layer.

    This ensures the agent starts by outputting residual=0 (pure base controller).
    Critical for residual RL stability (Silver et al. 2018).
    """
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()
        # Also set log_std low so initial exploration is small
        model.policy.log_std.fill_(-2.0)  # std ≈ 0.14
    print("  Zero-initialized action_net weights and bias")
    print(f"  Initial log_std = {model.policy.log_std.data.tolist()}")


def evaluate(model, throttle_model_path, n=200, residual_bound=0.3, **env_kwargs):
    from stable_baselines3 import PPO
    throttle_model = PPO.load(str(throttle_model_path))
    env = GimbalResidualWrapper(
        RocketLandingEnvV2(**env_kwargs), throttle_model, residual_bound=residual_bound,
    )
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

    if not THROTTLE_MODEL_PATH.exists():
        print(f"ERROR: Throttle model not found at {THROTTLE_MODEL_PATH}")
        print("Run train_throttle_v2.py first.")
        sys.exit(1)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    (SAVE_DIR / "checkpoints").mkdir(exist_ok=True)

    total_steps = 3_000_000
    n_envs = 4
    residual_bound = 0.3

    print("=" * 60)
    print("Gimbal Residual PPO v2")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Throttle model: {THROTTLE_MODEL_PATH}")
    print(f"Residual bound: {residual_bound}")
    print(f"Techniques: zero-init, L2 penalty, potential shaping, 4x pos reward")
    print("=" * 60)

    env_kwargs = dict(
        init_altitude_range=(5.0, 60.0),
        init_velocity_range=(-8.0, 0.0),
        init_horizontal_range=(-5.0, 5.0),
        init_tilt_range=(-0.15, 0.15),
        init_fuel_range=(0.6, 1.0),
    )

    train_envs = SubprocVecEnv([
        make_env(i, THROTTLE_MODEL_PATH, residual_bound, **env_kwargs)
        for i in range(n_envs)
    ])
    eval_env = DummyVecEnv([
        make_env(999, THROTTLE_MODEL_PATH, residual_bound, **env_kwargs)
    ])

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
        ent_coef=0.005,    # lower entropy — we want small, precise corrections
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        tensorboard_log=str(SAVE_DIR / "tensorboard"),
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
    )

    # Zero-initialize the residual output layer (Silver et al. 2018)
    zero_init_action_net(model)

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
                        dist = info.get("landing_dist", 99)
                        self.recent_dist.append(dist)
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
            name_prefix="gimbal_res",
        ),
    ]

    notify("gimbal residual v2", f"{total_steps:,} steps, zero-init + L2 + shaping")

    # Baseline eval: how does the pure base controller (zero residuals) perform?
    print("\nBaseline eval (zero residuals):")
    evaluate(model, THROTTLE_MODEL_PATH, 200, residual_bound,
             init_altitude_range=(10., 60.), init_velocity_range=(-8., 0.),
             init_horizontal_range=(-5., 5.), init_tilt_range=(-0.15, 0.15))

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
        ("L0", dict(init_altitude_range=(3., 8.), init_velocity_range=(-2., 0.),
                     init_horizontal_range=(-0.5, 0.5), init_tilt_range=(-0.02, 0.02))),
        ("L3", dict(init_altitude_range=(10., 60.), init_velocity_range=(-8., 0.),
                     init_horizontal_range=(-5., 5.), init_tilt_range=(-0.15, 0.15))),
    ]:
        print(f"  {name}:")
        evaluate(model, THROTTLE_MODEL_PATH, 200, residual_bound, **kw)

    notify("gimbal residual v2 done", "Check results")
    train_envs.close()
    eval_env.close()


if __name__ == "__main__":
    main()
