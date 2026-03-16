"""
Throttle-only PPO v2: fixes the two root causes of failure.

Root cause analysis from flight log diagnostics:
1. GIMBAL BUG: proportional damping (gimbal = -K*omega) only dampens angular
   velocity but never corrects tilt back to vertical. A 6° initial tilt =
   permanent horizontal drift → hover until fuel exhaustion (68% of episodes).
2. REWARD BUG: +0.02 alive bonus means hovering collects +16 reward over 800
   steps, which exceeds the -50 truncation penalty. Hovering is rewarded.

Fixes:
1. Add tilt correction to gimbal: gimbal = -K_d*omega - K_p*tilt_error
   This actively restores the rocket to vertical, eliminating horizontal drift.
2. Remove alive bonus, add per-step penalty proportional to altitude.
   Hovering at any altitude is now costly.

Usage:
    cd backend && uv run python scripts/train_throttle_v2.py
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
import gymnasium as gym
from gymnasium import spaces

from simulacra.gym_env import RocketLandingEnvV2

SAVE_DIR = Path("runs/throttle_v2")
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
    """Rotate vector from body frame to world frame using quaternion."""
    qv = np.array([qx, qy, qz])
    vn = np.array(v, dtype=float)
    cross1 = np.cross(qv, vn)
    cross2 = np.cross(qv, cross1)
    return vn + 2.0 * qw * cross1 + 2.0 * cross2


def _rotate_world_to_body(qw, qx, qy, qz, v):
    """Rotate vector from world frame to body frame using conjugate quaternion."""
    qv = np.array([-qx, -qy, -qz])
    vn = np.array(v, dtype=float)
    cross1 = np.cross(qv, vn)
    cross2 = np.cross(qv, cross1)
    return vn + 2.0 * qw * cross1 + 2.0 * cross2


class ThrottleOnlyV2Wrapper(gym.Wrapper):
    """
    1D action (throttle only) with torque-based gimbal attitude controller.

    Replicates the LandingController's proven attitude law:
      error = cross(nose_world, desired_world)  → body frame
      tau = Kp * error - Kd * omega
      gimbal = -tau / (thrust * moment_arm)

    This is stable because it accounts for the thrust-torque dynamics,
    unlike simple gain-based approaches that saturate and diverge.
    """

    ATT_KP = 500.0       # attitude proportional gain (N·m / rad)
    ATT_KD = 200.0       # attitude derivative gain (N·m·s / rad)
    MAX_THRUST = 2000.0
    MOMENT_ARM = 2.0
    MAX_GIMBAL = 0.26     # rad

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
        )

    def step(self, action):
        throttle = float(action[0])
        obs = self._last_obs

        # obs: [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
        qw, qx, qy, qz = float(obs[6]), float(obs[7]), float(obs[8]), float(obs[9])
        wx, wy, wz = float(obs[10]), float(obs[11]), float(obs[12])

        # Torque-based attitude control (same as LandingController)
        nose_world = _rotate_body_to_world(qw, qx, qy, qz, [1, 0, 0])
        desired = np.array([0, 1, 0])
        error_world = np.cross(nose_world, desired)
        error_body = _rotate_world_to_body(qw, qx, qy, qz, error_world)
        omega = np.array([wx, wy, wz])

        tau = self.ATT_KP * error_body - self.ATT_KD * omega

        thrust_n = throttle * self.MAX_THRUST
        if thrust_n > 100:
            scale = 1.0 / (thrust_n * self.MOMENT_ARM)
            outer_cmd = float(np.clip(-tau[2] * scale, -self.MAX_GIMBAL, self.MAX_GIMBAL))
            inner_cmd = float(np.clip(-tau[1] * scale, -self.MAX_GIMBAL, self.MAX_GIMBAL))
        else:
            outer_cmd = 0.0
            inner_cmd = 0.0

        # Normalize to [-1, 1] action space
        outer_norm = outer_cmd / self.MAX_GIMBAL
        inner_norm = inner_cmd / self.MAX_GIMBAL

        full = np.array([throttle, outer_norm, inner_norm], dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(full)
        self._last_obs = obs
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info


class LandingRewardV2(gym.Wrapper):
    """
    Improved reward function:
    - No alive bonus (hovering is not rewarded)
    - Per-step altitude penalty (pressure to descend)
    - Stronger descent incentive near ground
    - Large terminal rewards (+200/-200)
    """

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

        # Compute tilt from nose direction
        nose = _rotate_body_to_world(
            float(obs[6]), float(obs[7]), float(obs[8]), float(obs[9]),
            [1, 0, 0],
        )
        tilt = math.acos(max(-1.0, min(1.0, nose[1])))
        omega = math.sqrt(wx**2 + wy**2 + wz**2)
        v_horiz = math.sqrt(vx**2 + vz**2)

        outcome = info.get("outcome")

        if outcome == "landed":
            vel = info.get("landing_vel", 3.0)
            lt = math.radians(info.get("landing_tilt", 15.0))
            dist = info.get("landing_dist", 10.0)
            return (200.0
                    + 50.0 * max(0, 1 - vel / 3)
                    + 20.0 * max(0, 1 - lt / 0.26)
                    + 30.0 * max(0, 1 - dist / 5.0))

        if outcome == "crash":
            return -200.0

        if terminated and outcome is None:
            return -100.0

        if truncated:
            return -50.0 - 0.5 * alt

        # Dense shaping — NO alive bonus
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

        # Horizontal distance penalty (stronger than v1)
        horiz_dist = math.sqrt(float(obs[0])**2 + float(obs[2])**2)

        return (
            -0.005 * alt              # pressure to descend (no alive bonus!)
            + descent_reward
            - 0.5 * excess ** 2
            - 0.10 * tilt ** 2
            - 0.03 * omega
            - 0.02 * v_horiz
            - 0.01 * horiz_dist
        )


def make_env(seed, **env_kwargs):
    def _init():
        from stable_baselines3.common.monitor import Monitor
        env = RocketLandingEnvV2(**env_kwargs)
        env = ThrottleOnlyV2Wrapper(env)
        env = LandingRewardV2(env)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def evaluate(model, n=200, **env_kwargs):
    env = ThrottleOnlyV2Wrapper(RocketLandingEnvV2(**env_kwargs))
    landed, crashed, fuel_out, timeout = 0, 0, 0, 0
    for ep in range(n):
        obs, _ = env.reset(seed=ep + 5000)
        done = False
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, t, tr, info = env.step(action)
            done = t or tr
            steps += 1
        outcome = info.get("outcome")
        if outcome == "landed":
            landed += 1
        elif outcome == "crash":
            crashed += 1
        elif steps >= 1000:
            timeout += 1
        else:
            fuel_out += 1
    env.close()
    rate = landed / n
    print(f"    {landed}/{n} landed ({100*rate:.0f}%), {crashed} crash, {fuel_out} fuel, {timeout} timeout")
    return rate


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
    print("Throttle-Only PPO v2 (fixed gimbal + reward)")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Fixes: tilt correction, velocity damping, no alive bonus")
    print("=" * 60)

    # Train on L3 directly — with the gimbal fix, should be learnable
    env_kwargs = dict(
        init_altitude_range=(5.0, 60.0),
        init_velocity_range=(-8.0, 0.0),
        init_horizontal_range=(-5.0, 5.0),
        init_tilt_range=(-0.15, 0.15),
        init_fuel_range=(0.6, 1.0),
    )

    train_envs = SubprocVecEnv([make_env(i, **env_kwargs) for i in range(n_envs)])
    eval_env = DummyVecEnv([make_env(999, **env_kwargs)])

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
                        if self.landings == 1:
                            notify("First landing!", f"Step {self.num_timesteps:,}")

            if time.time() - self.last_log > 60:
                self.last_log = time.time()
                rate = sum(self.recent) / max(1, len(self.recent))
                elapsed = (time.time() - self.start) / 60
                print(f"  [{elapsed:.0f}m] steps={self.num_timesteps:,}, "
                      f"landings={self.landings}, rate={100*rate:.0f}%")
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
            name_prefix="ppo_v2",
        ),
    ]

    notify("v2 started", f"{total_steps:,} steps, fixed gimbal + reward")

    try:
        model.learn(total_timesteps=total_steps, callback=callbacks, progress_bar=True)
    except KeyboardInterrupt:
        pass

    model.save(SAVE_DIR / "final_model")

    # Final eval
    print(f"\n{'='*60}")
    print("FINAL EVALUATION")
    for name, kw in [
        ("L0", dict(init_altitude_range=(3., 8.), init_velocity_range=(-2., 0.),
                     init_horizontal_range=(-0.5, 0.5), init_tilt_range=(-0.02, 0.02))),
        ("L3", dict(init_altitude_range=(10., 60.), init_velocity_range=(-8., 0.),
                     init_horizontal_range=(-5., 5.), init_tilt_range=(-0.15, 0.15))),
    ]:
        print(f"  {name}:")
        evaluate(model, 200, **kw)

    notify("v2 done", "Check results")
    train_envs.close()
    eval_env.close()


if __name__ == "__main__":
    main()
