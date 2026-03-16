"""
Residual gimbal learning: learn small corrections on top of proportional damping.

Action space: [throttle (0-1), gimbal_residual_outer (-1,1), gimbal_residual_inner (-1,1)]
Actual gimbal = proportional_damping(omega) + residual * bound

Starts with bound=0.1, increases as agent improves.
Uses SAC for stability.

Usage:
    cd backend && uv run python scripts/train_residual_gimbal.py
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

SAVE_DIR = Path("runs/residual_gimbal")
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


class ResidualGimbalWrapper(gym.Wrapper):
    """
    3D action space: [throttle, gimbal_residual_outer, gimbal_residual_inner]

    Baseline = torque-based attitude controller (LandingController-style).
    Actual gimbal = baseline + residual * bound
    The residual is bounded so exploration can't tumble the rocket.
    """

    ATT_KP = 500.0
    ATT_KD = 200.0
    MAX_THRUST = 2000.0
    MOMENT_ARM = 2.0
    MAX_GIMBAL = 0.26

    def __init__(self, env, residual_bound=0.1):
        super().__init__(env)
        self.residual_bound = residual_bound

        # 3D action: throttle + 2 gimbal residuals
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )

    def step(self, action):
        throttle = float(action[0])
        res_outer = float(action[1]) * self.residual_bound
        res_inner = float(action[2]) * self.residual_bound

        obs = self._last_obs
        qw, qx, qy, qz = float(obs[6]), float(obs[7]), float(obs[8]), float(obs[9])
        wx, wy, wz = float(obs[10]), float(obs[11]), float(obs[12])

        # Torque-based attitude baseline (LandingController-style)
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

        # Baseline normalized + learned residual
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

        nose = _rotate_body_to_world(qw, float(obs[7]), float(obs[8]), float(obs[9]), [1, 0, 0])
        tilt = math.acos(max(-1.0, min(1.0, nose[1])))
        omega = math.sqrt(wx**2 + wy**2 + wz**2)
        v_horiz = math.sqrt(vx**2 + vz**2)

        outcome = info.get("outcome")

        if outcome == "landed":
            vel = info.get("landing_vel", 3.0)
            lt = math.radians(info.get("landing_tilt", 15.0))
            # Extra bonus for position accuracy (residual gimbals can steer)
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

        # Also reward reducing horizontal distance (gimbals can help)
        horiz_dist = math.sqrt(float(obs[0])**2 + float(obs[2])**2)

        return (
            -0.005 * alt              # pressure to descend (no alive bonus)
            + descent_reward
            - 0.5 * excess ** 2
            - 0.10 * tilt ** 2
            - 0.03 * omega
            - 0.02 * v_horiz
            - 0.01 * horiz_dist
        )


def make_env(seed, residual_bound=0.1, **env_kwargs):
    def _init():
        from stable_baselines3.common.monitor import Monitor
        env = RocketLandingEnvV2(**env_kwargs)
        env = ResidualGimbalWrapper(env, residual_bound=residual_bound)
        env = LandingRewardWrapper(env)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def evaluate(model, n=200, residual_bound=0.1, **env_kwargs):
    env = ResidualGimbalWrapper(RocketLandingEnvV2(**env_kwargs), residual_bound=residual_bound)
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

    # Phase schedule: increase residual bound over training
    phases = [
        (1_000_000, 0.1, "Phase 1: bound=0.1"),
        (1_000_000, 0.2, "Phase 2: bound=0.2"),
        (1_000_000, 0.4, "Phase 3: bound=0.4"),
    ]

    print("=" * 60)
    print("Residual Gimbal SAC")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)

    notify("Residual gimbal started", "3 phases, increasing bound")

    env_kwargs = dict(
        init_altitude_range=(5.0, 60.0),
        init_velocity_range=(-8.0, 0.0),
        init_horizontal_range=(-5.0, 5.0),
        init_tilt_range=(-0.15, 0.15),
        init_fuel_range=(0.6, 1.0),
    )

    from stable_baselines3.common.vec_env import VecNormalize

    model = None

    for phase_steps, bound, phase_name in phases:
        if _shutdown:
            break

        print(f"\n{'='*60}")
        print(f"{phase_name}")
        print(f"{'='*60}")

        train_env = DummyVecEnv([make_env(0, residual_bound=bound, **env_kwargs)])
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True,
                                 clip_obs=10.0, clip_reward=10.0, gamma=0.99)
        eval_env = DummyVecEnv([make_env(999, residual_bound=bound, **env_kwargs)])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                                clip_obs=10.0, training=False)

        if model is None:
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
        else:
            model.set_env(train_env)

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
                log_path=str(SAVE_DIR / f"eval_logs_bound{bound}"),
                eval_freq=5_000,
                n_eval_episodes=20,
                deterministic=True,
                verbose=0,
            ),
            CheckpointCallback(
                save_freq=100_000,
                save_path=str(SAVE_DIR / "checkpoints"),
                name_prefix=f"sac_b{bound}",
            ),
        ]

        try:
            model.learn(total_timesteps=phase_steps, callback=callbacks,
                        progress_bar=True, reset_num_timesteps=False)
        except KeyboardInterrupt:
            break

        # Eval this phase
        rate = evaluate(model, 200, residual_bound=bound,
                        init_altitude_range=(10., 60.), init_velocity_range=(-8., 0.),
                        init_horizontal_range=(-5., 5.), init_tilt_range=(-0.15, 0.15))
        print(f"  L3 landing rate: {100*rate:.0f}%")
        notify(f"{phase_name} done", f"L3: {100*rate:.0f}%")

        train_env.close()
        eval_env.close()

    model.save(SAVE_DIR / "final_model")

    # Final eval at highest bound
    final_bound = phases[-1][1]
    print(f"\n{'='*60}")
    print(f"FINAL EVAL (bound={final_bound})")
    for name, kw in [
        ("L0", dict(init_altitude_range=(3., 8.), init_velocity_range=(-2., 0.),
                     init_horizontal_range=(-0.5, 0.5), init_tilt_range=(-0.02, 0.02))),
        ("L3", dict(init_altitude_range=(10., 60.), init_velocity_range=(-8., 0.),
                     init_horizontal_range=(-5., 5.), init_tilt_range=(-0.15, 0.15))),
    ]:
        rate = evaluate(model, 200, residual_bound=final_bound, **kw)
        print(f"  {name}: {100*rate:.0f}%")

    notify("Residual gimbal done", "Check results")


if __name__ == "__main__":
    main()
