"""
Throttle-only PPO with curriculum learning for rocket landing.

Key insight: gimbal exploration noise tumbles the rocket, so we fix gimbals
to proportional damping (gimbal = -0.5 * omega) and only learn throttle.
This reduces the action space to 1D and eliminates the stability problem.

Pause/resume: Ctrl+C saves checkpoint. Re-run to resume from last checkpoint.
Pushover notifications on milestones.

Usage:
    cd backend && uv run python scripts/train_landing_rl.py
    cd backend && uv run python scripts/train_landing_rl.py --eval-only
    cd backend && uv run python scripts/train_landing_rl.py --resume
"""

import sys
import os
import signal
import subprocess
import argparse
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


# ── Config ───────────────────────────────────────────────

SAVE_DIR = Path("runs/throttle_ppo")
_shutdown = False


def _signal_handler(signum, frame):
    global _shutdown
    _shutdown = True
    print("\n[!] Shutdown requested, saving checkpoint...")


def notify(title: str, message: str):
    try:
        subprocess.run(
            [os.path.expanduser("~/.claude/scripts/pushover"), message, f"Simulacra: {title}"],
            capture_output=True, timeout=10,
        )
    except Exception:
        pass
    print(f"[NOTIFY] {title}: {message}")


# ── Curriculum levels ────────────────────────────────────

CURRICULUM = [
    {   # L0: Very easy — low altitude, near-stationary, upright
        "name": "L0 (easy)",
        "altitude_range": (3.0, 8.0),
        "velocity_range": (-2.0, 0.0),
        "horizontal_range": (-0.5, 0.5),
        "tilt_range": (-0.02, 0.02),
        "fuel_range": (0.9, 1.0),
        "promote_rate": 0.60,
    },
    {   # L1: Moderate altitude, some velocity
        "name": "L1 (medium)",
        "altitude_range": (5.0, 15.0),
        "velocity_range": (-4.0, 0.0),
        "horizontal_range": (-1.0, 1.0),
        "tilt_range": (-0.05, 0.05),
        "fuel_range": (0.8, 1.0),
        "promote_rate": 0.55,
    },
    {   # L2: Higher, faster, some perturbation
        "name": "L2 (hard)",
        "altitude_range": (8.0, 30.0),
        "velocity_range": (-6.0, 0.0),
        "horizontal_range": (-2.0, 2.0),
        "tilt_range": (-0.10, 0.10),
        "fuel_range": (0.7, 1.0),
        "promote_rate": 0.50,
    },
    {   # L3: Full difficulty
        "name": "L3 (full)",
        "altitude_range": (10.0, 60.0),
        "velocity_range": (-8.0, 0.0),
        "horizontal_range": (-5.0, 5.0),
        "tilt_range": (-0.15, 0.15),
        "fuel_range": (0.6, 1.0),
        "promote_rate": None,  # Final level
    },
]


# ── Wrappers ─────────────────────────────────────────────

class ThrottleOnlyWrapper(gym.Wrapper):
    """
    Reduces action space to 1D (throttle only).
    Gimbals are fixed to proportional damping: gimbal = -gain * omega.
    This is what the expert controller uses and keeps the rocket stable.
    """

    def __init__(self, env, gimbal_gain: float = 0.5, gimbal_clamp: float = 0.3):
        super().__init__(env)
        self.gimbal_gain = gimbal_gain
        self.gimbal_clamp = gimbal_clamp
        # 1D action space: just throttle [0, 1]
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
        )

    def step(self, action):
        throttle = float(action[0])

        # Read angular velocity from last observation
        # obs indices: [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
        obs = self.env._get_obs() if hasattr(self.env, '_get_obs') else self._last_obs
        wy = float(obs[11])
        wz = float(obs[12])

        # Proportional gimbal damping
        gimbal_outer = np.clip(-self.gimbal_gain * wz, -self.gimbal_clamp, self.gimbal_clamp)
        gimbal_inner = np.clip(-self.gimbal_gain * wy, -self.gimbal_clamp, self.gimbal_clamp)

        full_action = np.array([throttle, gimbal_outer, gimbal_inner], dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(full_action)
        self._last_obs = obs
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info


class GlideRewardWrapper(gym.Wrapper):
    """
    Dense reward shaping based on glide path following.

    Reward = alive_bonus
             - glide_error (distance from target descent rate)
             - tilt_penalty
             - horizontal_penalty
             + landing_bonus / crash_penalty
    """

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = self._compute_reward(obs, info, terminated, truncated)
        return obs, reward, terminated, truncated, info

    def _compute_reward(self, obs, info, terminated, truncated):
        alt = float(obs[1])
        vy = float(obs[4])
        vx, vz = float(obs[3]), float(obs[5])
        qw = float(obs[6])
        wx, wy, wz = float(obs[10]), float(obs[11]), float(obs[12])

        tilt = 2.0 * math.acos(min(1.0, abs(qw)))
        omega = math.sqrt(wx**2 + wy**2 + wz**2)
        v_horiz = math.sqrt(vx**2 + vz**2)

        outcome = info.get("outcome")

        # Terminal rewards
        if outcome == "landed":
            vel = info.get("landing_vel", 3.0)
            landing_tilt = math.radians(info.get("landing_tilt", 15.0))
            return 100.0 + 50.0 * max(0, 1 - vel / 3) + 20.0 * max(0, 1 - landing_tilt / 0.26)

        if outcome == "crash":
            return -50.0

        if terminated and outcome is None:
            # Tilt/fuel/altitude termination
            return -30.0

        if truncated:
            # Timeout — penalize remaining altitude (should have descended)
            return -5.0 - 0.1 * alt

        # Dense shaping during flight
        # Target descent rate: glide path profile
        k = 1.5
        if alt > 3.0:
            target_vy = -k * math.sqrt(max(0.1, alt))
        elif alt > 0.5:
            target_vy = max(-k * math.sqrt(alt), -1.0)
        else:
            target_vy = -0.3

        # Glide path error (most important signal)
        glide_error = abs(vy - target_vy)

        return (
            + 0.05                        # alive bonus
            - 0.10 * glide_error          # follow glide path
            - 0.15 * tilt ** 2            # stay upright (quadratic)
            - 0.05 * omega                # angular velocity damping
            - 0.01 * v_horiz              # minimize horizontal drift
            - 0.002 * alt                 # progress toward ground
        )


class CurriculumWrapper(gym.Wrapper):
    """
    Adjusts initial conditions based on curriculum level.
    Tracks success rate and promotes when threshold is met.
    """

    def __init__(self, env, curriculum=CURRICULUM, window=100):
        super().__init__(env)
        self.curriculum = curriculum
        self.level = 0
        self.window = window
        self.outcomes = deque(maxlen=window)
        self.episode_count = 0
        self.total_landings = 0

    @property
    def success_rate(self):
        if len(self.outcomes) < 10:
            return 0.0
        return sum(self.outcomes) / len(self.outcomes)

    @property
    def level_config(self):
        return self.curriculum[self.level]

    def reset(self, **kwargs):
        cfg = self.level_config
        options = {
            "altitude_range": cfg["altitude_range"],
            "velocity_range": cfg["velocity_range"],
            "horizontal_range": cfg["horizontal_range"],
            "tilt_range": cfg["tilt_range"],
            "fuel_range": cfg["fuel_range"],
        }
        if "options" in kwargs:
            kwargs["options"].update(options)
        else:
            kwargs["options"] = options
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            self.episode_count += 1
            landed = info.get("outcome") == "landed"
            self.outcomes.append(1.0 if landed else 0.0)
            if landed:
                self.total_landings += 1

            # Check promotion
            promote_rate = self.level_config.get("promote_rate")
            if (promote_rate is not None
                    and len(self.outcomes) >= self.window
                    and self.success_rate >= promote_rate
                    and self.level < len(self.curriculum) - 1):
                self.level += 1
                self.outcomes.clear()
                info["curriculum_promoted"] = True
                info["new_level"] = self.level

        return obs, reward, terminated, truncated, info


# ── Training ─────────────────────────────────────────────

def make_env(seed: int, level: int = 0):
    """Factory for creating wrapped training environments."""
    def _init():
        from stable_baselines3.common.monitor import Monitor

        cfg = CURRICULUM[level]
        env = RocketLandingEnvV2(
            init_altitude_range=cfg["altitude_range"],
            init_velocity_range=cfg["velocity_range"],
            init_horizontal_range=cfg["horizontal_range"],
            init_tilt_range=cfg["tilt_range"],
            init_fuel_range=cfg["fuel_range"],
        )
        env = ThrottleOnlyWrapper(env)
        env = GlideRewardWrapper(env)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def make_curriculum_env(seed: int):
    """Factory for curriculum env (single env, handles its own level progression)."""
    def _init():
        from stable_baselines3.common.monitor import Monitor

        env = RocketLandingEnvV2()
        env = ThrottleOnlyWrapper(env)
        env = GlideRewardWrapper(env)
        env = CurriculumWrapper(env)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def evaluate(model, n_episodes=100, env_kwargs=None, label=""):
    """Evaluate a trained model."""
    if env_kwargs is None:
        env_kwargs = {}

    env = RocketLandingEnvV2(**env_kwargs)
    env = ThrottleOnlyWrapper(env)

    outcomes = {"landed": 0, "crash": 0, "other": 0}
    landing_vels = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep + 5000)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc

        outcome = info.get("outcome", "other")
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        if outcome == "landed":
            landing_vels.append(info.get("landing_vel", 0))

    env.close()
    rate = outcomes["landed"] / n_episodes
    if label:
        avg_vel = f", avg_vel={np.mean(landing_vels):.2f}m/s" if landing_vels else ""
        print(f"  {label}: {outcomes['landed']}/{n_episodes} ({100*rate:.0f}%) landed, "
              f"{outcomes.get('crash',0)} crash, {outcomes.get('other',0)} other{avg_vel}")
    return rate, outcomes


class TrainingCallback:
    """SB3 callback for notifications, curriculum tracking, and shutdown."""

    def __init__(self, save_dir, n_envs):
        from stable_baselines3.common.callbacks import BaseCallback

        outer = self

        class _CB(BaseCallback):
            def __init__(self, verbose=0):
                super().__init__(verbose)
                self.episode_count = 0
                self.landing_count = 0
                self.recent_outcomes = deque(maxlen=200)
                self.best_rate = 0.0
                self.last_notify_rate = -1.0
                self.first_landing_notified = False
                self.start_time = time.time()
                self.last_log_time = time.time()

            def _on_step(self):
                if _shutdown:
                    ckpt_path = str(save_dir / "checkpoints" / "interrupted")
                    self.model.save(ckpt_path)
                    # Save training state
                    state = {
                        "timesteps": self.num_timesteps,
                        "episode_count": self.episode_count,
                        "landing_count": self.landing_count,
                        "best_rate": self.best_rate,
                    }
                    with open(save_dir / "training_state.json", "w") as f:
                        json.dump(state, f)
                    notify("Paused", f"Saved at {self.num_timesteps:,} steps, "
                           f"{self.landing_count} landings ({100*self.best_rate:.0f}% best)")
                    return False

                for info in self.locals.get("infos", []):
                    if "episode" in info:
                        self.episode_count += 1
                        outcome = info.get("outcome", "other")
                        landed = outcome == "landed"
                        self.recent_outcomes.append(1.0 if landed else 0.0)

                        if landed:
                            self.landing_count += 1

                        # First landing notification
                        if landed and not self.first_landing_notified:
                            self.first_landing_notified = True
                            notify("First landing!",
                                   f"Step {self.num_timesteps:,}, ep {self.episode_count}")

                        # Curriculum promotion
                        if info.get("curriculum_promoted"):
                            new_level = info.get("new_level", "?")
                            notify(f"Promoted to L{new_level}",
                                   f"Step {self.num_timesteps:,}, ep {self.episode_count}")

                        # Periodic rate tracking
                        if len(self.recent_outcomes) >= 50:
                            rate = sum(self.recent_outcomes) / len(self.recent_outcomes)
                            if rate > self.best_rate + 0.05:
                                self.best_rate = rate
                                notify("New best rate",
                                       f"{100*rate:.0f}% at step {self.num_timesteps:,}")

                # Periodic log (every 60s)
                now = time.time()
                if now - self.last_log_time > 60:
                    self.last_log_time = now
                    rate = (sum(self.recent_outcomes) / len(self.recent_outcomes)
                            if self.recent_outcomes else 0)
                    elapsed = now - self.start_time
                    sps = self.num_timesteps / elapsed if elapsed > 0 else 0
                    print(f"  [{elapsed/60:.0f}m] steps={self.num_timesteps:,}, "
                          f"eps={self.episode_count}, landings={self.landing_count}, "
                          f"rate={100*rate:.0f}%, sps={sps:.0f}")

                return True

        self._cls = _CB

    def make(self):
        return self._cls()


def train(total_steps=10_000_000, n_envs=4, resume=False):
    """Main training loop."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    (SAVE_DIR / "checkpoints").mkdir(exist_ok=True)

    # Create environments — each env has its own curriculum tracker
    print(f"Creating {n_envs} training environments with curriculum...")
    train_envs = SubprocVecEnv([make_curriculum_env(seed=i) for i in range(n_envs)])

    # Eval env at full difficulty (L3)
    eval_env = DummyVecEnv([make_env(seed=999, level=3)])

    # Check for checkpoint to resume
    ckpt_path = SAVE_DIR / "checkpoints" / "interrupted.zip"
    if resume and ckpt_path.exists():
        print(f"Resuming from checkpoint...")
        model = PPO.load(
            str(ckpt_path), env=train_envs,
            tensorboard_log=str(SAVE_DIR / "tensorboard"),
        )
        # Load training state
        state_path = SAVE_DIR / "training_state.json"
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            print(f"  Resumed: {state['timesteps']:,} steps, {state['landing_count']} landings")
    else:
        print("Creating new PPO model (1D action = throttle only)...")
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
            ent_coef=0.01,      # Moderate exploration for 1D
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0,
            tensorboard_log=str(SAVE_DIR / "tensorboard"),
            policy_kwargs=dict(
                net_arch=dict(pi=[64, 64], vf=[64, 64]),
            ),
        )

    # Callbacks
    cb = TrainingCallback(SAVE_DIR, n_envs)
    callbacks = [
        cb.make(),
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
            name_prefix="ppo",
        ),
    ]

    print(f"Training for {total_steps:,} steps...")
    print(f"  Tensorboard: tensorboard --logdir={SAVE_DIR / 'tensorboard'}")
    print(f"  Press Ctrl+C to pause and save\n")

    try:
        model.learn(
            total_timesteps=total_steps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        pass

    # Save final model
    model.save(SAVE_DIR / "final_model")
    print(f"\nSaved final model to {SAVE_DIR / 'final_model'}")

    train_envs.close()
    eval_env.close()
    return model


def eval_only():
    """Evaluate all available models."""
    from stable_baselines3 import PPO

    configs = [
        ("L0 Easy (3-8m)", dict(
            init_altitude_range=(3.0, 8.0), init_velocity_range=(-2.0, 0.0),
            init_horizontal_range=(-0.5, 0.5), init_tilt_range=(-0.02, 0.02))),
        ("L1 Medium (5-15m)", dict(
            init_altitude_range=(5.0, 15.0), init_velocity_range=(-4.0, 0.0),
            init_horizontal_range=(-1.0, 1.0), init_tilt_range=(-0.05, 0.05))),
        ("L2 Hard (8-30m)", dict(
            init_altitude_range=(8.0, 30.0), init_velocity_range=(-6.0, 0.0),
            init_horizontal_range=(-2.0, 2.0), init_tilt_range=(-0.10, 0.10))),
        ("L3 Full (10-60m)", dict(
            init_altitude_range=(10.0, 60.0), init_velocity_range=(-8.0, 0.0),
            init_horizontal_range=(-5.0, 5.0), init_tilt_range=(-0.15, 0.15))),
    ]

    # Find best model
    for model_name, model_path in [
        ("best_model", SAVE_DIR / "best_model" / "best_model.zip"),
        ("final_model", SAVE_DIR / "final_model.zip"),
        ("interrupted", SAVE_DIR / "checkpoints" / "interrupted.zip"),
    ]:
        if model_path.exists():
            print(f"\n{'='*60}")
            print(f"Evaluating: {model_name} ({model_path})")
            print(f"{'='*60}")
            model = PPO.load(str(model_path))
            for name, kwargs in configs:
                evaluate(model, n_episodes=200, env_kwargs=kwargs, label=name)


def main():
    parser = argparse.ArgumentParser(description="Throttle-only PPO landing trainer")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate existing models")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--steps", type=int, default=10_000_000, help="Total training steps")
    parser.add_argument("--n-envs", type=int, default=4, help="Parallel environments")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    print("=" * 60)
    print("Throttle-Only PPO with Curriculum")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Save dir: {SAVE_DIR}")
    print("=" * 60)

    if args.eval_only:
        eval_only()
    else:
        notify("Training started", f"Throttle-only PPO, {args.steps:,} steps, {args.n_envs} envs")
        model = train(total_steps=args.steps, n_envs=args.n_envs, resume=args.resume)

        # Final evaluation
        print(f"\n{'='*60}")
        print("FINAL EVALUATION")
        print(f"{'='*60}")
        for name, kwargs in [
            ("L0 Easy", dict(init_altitude_range=(3.0, 8.0), init_velocity_range=(-2.0, 0.0),
                             init_horizontal_range=(-0.5, 0.5), init_tilt_range=(-0.02, 0.02))),
            ("L3 Full", dict(init_altitude_range=(10.0, 60.0), init_velocity_range=(-8.0, 0.0),
                             init_horizontal_range=(-5.0, 5.0), init_tilt_range=(-0.15, 0.15))),
        ]:
            evaluate(model, n_episodes=200, env_kwargs=kwargs, label=name)

        notify("Training complete", "Check results")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
