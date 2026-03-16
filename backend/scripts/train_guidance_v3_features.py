"""
Guidance v3 — Feature Engineering approach.

Hypothesis: the 64×64 MLP fails at 200m because it wastes capacity approximating
elementary functions (sqrt, x², x/y) that are intermediate steps of the optimal
guidance law. By providing these as pre-computed observation features, the network
only needs to learn a near-linear combination — well within 64×64 capacity.

Engineered features (appended to the 13-dim base observation):
  [13] sqrt(y)           — velocity reference: v_desired = -k·√h
  [14] vy² / (2·y)       — required deceleration to stop. THE brake-initiation feature.
  [15] sqrt(x² + z²)     — horizontal distance to pad
  [16] horiz_dist / y    — glide slope tangent (how aggressively to steer)

Total obs: 17 dimensions. Network: 64×64 (same as v1).

Usage:
    cd backend && uv run python scripts/train_guidance_v3_features.py
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

sys.path.insert(0, str(Path(__file__).parent))
from train_guidance import (
    GuidanceWrapper, GuidanceReward, _rotate_body_to_world, _rotate_world_to_body,
    evaluate as _evaluate_raw, notify,
)

SAVE_DIR = Path("runs/guidance_v3_features")
_shutdown = False


def _signal_handler(signum, frame):
    global _shutdown
    _shutdown = True
    print("\n[!] Shutdown requested...")


class PhysicsFeatureWrapper(gym.ObservationWrapper):
    """
    Augments the 13-dim observation with 4 pre-computed physics features.

    These are the intermediate quantities that the optimal guidance law computes,
    which a small ReLU network struggles to approximate from raw state:

      sqrt(y)       — appears in v_desired = -k·sqrt(h)
      vy²/(2y)      — required deceleration to stop at ground level
      horiz_dist    — sqrt(x² + z²), the lateral error magnitude
      horiz_dist/y  — glide slope tangent, measures urgency of lateral correction

    The network receives these "for free" and only needs to learn the policy
    as a function of these derived quantities — a near-linear mapping.
    """

    def __init__(self, env):
        super().__init__(env)
        base_space = env.observation_space
        low = np.concatenate([base_space.low, np.array([-np.inf]*4, dtype=np.float32)])
        high = np.concatenate([base_space.high, np.array([np.inf]*4, dtype=np.float32)])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        x, y, z = float(obs[0]), float(obs[1]), float(obs[2])
        vy = float(obs[4])

        y_safe = max(y, 0.1)
        sqrt_y = math.sqrt(y_safe)
        decel_required = (vy ** 2) / (2.0 * y_safe)   # always positive (vy² >= 0)
        horiz_dist = math.sqrt(x ** 2 + z ** 2)
        glide_slope = horiz_dist / y_safe

        features = np.array([sqrt_y, decel_required, horiz_dist, glide_slope],
                            dtype=np.float32)
        return np.concatenate([obs, features])


def make_env(seed, **env_kwargs):
    def _init():
        from stable_baselines3.common.monitor import Monitor
        env = RocketLandingEnvV2(**env_kwargs)
        env = GuidanceWrapper(env)
        env = PhysicsFeatureWrapper(env)
        env = GuidanceReward(env)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def evaluate(model, n=200, **env_kwargs):
    """Evaluate with the feature wrapper in place."""
    env = PhysicsFeatureWrapper(GuidanceWrapper(RocketLandingEnvV2(**env_kwargs)))
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


# Identical curriculum to v3_bignet for fair comparison
STAGES = [
    {
        "name": "Stage 1: 80m, ±7m",
        "steps": 2_000_000,
        "env_kwargs": dict(
            init_altitude_range=(5.0, 80.0),
            init_velocity_range=(-9.0, 0.0),
            init_horizontal_range=(-7.0, 7.0),
            init_tilt_range=(-0.17, 0.17),
            init_fuel_range=(0.6, 1.0),
        ),
    },
    {
        "name": "Stage 2: 130m, ±12m",
        "steps": 2_500_000,
        "env_kwargs": dict(
            init_altitude_range=(5.0, 130.0),
            init_velocity_range=(-12.0, 0.0),
            init_horizontal_range=(-12.0, 12.0),
            init_tilt_range=(-0.25, 0.25),
            init_fuel_range=(0.5, 1.0),
        ),
    },
    {
        "name": "Stage 3: 200m, ±20m",
        "steps": 3_500_000,
        "env_kwargs": dict(
            init_altitude_range=(5.0, 200.0),
            init_velocity_range=(-15.0, 0.0),
            init_horizontal_range=(-20.0, 20.0),
            init_tilt_range=(-0.3, 0.3),
            init_fuel_range=(0.4, 1.0),
        ),
    },
]

EVAL_SCENARIOS = [
    ("v1 range (5-60m, ±5m)", dict(
        init_altitude_range=(5., 60.), init_velocity_range=(-8., 0.),
        init_horizontal_range=(-5., 5.), init_tilt_range=(-0.15, 0.15))),
    ("Medium (5-130m, ±12m)", dict(
        init_altitude_range=(5., 130.), init_velocity_range=(-12., 0.),
        init_horizontal_range=(-12., 12.), init_tilt_range=(-0.25, 0.25))),
    ("Hard (5-200m, ±20m)", dict(
        init_altitude_range=(5., 200.), init_velocity_range=(-15., 0.),
        init_horizontal_range=(-20., 20.), init_tilt_range=(-0.3, 0.3))),
]


def full_eval(model, label=""):
    if label:
        print(f"\n  {label}")
    for name, kw in EVAL_SCENARIOS:
        print(f"  {name}:")
        evaluate(model, 100, **kw)


def main():
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    (SAVE_DIR / "checkpoints").mkdir(exist_ok=True)

    n_envs = 4

    print("=" * 60)
    print("Guidance v3 — Feature Engineering (64×64)")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Features: sqrt(y), vy²/(2y), horiz_dist, glide_slope")
    print(f"Network: pi=[64, 64], vf=[64, 64]")
    print(f"Stages: {len(STAGES)}")
    print("=" * 60)

    notify("v3 features started", f"{len(STAGES)} stages, 64x64 + 4 physics features")

    # Check for resume
    checkpoints = sorted(SAVE_DIR.glob("checkpoints/v3feat_*_steps.zip"))
    model = None
    resume_stage = 0

    if checkpoints:
        last_cp = checkpoints[-1]
        step_count = int(last_cp.stem.split("_")[-2])
        cumulative = 0
        for i, stage in enumerate(STAGES):
            cumulative += stage["steps"]
            if step_count < cumulative:
                resume_stage = i
                break
        else:
            resume_stage = len(STAGES)
        print(f"  Found checkpoint: {last_cp.name} (stage {resume_stage})")

    for stage_idx, stage in enumerate(STAGES):
        if stage_idx < resume_stage:
            print(f"\n  Skipping {stage['name']} (already completed)")
            continue

        if _shutdown:
            break

        print(f"\n{'='*60}")
        print(f"  {stage['name']}")
        print(f"  Steps: {stage['steps']:,}")
        print(f"{'='*60}")

        train_envs = SubprocVecEnv([
            make_env(i, **stage["env_kwargs"]) for i in range(n_envs)
        ])
        eval_env = DummyVecEnv([
            make_env(999, **stage["env_kwargs"])
        ])

        if model is None:
            if checkpoints:
                print(f"  Resuming from {checkpoints[-1]}")
                model = PPO.load(checkpoints[-1], env=train_envs,
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
                    # Same 64×64 as v1 — the features do the heavy lifting
                    policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
                )
        else:
            model.set_env(train_envs)

        class ProgressCB(BaseCallback):
            def __init__(self, stage_name):
                super().__init__()
                self.stage_name = stage_name
                self.start = time.time()
                self.last_log = time.time()
                self.episodes = 0
                self.landings = 0
                self.recent = deque(maxlen=200)
                self.recent_dist = deque(maxlen=200)

            def _on_step(self):
                if _shutdown:
                    self.model.save(str(SAVE_DIR / "checkpoints" / "interrupted"))
                    notify("Paused", f"{self.stage_name}, step {self.num_timesteps:,}")
                    return False

                for info in self.locals.get("infos", []):
                    if "episode" in info:
                        self.episodes += 1
                        landed = info.get("outcome") == "landed"
                        self.recent.append(1 if landed else 0)
                        if landed:
                            self.landings += 1
                            self.recent_dist.append(info.get("landing_dist", 99))

                if time.time() - self.last_log > 60:
                    self.last_log = time.time()
                    rate = sum(self.recent) / max(1, len(self.recent))
                    avg_dist = np.mean(self.recent_dist) if self.recent_dist else 0
                    elapsed = (time.time() - self.start) / 60
                    print(f"  [{elapsed:.0f}m] steps={self.num_timesteps:,}, "
                          f"rate={100*rate:.0f}%, avg_dist={avg_dist:.2f}m")
                return True

        callbacks = [
            ProgressCB(stage["name"]),
            EvalCallback(
                eval_env,
                best_model_save_path=str(SAVE_DIR / "best_model"),
                log_path=str(SAVE_DIR / f"eval_logs_stage{stage_idx}"),
                eval_freq=10_000 // n_envs,
                n_eval_episodes=20,
                deterministic=True,
                verbose=0,
            ),
            CheckpointCallback(
                save_freq=200_000 // n_envs,
                save_path=str(SAVE_DIR / "checkpoints"),
                name_prefix="v3feat",
            ),
        ]

        try:
            model.learn(
                total_timesteps=stage["steps"],
                callback=callbacks,
                progress_bar=True,
                reset_num_timesteps=False,
            )
        except KeyboardInterrupt:
            break

        model.save(SAVE_DIR / f"stage{stage_idx}_model")

        print(f"\n  --- Eval after {stage['name']} ---")
        full_eval(model)

        stage_rate, stage_dist = evaluate(model, 200, **stage["env_kwargs"])
        notify(f"{stage['name']} done",
               f"{100*stage_rate:.0f}% landed, {stage_dist:.1f}m avg dist")

        train_envs.close()
        eval_env.close()

    # Final eval
    if model is not None:
        model.save(SAVE_DIR / "final_model")
        print(f"\n{'='*60}")
        print("FINAL EVALUATION — v3 Features (64×64 + physics features)")
        print(f"{'='*60}")
        full_eval(model, "All scenarios:")

    notify("v3 features done", "Check results")


if __name__ == "__main__":
    main()
