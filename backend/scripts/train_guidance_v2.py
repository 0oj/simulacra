"""
Guidance PPO v2: fine-tune from v1 best model on expanded range + curriculum.

Strategy:
  B) Load v1 best model, train on wider range (5-200m, ±15m offset)
  C) Curriculum: expand range as success rate improves

Stages:
  1. Current range but with some harder episodes mixed in (warmup)
  2. Full expanded range (5-200m altitude, ±15m horizontal, -15 m/s velocity)
  3. Extreme range if stage 2 succeeds (5-300m, ±25m, -20 m/s)

Usage:
    cd backend && uv run python scripts/train_guidance_v2.py
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

# Import from train_guidance.py (same directory)
sys.path.insert(0, str(Path(__file__).parent))
from train_guidance import (
    GuidanceWrapper, GuidanceReward, _rotate_body_to_world, _rotate_world_to_body,
    evaluate, notify,
)

SAVE_DIR = Path("runs/guidance_v2")
V1_MODEL = Path("runs/guidance_ppo/best_model/best_model.zip")
_shutdown = False


def _signal_handler(signum, frame):
    global _shutdown
    _shutdown = True
    print("\n[!] Shutdown requested...")


STAGES = [
    {
        "name": "Stage 1: 80m, ±7m",
        "steps": 1_000_000,
        "env_kwargs": dict(
            init_altitude_range=(5.0, 80.0),
            init_velocity_range=(-9.0, 0.0),
            init_horizontal_range=(-7.0, 7.0),
            init_tilt_range=(-0.17, 0.17),
            init_fuel_range=(0.6, 1.0),
        ),
    },
    {
        "name": "Stage 2: 100m, ±10m",
        "steps": 1_000_000,
        "env_kwargs": dict(
            init_altitude_range=(5.0, 100.0),
            init_velocity_range=(-10.0, 0.0),
            init_horizontal_range=(-10.0, 10.0),
            init_tilt_range=(-0.2, 0.2),
            init_fuel_range=(0.5, 1.0),
        ),
    },
    {
        "name": "Stage 3: 130m, ±12m",
        "steps": 1_500_000,
        "env_kwargs": dict(
            init_altitude_range=(5.0, 130.0),
            init_velocity_range=(-12.0, 0.0),
            init_horizontal_range=(-12.0, 12.0),
            init_tilt_range=(-0.25, 0.25),
            init_fuel_range=(0.5, 1.0),
        ),
    },
    {
        "name": "Stage 4: 170m, ±15m",
        "steps": 1_500_000,
        "env_kwargs": dict(
            init_altitude_range=(5.0, 170.0),
            init_velocity_range=(-14.0, 0.0),
            init_horizontal_range=(-15.0, 15.0),
            init_tilt_range=(-0.3, 0.3),
            init_fuel_range=(0.4, 1.0),
        ),
    },
    {
        "name": "Stage 5: 200m, ±20m",
        "steps": 2_000_000,
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
    ("Medium (5-100m, ±10m)", dict(
        init_altitude_range=(5., 100.), init_velocity_range=(-10., 0.),
        init_horizontal_range=(-10., 10.), init_tilt_range=(-0.2, 0.2))),
    ("Hard (5-200m, ±15m)", dict(
        init_altitude_range=(5., 200.), init_velocity_range=(-15., 0.),
        init_horizontal_range=(-15., 15.), init_tilt_range=(-0.3, 0.3))),
]


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


def full_eval(model, label=""):
    if label:
        print(f"\n  {label}")
    for name, kw in EVAL_SCENARIOS:
        rate, dist = evaluate(model, 100, **kw)


def main():
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    if not V1_MODEL.exists():
        print(f"ERROR: v1 model not found at {V1_MODEL}")
        print("Run train_guidance.py first.")
        sys.exit(1)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    (SAVE_DIR / "checkpoints").mkdir(exist_ok=True)

    n_envs = 4

    print("=" * 60)
    print("Guidance PPO v2 — curriculum fine-tune")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Base model: {V1_MODEL}")
    print(f"Stages: {len(STAGES)}")
    print("=" * 60)

    notify("guidance v2 started", f"{len(STAGES)} stages, fine-tune from v1")

    # Check for resume
    checkpoints = sorted(SAVE_DIR.glob("checkpoints/guidance_v2_*_steps.zip"))
    model = None
    resume_stage = 0

    if checkpoints:
        # Find which stage we're in based on step count
        last_cp = checkpoints[-1]
        step_count = int(last_cp.stem.split("_")[-2])
        cumulative = 0
        for i, stage in enumerate(STAGES):
            cumulative += stage["steps"]
            if step_count < cumulative:
                resume_stage = i
                break
        else:
            resume_stage = len(STAGES)  # all done

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
        print(f"  Alt: {stage['env_kwargs']['init_altitude_range']}")
        print(f"  Horiz: {stage['env_kwargs']['init_horizontal_range']}")
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
                print(f"  Loading v1 model: {V1_MODEL}")
                model = PPO.load(V1_MODEL, env=train_envs,
                                 tensorboard_log=str(SAVE_DIR / "tensorboard"))
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
                name_prefix="guidance_v2",
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

        # Eval after each stage
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
        print("FINAL EVALUATION")
        print(f"{'='*60}")
        full_eval(model, "All scenarios:")

    notify("guidance v2 done", "Check results")


if __name__ == "__main__":
    main()
