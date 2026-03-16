"""
High-Altitude Training (500m) — GPU-targeted PPO with 5-stage curriculum.

Combines all findings from the feature ablation study and DR experiments:
  - Variant A features (sqrt_y, brake_margin, vel_error, glide_slope) — proven best
  - 256×256 network — GPU makes this affordable, higher-alt needs more capacity
  - Potential-based altitude reward — avoids massive negative baseline at 500m
  - 5-stage curriculum: 80m → 150m → 300m → 500m → 500m+DR
  - Episode length scales with altitude (1000 → 1500 → 2000 steps)

Designed for UCL RTX 3090 Ti (mallard-l). ~20M total steps, expect 6-10 hours.

Usage:
    cd backend
    uv run python scripts/train_v4_highalt.py

Results saved to runs/v4_highalt/
"""

import sys
import os
import json
import signal
import math
import time
from pathlib import Path
from datetime import datetime
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from simulacra.gym_env import RocketLandingEnvV2, DomainRandomization

from train_guidance import (
    GuidanceWrapper, GuidanceReward, _rotate_body_to_world,
    notify,
)
from train_feature_ablation import (
    compute_all_features, FeatureWrapper,
)


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

# Variant A: proven winner from ablation study
FEATURE_SET = ["sqrt_y", "brake_margin", "vel_error", "glide_slope"]

# 256×256 — GPU makes this affordable, high-altitude needs capacity
NETWORK = dict(pi=[256, 256], vf=[256, 256])

# PPO hyperparameters
PPO_KWARGS = dict(
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,       # larger batch for GPU throughput
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
)

N_ENVS = 8  # GPU can handle more parallel envs

# Fixed max_altitude for ALL stages — ensures consistent observation space bounds
# across the curriculum. Without this, each stage auto-derives different max_altitude
# from init_altitude_range, causing SB3 to reject set_env() at stage transitions.
MAX_ALTITUDE = 750.0  # 500m * 1.5 headroom

# ──────────────────────────────────────────────────────────────
# Domain randomization schedules
# ──────────────────────────────────────────────────────────────

DR_MILD = DomainRandomization(
    dry_mass_scale=(0.95, 1.05),       # ±5%
    fuel_mass_scale=(0.95, 1.05),
    max_thrust_scale=(0.97, 1.03),     # ±3%
    burn_rate_scale=(0.95, 1.05),
    gimbal_rate_scale=(0.9, 1.1),      # ±10%
    gimbal_tau_scale=(0.85, 1.15),     # ±15%
    wind_speed_range=(0.0, 1.0),       # 0-1 m/s
)

DR_FULL = DomainRandomization(
    dry_mass_scale=(0.85, 1.15),       # ±15%
    fuel_mass_scale=(0.85, 1.15),
    max_thrust_scale=(0.92, 1.08),     # ±8%
    burn_rate_scale=(0.85, 1.15),
    length_scale=(0.95, 1.05),         # ±5%
    gimbal_rate_scale=(0.8, 1.2),      # ±20%
    gimbal_tau_scale=(0.7, 1.3),       # ±30%
    wind_speed_range=(0.0, 3.0),       # 0-3 m/s
    gravity_scale=(0.995, 1.005),      # ±0.5%
    air_density_scale=(0.9, 1.1),      # ±10%
    gps_noise_std=(0.0, 1.5),          # 0-1.5m
    baro_noise_std=(0.0, 0.5),         # 0-0.5m
    velocity_noise_std=(0.0, 0.3),     # 0-0.3 m/s
    orientation_noise_std=(0.0, 0.02), # 0-1.1°
    gyro_noise_std=(0.0, 0.01),        # 0-0.01 rad/s
    gyro_bias_range=(-0.005, 0.005),   # ±0.005 rad/s
)

# ──────────────────────────────────────────────────────────────
# 5-stage curriculum
# ──────────────────────────────────────────────────────────────

STAGES = [
    {
        "name": "Stage 1: 80m, nominal",
        "steps": 2_000_000,
        "max_episode_steps": 1000,
        "dr": None,
        "env_kwargs": dict(
            max_altitude=MAX_ALTITUDE,
            init_altitude_range=(5.0, 80.0),
            init_velocity_range=(-9.0, 0.0),
            init_horizontal_range=(-7.0, 7.0),
            init_tilt_range=(-0.17, 0.17),
            init_fuel_range=(0.6, 1.0),
        ),
    },
    {
        "name": "Stage 2: 150m, nominal",
        "steps": 3_000_000,
        "max_episode_steps": 1000,
        "dr": None,
        "env_kwargs": dict(
            max_altitude=MAX_ALTITUDE,
            init_altitude_range=(5.0, 150.0),
            init_velocity_range=(-15.0, 0.0),
            init_horizontal_range=(-15.0, 15.0),
            init_tilt_range=(-0.25, 0.25),
            init_fuel_range=(0.5, 1.0),
        ),
    },
    {
        "name": "Stage 3: 300m, mild DR",
        "steps": 4_000_000,
        "max_episode_steps": 1500,
        "dr": DR_MILD,
        "env_kwargs": dict(
            max_altitude=MAX_ALTITUDE,
            init_altitude_range=(5.0, 300.0),
            init_velocity_range=(-25.0, 0.0),
            init_horizontal_range=(-25.0, 25.0),
            init_tilt_range=(-0.30, 0.30),
            init_fuel_range=(0.5, 1.0),
        ),
    },
    {
        "name": "Stage 4: 500m, mild DR",
        "steps": 5_000_000,
        "max_episode_steps": 2000,
        "dr": DR_MILD,
        "env_kwargs": dict(
            max_altitude=MAX_ALTITUDE,
            init_altitude_range=(5.0, 500.0),
            init_velocity_range=(-35.0, 0.0),
            init_horizontal_range=(-40.0, 40.0),
            init_tilt_range=(-0.35, 0.35),
            init_fuel_range=(0.6, 1.0),
        ),
    },
    {
        "name": "Stage 5: 500m, full DR",
        "steps": 6_000_000,
        "max_episode_steps": 2000,
        "dr": DR_FULL,
        "env_kwargs": dict(
            max_altitude=MAX_ALTITUDE,
            init_altitude_range=(5.0, 500.0),
            init_velocity_range=(-35.0, 0.0),
            init_horizontal_range=(-40.0, 40.0),
            init_tilt_range=(-0.35, 0.35),
            init_fuel_range=(0.5, 1.0),
        ),
    },
]

EVAL_SCENARIOS = [
    ("easy_100m", dict(
        max_altitude=MAX_ALTITUDE,
        init_altitude_range=(5., 100.), init_velocity_range=(-10., 0.),
        init_horizontal_range=(-10., 10.), init_tilt_range=(-0.17, 0.17))),
    ("medium_200m", dict(
        max_altitude=MAX_ALTITUDE,
        init_altitude_range=(5., 200.), init_velocity_range=(-15., 0.),
        init_horizontal_range=(-20., 20.), init_tilt_range=(-0.25, 0.25))),
    ("hard_500m", dict(
        max_altitude=MAX_ALTITUDE,
        init_altitude_range=(5., 500.), init_velocity_range=(-35., 0.),
        init_horizontal_range=(-40., 40.), init_tilt_range=(-0.35, 0.35))),
    ("hard_500m_dr", dict(
        max_altitude=MAX_ALTITUDE,
        init_altitude_range=(5., 500.), init_velocity_range=(-35., 0.),
        init_horizontal_range=(-40., 40.), init_tilt_range=(-0.35, 0.35))),
]


_shutdown = False


def _signal_handler(signum, frame):
    global _shutdown
    _shutdown = True
    print("\n[!] Shutdown requested...")


def make_env(seed, feature_names, dr=None, max_episode_steps=1000, **env_kwargs):
    def _init():
        from stable_baselines3.common.monitor import Monitor
        env = RocketLandingEnvV2(
            max_episode_steps=max_episode_steps,
            domain_randomization=dr,
            **env_kwargs,
        )
        env = GuidanceWrapper(env)
        env = FeatureWrapper(env, feature_names)
        env = GuidanceReward(env)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def evaluate(model, feature_names, n=100, dr=None, max_episode_steps=2000, **env_kwargs):
    """Evaluate with features and optional DR."""
    env = FeatureWrapper(
        GuidanceWrapper(RocketLandingEnvV2(
            max_episode_steps=max_episode_steps,
            domain_randomization=dr,
            **env_kwargs,
        )),
        feature_names,
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
    return rate, avg_dist, landed


def full_eval(model, feature_names, save_dir, stage_name):
    """Run all eval scenarios and save structured results."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "stage": stage_name,
        "scenarios": {},
    }
    for scenario_name, kw in EVAL_SCENARIOS:
        dr = DR_FULL if scenario_name.endswith("_dr") else None
        rate, avg_dist, landed = evaluate(model, feature_names, 100, dr=dr, **kw)
        results["scenarios"][scenario_name] = {
            "landing_rate": rate,
            "avg_dist_m": round(avg_dist, 2),
            "landed": landed,
            "total": 100,
        }
        print(f"    {scenario_name}: {landed}/100 ({100*rate:.0f}%), avg dist={avg_dist:.2f}m")

    results_file = save_dir / "results.jsonl"
    with open(results_file, "a") as f:
        f.write(json.dumps(results) + "\n")

    return results


def main():
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback

    save_dir = Path("runs/v4_highalt")
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "checkpoints").mkdir(exist_ok=True)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    feature_names = FEATURE_SET
    n_features = len(feature_names)

    # Save experiment config
    config = {
        "name": "v4_highalt",
        "description": "High-altitude (500m) training with 5-stage curriculum, 256x256 net, DR",
        "features": feature_names,
        "n_features": n_features,
        "total_obs_dims": 13 + n_features,
        "network": NETWORK,
        "n_envs": N_ENVS,
        "ppo": PPO_KWARGS,
        "dr_mild": {k: v for k, v in DR_MILD.__dict__.items()},
        "dr_full": {k: v for k, v in DR_FULL.__dict__.items()},
        "stages": [
            {"name": s["name"], "steps": s["steps"],
             "max_episode_steps": s["max_episode_steps"],
             "has_dr": s["dr"] is not None}
            for s in STAGES
        ],
        "total_steps": sum(s["steps"] for s in STAGES),
        "started": datetime.now().isoformat(),
    }
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    total_steps = sum(s["steps"] for s in STAGES)
    print("=" * 60)
    print("v4 High-Altitude Training (500m)")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Features ({n_features}): {', '.join(feature_names)}")
    print(f"Obs dims: {13 + n_features}")
    print(f"Network: {NETWORK}")
    print(f"Total steps: {total_steps:,} across {len(STAGES)} stages")
    print(f"Envs: {N_ENVS}")
    print("=" * 60)

    notify("v4 highalt started", f"{total_steps/1e6:.0f}M steps, 500m, 256x256")

    # Check for resume
    checkpoints = sorted(save_dir.glob("checkpoints/v4_*_steps.zip"))
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
        print(f"  Found checkpoint: {last_cp.name} (resuming from stage {resume_stage})")

    for stage_idx, stage in enumerate(STAGES):
        if stage_idx < resume_stage:
            print(f"\n  Skipping {stage['name']} (already completed)")
            continue

        if _shutdown:
            break

        dr = stage["dr"]
        max_ep_steps = stage["max_episode_steps"]
        print(f"\n{'='*60}")
        print(f"  {stage['name']}")
        print(f"  Steps: {stage['steps']:,}, Episode len: {max_ep_steps}")
        print(f"  DR: {'Yes' if dr else 'No (nominal)'}")
        print(f"{'='*60}")

        train_envs = SubprocVecEnv([
            make_env(i, feature_names, dr=dr, max_episode_steps=max_ep_steps,
                     **stage["env_kwargs"])
            for i in range(N_ENVS)
        ])
        eval_env = DummyVecEnv([
            make_env(999, feature_names, dr=None, max_episode_steps=max_ep_steps,
                     **stage["env_kwargs"])
        ])

        if model is None:
            if checkpoints:
                print(f"  Resuming from {checkpoints[-1]}")
                model = PPO.load(checkpoints[-1], env=train_envs,
                                 tensorboard_log=str(save_dir / "tensorboard"))
            else:
                model = PPO(
                    "MlpPolicy",
                    train_envs,
                    verbose=0,
                    tensorboard_log=str(save_dir / "tensorboard"),
                    policy_kwargs=dict(net_arch=NETWORK),
                    **PPO_KWARGS,
                )
        else:
            model.set_env(train_envs)

        class ProgressCB(BaseCallback):
            def __init__(self, stage_name, save_dir):
                super().__init__()
                self.stage_name = stage_name
                self.save_dir = save_dir
                self.start = time.time()
                self.last_log = time.time()
                self.episodes = 0
                self.landings = 0
                self.recent = deque(maxlen=200)
                self.recent_dist = deque(maxlen=200)
                self.training_log = []

            def _on_step(self):
                if _shutdown:
                    self.model.save(str(self.save_dir / "checkpoints" / "interrupted"))
                    notify("Paused", f"v4: {self.stage_name}")
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
                    avg_dist = float(np.mean(self.recent_dist)) if self.recent_dist else 0
                    elapsed = (time.time() - self.start) / 60

                    entry = {
                        "elapsed_min": round(elapsed, 1),
                        "steps": self.num_timesteps,
                        "stage": self.stage_name,
                        "landing_rate": round(rate, 3),
                        "avg_dist": round(avg_dist, 2),
                        "episodes": self.episodes,
                    }
                    self.training_log.append(entry)

                    print(f"  [{elapsed:.0f}m] steps={self.num_timesteps:,}, "
                          f"rate={100*rate:.0f}%, avg_dist={avg_dist:.2f}m")
                return True

            def _on_training_end(self):
                log_file = self.save_dir / f"training_log_stage{stage_idx}.jsonl"
                with open(log_file, "w") as f:
                    for entry in self.training_log:
                        f.write(json.dumps(entry) + "\n")

        progress_cb = ProgressCB(stage["name"], save_dir)

        callbacks = [
            progress_cb,
            EvalCallback(
                eval_env,
                best_model_save_path=str(save_dir / "best_model"),
                log_path=str(save_dir / f"eval_logs_stage{stage_idx}"),
                eval_freq=10_000 // N_ENVS,
                n_eval_episodes=20,
                deterministic=True,
                verbose=0,
            ),
            CheckpointCallback(
                save_freq=200_000 // N_ENVS,
                save_path=str(save_dir / "checkpoints"),
                name_prefix="v4",
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

        model.save(save_dir / f"stage{stage_idx}_model")

        # Structured evaluation
        print(f"\n  --- Eval after {stage['name']} ---")
        results = full_eval(model, feature_names, save_dir, stage["name"])

        hard = results["scenarios"].get("hard_500m", {})
        hard_dr = results["scenarios"].get("hard_500m_dr", {})
        notify(f"v4 {stage['name']}",
               f"hard={hard.get('landing_rate', 0)*100:.0f}%, "
               f"hard_dr={hard_dr.get('landing_rate', 0)*100:.0f}%")

        train_envs.close()
        eval_env.close()

    # Final evaluation
    if model is not None:
        model.save(save_dir / "final_model")
        print(f"\n{'='*60}")
        print("FINAL — v4 High-Altitude")
        print(f"Features: {', '.join(feature_names)}")
        print(f"Network: {NETWORK}")
        print(f"{'='*60}")
        final_results = full_eval(model, feature_names, save_dir, "FINAL")

        config["completed"] = datetime.now().isoformat()
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

    notify("v4 highalt done", "Check results")


if __name__ == "__main__":
    main()
