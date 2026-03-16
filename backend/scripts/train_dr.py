"""
Domain Randomization Training — Robust landing policy via physics variation.

Trains with the best physics features (from ablation study) + domain randomization
to produce a policy that generalizes across physical parameter variation.

Approach (inspired by Dresia 2025):
  Stage 1: Low altitude, nominal physics — learn basic landing
  Stage 2: Medium altitude, mild DR — begin generalizing
  Stage 3: High altitude, full DR + sensor noise — robustify

Usage:
    cd backend
    uv run python scripts/train_dr.py

Results saved to runs/guidance_dr/
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

# Best feature set — confirmed by ablation study (Variant C: minimal)
# brake_margin + vel_error alone achieve 42% Hard at Stage 2, matching v3_bignet final
# while using 5.5× fewer parameters. Adding more features hurts (B: 5% Hard).
FEATURE_SET = ["brake_margin", "vel_error"]

# Network: 64×64 matches ablation C's proven architecture.
# C with 64×64 reached 35% Hard at Stage 1 vs 4% Hard for 128×128 + brake_focus.
# DR variation should be handled by the physics-informed features, not capacity.
NETWORK = dict(pi=[64, 64], vf=[64, 64])

# Domain randomization schedules (mild → full)
DR_MILD = DomainRandomization(
    # Mild physics variation
    dry_mass_scale=(0.95, 1.05),       # ±5%
    fuel_mass_scale=(0.95, 1.05),
    max_thrust_scale=(0.97, 1.03),     # ±3%
    burn_rate_scale=(0.95, 1.05),
    # Mild actuator variation
    gimbal_rate_scale=(0.9, 1.1),      # ±10%
    gimbal_tau_scale=(0.85, 1.15),     # ±15%
    # Light wind
    wind_speed_range=(0.0, 1.0),       # 0-1 m/s
    # No sensor noise yet
)

DR_FULL = DomainRandomization(
    # Full physics variation
    dry_mass_scale=(0.85, 1.15),       # ±15%
    fuel_mass_scale=(0.85, 1.15),
    max_thrust_scale=(0.92, 1.08),     # ±8%
    burn_rate_scale=(0.85, 1.15),
    length_scale=(0.95, 1.05),         # ±5%
    # Full actuator variation
    gimbal_rate_scale=(0.8, 1.2),      # ±20%
    gimbal_tau_scale=(0.7, 1.3),       # ±30%
    # Moderate wind
    wind_speed_range=(0.0, 3.0),       # 0-3 m/s
    # Mild gravity/atmosphere
    gravity_scale=(0.995, 1.005),      # ±0.5%
    air_density_scale=(0.9, 1.1),      # ±10%
    # Sensor noise
    gps_noise_std=(0.0, 1.5),          # 0-1.5m
    baro_noise_std=(0.0, 0.5),         # 0-0.5m
    velocity_noise_std=(0.0, 0.3),     # 0-0.3 m/s
    orientation_noise_std=(0.0, 0.02), # 0-1.1°
    gyro_noise_std=(0.0, 0.01),        # 0-0.01 rad/s
    gyro_bias_range=(-0.005, 0.005),   # ±0.005 rad/s
)

STAGES = [
    {
        "name": "Stage 1: 80m, nominal",
        "steps": 2_000_000,
        "dr": None,  # No DR — learn basics first
        "env_kwargs": dict(
            init_altitude_range=(5.0, 80.0),
            init_velocity_range=(-9.0, 0.0),
            init_horizontal_range=(-7.0, 7.0),
            init_tilt_range=(-0.17, 0.17),
            init_fuel_range=(0.6, 1.0),
        ),
    },
    {
        "name": "Stage 2: 130m, mild DR",
        "steps": 2_500_000,
        "dr": DR_MILD,
        "env_kwargs": dict(
            init_altitude_range=(5.0, 130.0),
            init_velocity_range=(-12.0, 0.0),
            init_horizontal_range=(-12.0, 12.0),
            init_tilt_range=(-0.25, 0.25),
            init_fuel_range=(0.5, 1.0),
        ),
    },
    {
        "name": "Stage 3: 200m, full DR",
        "steps": 4_000_000,  # More steps for harder task
        "dr": DR_FULL,
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
    ("easy_60m", dict(
        init_altitude_range=(5., 60.), init_velocity_range=(-8., 0.),
        init_horizontal_range=(-5., 5.), init_tilt_range=(-0.15, 0.15))),
    ("medium_130m", dict(
        init_altitude_range=(5., 130.), init_velocity_range=(-12., 0.),
        init_horizontal_range=(-12., 12.), init_tilt_range=(-0.25, 0.25))),
    ("hard_200m", dict(
        init_altitude_range=(5., 200.), init_velocity_range=(-15., 0.),
        init_horizontal_range=(-20., 20.), init_tilt_range=(-0.3, 0.3))),
    # DR robustness eval: hard scenario WITH domain randomization
    ("hard_200m_dr", dict(
        init_altitude_range=(5., 200.), init_velocity_range=(-15., 0.),
        init_horizontal_range=(-20., 20.), init_tilt_range=(-0.3, 0.3))),
]


_shutdown = False


def _signal_handler(signum, frame):
    global _shutdown
    _shutdown = True
    print("\n[!] Shutdown requested...")


def make_env(seed, feature_names, dr=None, **env_kwargs):
    def _init():
        from stable_baselines3.common.monitor import Monitor
        env = RocketLandingEnvV2(domain_randomization=dr, **env_kwargs)
        env = GuidanceWrapper(env)
        env = FeatureWrapper(env, feature_names)
        env = GuidanceReward(env)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def evaluate(model, feature_names, n=100, dr=None, **env_kwargs):
    """Evaluate with the appropriate feature wrapper and optional DR."""
    env = FeatureWrapper(
        GuidanceWrapper(RocketLandingEnvV2(domain_randomization=dr, **env_kwargs)),
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
        # Use DR for the DR eval scenario
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

    save_dir = Path("runs/guidance_dr")
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "checkpoints").mkdir(exist_ok=True)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    feature_names = FEATURE_SET
    n_envs = 4
    n_features = len(feature_names)

    # Save experiment config
    config = {
        "name": "guidance_dr",
        "description": "Domain randomized training with physics features + bigger network",
        "features": feature_names,
        "n_features": n_features,
        "total_obs_dims": 13 + n_features,
        "network": NETWORK,
        "n_envs": n_envs,
        "ppo": {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
        },
        "dr_mild": {k: v for k, v in DR_MILD.__dict__.items()},
        "dr_full": {k: v for k, v in DR_FULL.__dict__.items()},
        "stages": [{"name": s["name"], "steps": s["steps"], "has_dr": s["dr"] is not None} for s in STAGES],
        "started": datetime.now().isoformat(),
    }
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    print("=" * 60)
    print("Domain Randomized Training")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Features ({n_features}): {', '.join(feature_names)}")
    print(f"Obs dims: {13 + n_features}")
    print(f"Network: {NETWORK}")
    print(f"Stages: nominal → mild DR → full DR + sensor noise")
    print("=" * 60)

    notify("DR training started", f"Features: {', '.join(feature_names)}, Net: {NETWORK}")

    # Check for resume
    checkpoints = sorted(save_dir.glob("checkpoints/dr_*_steps.zip"))
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
        print(f"\n{'='*60}")
        print(f"  {stage['name']}")
        print(f"  Steps: {stage['steps']:,}")
        print(f"  DR: {'Yes' if dr else 'No (nominal)'}")
        print(f"{'='*60}")

        train_envs = SubprocVecEnv([
            make_env(i, feature_names, dr=dr, **stage["env_kwargs"]) for i in range(n_envs)
        ])
        eval_env = DummyVecEnv([
            make_env(999, feature_names, dr=None, **stage["env_kwargs"])  # Eval on nominal
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
                    tensorboard_log=str(save_dir / "tensorboard"),
                    policy_kwargs=dict(net_arch=NETWORK),
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
                    notify("Paused", f"DR: {self.stage_name}")
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
                eval_freq=10_000 // n_envs,
                n_eval_episodes=20,
                deterministic=True,
                verbose=0,
            ),
            CheckpointCallback(
                save_freq=200_000 // n_envs,
                save_path=str(save_dir / "checkpoints"),
                name_prefix="dr",
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

        # Structured evaluation (on NOMINAL physics — no DR)
        print(f"\n  --- Eval after {stage['name']} ---")
        results = full_eval(model, feature_names, save_dir, stage["name"])

        hard = results["scenarios"].get("hard_200m", {})
        hard_dr = results["scenarios"].get("hard_200m_dr", {})
        notify(f"DR {stage['name']}",
               f"hard={hard.get('landing_rate', 0)*100:.0f}%, "
               f"hard_dr={hard_dr.get('landing_rate', 0)*100:.0f}%")

        train_envs.close()
        eval_env.close()

    # Final evaluation
    if model is not None:
        model.save(save_dir / "final_model")
        print(f"\n{'='*60}")
        print("FINAL — DR Training")
        print(f"Features: {', '.join(feature_names)}")
        print(f"Network: {NETWORK}")
        print(f"{'='*60}")
        final_results = full_eval(model, feature_names, save_dir, "FINAL")

        config["completed"] = datetime.now().isoformat()
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

    notify("DR training done", "Check results")


if __name__ == "__main__":
    main()
