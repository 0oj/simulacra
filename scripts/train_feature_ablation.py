"""
Feature Ablation Study — Systematic comparison of physics-informed observation features.

This script runs a single feature variant specified by command-line argument.
Run 4 instances in parallel for the full ablation:

    cd backend
    uv run python scripts/train_feature_ablation.py A &   # brake_focus
    uv run python scripts/train_feature_ablation.py B &   # kitchen_sink
    uv run python scripts/train_feature_ablation.py C &   # minimal
    uv run python scripts/train_feature_ablation.py D &   # energy

Each variant uses identical training config (PPO, 64×64, 3-stage curriculum)
differing ONLY in the physics features appended to observations.

Results are saved to runs/ablation_{variant}/ with:
  - checkpoints/ (periodic model snapshots)
  - eval_logs_stage{N}/ (SB3 eval callback logs)
  - results.jsonl (structured results for thesis plots)
  - tensorboard/ (training curves)
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

from simulacra.gym_env import RocketLandingEnvV2
from train_guidance import (
    GuidanceWrapper, GuidanceReward, _rotate_body_to_world,
    evaluate as _evaluate_raw, notify,
)

# ──────────────────────────────────────────────────────────────
# Feature definitions
# ──────────────────────────────────────────────────────────────

# Constants matching the reward function and autopilot
A_MAX = 4.0       # m/s², approximate max deceleration (from GuidanceReward)
K_DESCENT = 2.0   # descent rate gain (from LandingController)
G = 9.81           # m/s², gravity


def compute_all_features(obs):
    """
    Compute all candidate physics features from the 13-dim base observation.

    Returns a dict of {feature_name: value} for logging and selection.
    """
    x, y, z = float(obs[0]), float(obs[1]), float(obs[2])
    vx, vy, vz = float(obs[3]), float(obs[4]), float(obs[5])
    qw, qx, qy, qz = float(obs[6]), float(obs[7]), float(obs[8]), float(obs[9])

    y_safe = max(y, 0.1)

    features = {}

    # F1: sqrt(y) — velocity reference scale
    features["sqrt_y"] = math.sqrt(y_safe)

    # F2: vy²/(2y) — required deceleration to stop
    features["decel_required"] = (vy ** 2) / (2.0 * y_safe)

    # F3: horizontal distance to pad
    features["horiz_dist"] = math.sqrt(x ** 2 + z ** 2)

    # F4: glide slope tangent
    features["glide_slope"] = features["horiz_dist"] / y_safe

    # F5: brake margin — altitude remaining after full brake
    # Positive = have altitude budget. Zero/negative = must brake NOW.
    features["brake_margin"] = y - (vy ** 2) / (2.0 * A_MAX)

    # F6: velocity error — gap between actual and desired descent rate
    # vy is negative when descending, desired = -K_DESCENT * sqrt(y)
    # vel_error > 0 means "too slow" (should coast), < 0 means "too fast" (should brake)
    features["vel_error"] = vy + K_DESCENT * math.sqrt(y_safe)

    # F7: nose_y — Y-component of the body X-axis in world frame (uprightness)
    # From quaternion rotation of [1,0,0]: y_component = 2(qw*qy + qx*qz)
    features["nose_y"] = 2.0 * (qw * qy + qx * qz)

    # F8: horizontal speed magnitude
    features["horiz_speed"] = math.sqrt(vx ** 2 + vz ** 2)

    # F9: specific mechanical energy (energy to dissipate before landing)
    features["specific_energy"] = (vy ** 2) / 2.0 + G * y_safe

    return features


# ──────────────────────────────────────────────────────────────
# Variant definitions
# ──────────────────────────────────────────────────────────────

VARIANTS = {
    "A": {
        "name": "brake_focus",
        "features": ["sqrt_y", "brake_margin", "vel_error", "glide_slope"],
        "hypothesis": (
            "Replace decel_required (uninformative at high altitude) and horiz_dist "
            "(redundant with x,z in obs) with brake_margin (informative at ALL altitudes) "
            "and vel_error (direct control error signal for throttle)."
        ),
    },
    "B": {
        "name": "kitchen_sink",
        "features": [
            "sqrt_y", "decel_required", "horiz_dist", "glide_slope",
            "brake_margin", "vel_error", "nose_y", "horiz_speed", "specific_energy",
        ],
        "hypothesis": (
            "All 9 features. Tests whether more features help or hurt. "
            "If this outperforms A, the network can use extra info. "
            "If it underperforms, the extra features add noise."
        ),
    },
    "C": {
        "name": "minimal",
        "features": ["brake_margin", "vel_error"],
        "hypothesis": (
            "Just 2 features: brake_margin (when to brake) and vel_error (how hard). "
            "Tests whether these two features alone capture the essential physics. "
            "If this matches A, the other features are unnecessary."
        ),
    },
    "D": {
        "name": "energy",
        "features": ["sqrt_y", "brake_margin", "specific_energy", "glide_slope"],
        "hypothesis": (
            "Energy-based framing. specific_energy tells the total energy budget; "
            "brake_margin tells the altitude budget; sqrt_y and glide_slope handle "
            "descent rate and lateral correction."
        ),
    },
}


# ──────────────────────────────────────────────────────────────
# Feature wrapper (parameterized by variant)
# ──────────────────────────────────────────────────────────────

class FeatureWrapper(gym.ObservationWrapper):
    """
    Appends selected physics features to the base 13-dim observation.

    The feature_names parameter selects which features from compute_all_features()
    to include. This allows systematic ablation studies.
    """

    def __init__(self, env, feature_names: list[str]):
        super().__init__(env)
        self.feature_names = feature_names
        n_features = len(feature_names)

        base_space = env.observation_space
        low = np.concatenate([
            base_space.low,
            np.full(n_features, -np.inf, dtype=np.float32),
        ])
        high = np.concatenate([
            base_space.high,
            np.full(n_features, np.inf, dtype=np.float32),
        ])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        all_features = compute_all_features(obs)
        selected = np.array(
            [all_features[name] for name in self.feature_names],
            dtype=np.float32,
        )
        return np.concatenate([obs, selected])


# ──────────────────────────────────────────────────────────────
# Training setup (identical to v3_features for fair comparison)
# ──────────────────────────────────────────────────────────────

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
    ("easy_60m", dict(
        init_altitude_range=(5., 60.), init_velocity_range=(-8., 0.),
        init_horizontal_range=(-5., 5.), init_tilt_range=(-0.15, 0.15))),
    ("medium_130m", dict(
        init_altitude_range=(5., 130.), init_velocity_range=(-12., 0.),
        init_horizontal_range=(-12., 12.), init_tilt_range=(-0.25, 0.25))),
    ("hard_200m", dict(
        init_altitude_range=(5., 200.), init_velocity_range=(-15., 0.),
        init_horizontal_range=(-20., 20.), init_tilt_range=(-0.3, 0.3))),
]

_shutdown = False


def _signal_handler(signum, frame):
    global _shutdown
    _shutdown = True
    print("\n[!] Shutdown requested...")


def make_env(seed, feature_names, **env_kwargs):
    def _init():
        from stable_baselines3.common.monitor import Monitor
        env = RocketLandingEnvV2(**env_kwargs)
        env = GuidanceWrapper(env)
        env = FeatureWrapper(env, feature_names)
        env = GuidanceReward(env)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def evaluate(model, feature_names, n=100, **env_kwargs):
    """Evaluate with the appropriate feature wrapper."""
    env = FeatureWrapper(
        GuidanceWrapper(RocketLandingEnvV2(**env_kwargs)),
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
        rate, avg_dist, landed = evaluate(model, feature_names, 100, **kw)
        results["scenarios"][scenario_name] = {
            "landing_rate": rate,
            "avg_dist_m": round(avg_dist, 2),
            "landed": landed,
            "total": 100,
        }
        print(f"    {scenario_name}: {landed}/100 ({100*rate:.0f}%), avg dist={avg_dist:.2f}m")

    # Append to JSONL results file
    results_file = save_dir / "results.jsonl"
    with open(results_file, "a") as f:
        f.write(json.dumps(results) + "\n")

    return results


def main():
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback

    # Parse variant from command line
    if len(sys.argv) < 2 or sys.argv[1] not in VARIANTS:
        print(f"Usage: {sys.argv[0]} <variant>")
        print(f"  Variants: {', '.join(VARIANTS.keys())}")
        for k, v in VARIANTS.items():
            print(f"    {k}: {v['name']} — {v['features']}")
        sys.exit(1)

    variant_key = sys.argv[1]
    variant = VARIANTS[variant_key]
    feature_names = variant["features"]
    variant_name = variant["name"]

    save_dir = Path(f"runs/ablation_{variant_key}_{variant_name}")

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "checkpoints").mkdir(exist_ok=True)

    n_envs = 4
    n_features = len(feature_names)

    # Save experiment config
    config = {
        "variant": variant_key,
        "name": variant_name,
        "features": feature_names,
        "hypothesis": variant["hypothesis"],
        "n_features": n_features,
        "total_obs_dims": 13 + n_features,
        "network": {"pi": [64, 64], "vf": [64, 64]},
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
        "stages": [{"name": s["name"], "steps": s["steps"]} for s in STAGES],
        "constants": {"a_max": A_MAX, "k_descent": K_DESCENT, "g": G},
        "started": datetime.now().isoformat(),
    }
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("=" * 60)
    print(f"Feature Ablation — Variant {variant_key}: {variant_name}")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Features ({n_features}): {', '.join(feature_names)}")
    print(f"Obs dims: {13 + n_features}")
    print(f"Network: pi=[64, 64], vf=[64, 64]")
    print(f"Hypothesis: {variant['hypothesis']}")
    print("=" * 60)

    notify(f"Ablation {variant_key} started",
           f"{variant_name}: {', '.join(feature_names)}")

    # Check for resume
    checkpoints = sorted(save_dir.glob(f"checkpoints/abl{variant_key}_*_steps.zip"))
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

        print(f"\n{'='*60}")
        print(f"  {stage['name']}")
        print(f"  Steps: {stage['steps']:,}")
        print(f"{'='*60}")

        train_envs = SubprocVecEnv([
            make_env(i, feature_names, **stage["env_kwargs"]) for i in range(n_envs)
        ])
        eval_env = DummyVecEnv([
            make_env(999, feature_names, **stage["env_kwargs"])
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
                    policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
                )
        else:
            model.set_env(train_envs)

        class ProgressCB(BaseCallback):
            def __init__(self, stage_name, save_dir, variant_key):
                super().__init__()
                self.stage_name = stage_name
                self.save_dir = save_dir
                self.variant_key = variant_key
                self.start = time.time()
                self.last_log = time.time()
                self.episodes = 0
                self.landings = 0
                self.recent = deque(maxlen=200)
                self.recent_dist = deque(maxlen=200)
                # Log training dynamics for thesis plots
                self.training_log = []

            def _on_step(self):
                if _shutdown:
                    self.model.save(str(self.save_dir / "checkpoints" / "interrupted"))
                    notify("Paused", f"{self.variant_key}: {self.stage_name}")
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
                # Save training dynamics log
                log_file = self.save_dir / f"training_log_stage{stage_idx}.jsonl"
                with open(log_file, "w") as f:
                    for entry in self.training_log:
                        f.write(json.dumps(entry) + "\n")

        progress_cb = ProgressCB(stage["name"], save_dir, variant_key)

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
                name_prefix=f"abl{variant_key}",
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

        # Summary notification
        hard = results["scenarios"].get("hard_200m", {})
        notify(f"Abl {variant_key} {stage['name']}",
               f"{hard.get('landing_rate', 0)*100:.0f}% hard, "
               f"{hard.get('avg_dist_m', 0):.1f}m")

        train_envs.close()
        eval_env.close()

    # Final evaluation
    if model is not None:
        model.save(save_dir / "final_model")
        print(f"\n{'='*60}")
        print(f"FINAL — Variant {variant_key}: {variant_name}")
        print(f"Features: {', '.join(feature_names)}")
        print(f"{'='*60}")
        final_results = full_eval(model, feature_names, save_dir, "FINAL")

        # Save completion marker
        config["completed"] = datetime.now().isoformat()
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    notify(f"Ablation {variant_key} done", "Check results")


if __name__ == "__main__":
    main()
