#!/usr/bin/env python3
"""v13 ablation: systematic safety/smoothness penalty experiments.

Approach C ablation study — isolates the contribution of safety obs dims,
safety reward penalties, and the new reversal penalty term.

Variants:
    C1  no_safety_28     28-dim obs, no safety reward, no reversal  (obs dim effect)
    C2  reversal_28      28-dim obs, no safety reward, reversal=0.05  (reversal only)
    C3  reversal_22      22-dim obs, no safety reward, reversal=0.05  (reversal w/o safety obs)
    C4  half_safety      28-dim obs, half safety weights, no reversal  (reduced penalty)
    C5  baseline_22      22-dim obs, no safety reward, no reversal  (v10c control)

Usage:
    uv run python scripts/train_v13_ablation.py C1  # no safety, 28-dim
    uv run python scripts/train_v13_ablation.py C2  # reversal penalty, 28-dim
    uv run python scripts/train_v13_ablation.py C3  # reversal penalty, 22-dim
    uv run python scripts/train_v13_ablation.py C4  # half safety weights
    uv run python scripts/train_v13_ablation.py C5  # baseline 22-dim (control)

All variants share identical PPO hyperparameters, scenario config, DR settings,
and eval protocol for fair comparison.
"""

import sys
import json
import time
import signal
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import jax
import jax.numpy as jnp


# ──────────────────────────────────────────────────────────────
# Variant Definitions
# ──────────────────────────────────────────────────────────────

VARIANTS = {
    "C1": {
        "name": "no_safety_28",
        "description": "Isolate: do extra obs dims alone hurt?",
        "include_safety_obs": True,
        "safety_reward_weights": {"throttle": 0.0, "gimbal": 0.0, "descent": 0.0},
        "reversal_penalty_weight": 0.0,
        "obs_dim": 28,
    },
    "C2": {
        "name": "reversal_28",
        "description": "New reversal penalty only, 28-dim obs",
        "include_safety_obs": True,
        "safety_reward_weights": {"throttle": 0.0, "gimbal": 0.0, "descent": 0.0},
        "reversal_penalty_weight": 0.05,
        "obs_dim": 28,
    },
    "C3": {
        "name": "reversal_22",
        "description": "Reversal penalty without safety obs",
        "include_safety_obs": False,
        "safety_reward_weights": {"throttle": 0.0, "gimbal": 0.0, "descent": 0.0},
        "reversal_penalty_weight": 0.05,
        "obs_dim": 22,
    },
    "C4": {
        "name": "half_safety",
        "description": "Reduced safety penalty (half v11 weights)",
        "include_safety_obs": True,
        "safety_reward_weights": {"throttle": 0.01, "gimbal": 0.005, "descent": 0.05},
        "reversal_penalty_weight": 0.0,
        "obs_dim": 28,
    },
    "C5": {
        "name": "baseline_22",
        "description": "v10c config (no safety, 22-dim obs) -- control",
        "include_safety_obs": False,
        "safety_reward_weights": {"throttle": 0.0, "gimbal": 0.0, "descent": 0.0},
        "reversal_penalty_weight": 0.0,
        "obs_dim": 22,
    },
}


# ──────────────────────────────────────────────────────────────
# Shared Configuration
# ──────────────────────────────────────────────────────────────

MAX_ALTITUDE = 100.0

ALL_SCENARIOS = [
    "vertical", "vertical_micro", "lateral", "lateral_short", "diagonal",
    "landing", "landing_precision", "landing_high", "liftoff", "liftoff_fast",
]

SCENARIO_WEIGHTS = [
    1.0,   # vertical
    0.5,   # vertical_micro
    1.0,   # lateral
    0.5,   # lateral_short
    1.0,   # diagonal
    2.0,   # landing
    1.5,   # landing_precision
    1.0,   # landing_high
    2.0,   # liftoff
    1.0,   # liftoff_fast
]

TOTAL_STEPS = 500_000_000  # 500M for fair comparison with v11/v12

LR_START = 2e-4
LR_END = 1e-5

EVAL_CONFIGS = {
    "vertical": dict(
        scenarios=["vertical"],
        alt_range=(5.0, 30.0),
        horiz_range=(-5.0, 5.0),
        vel_range=(-1.5, 1.5),
    ),
    "lateral": dict(
        scenarios=["lateral"],
        alt_range=(10.0, 30.0),
        horiz_range=(-15.0, 15.0),
        vel_range=(-1.5, 1.5),
    ),
    "diagonal": dict(
        scenarios=["diagonal"],
        alt_range=(5.0, 40.0),
        horiz_range=(-15.0, 15.0),
        vel_range=(-2.0, 2.0),
    ),
    "landing": dict(
        scenarios=["landing"],
        alt_range=(10.0, 40.0),
        horiz_range=(-10.0, 10.0),
        vel_range=(-2.0, 0.0),
    ),
    "liftoff": dict(
        scenarios=["liftoff"],
        alt_range=(5.0, 25.0),
        horiz_range=(-5.0, 5.0),
        vel_range=(-0.5, 0.5),
    ),
}

EVAL_INTERVAL = 2_000_000  # eval every 2M steps (1000 episodes across 5 scenarios)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

_shutdown = False


def _signal_handler(signum, frame):
    global _shutdown
    _shutdown = True
    print("\n[!] Shutdown requested, finishing current rollout...")


def hopper_eval(trainer, variant, n_episodes=200, max_steps=750, **env_kwargs):
    """Fast batched evaluation using pure-JAX PPO inference.

    Creates eval env with the variant's obs settings (include_safety_obs)
    so the obs_dim matches the trained network.
    """
    from simulacra.jax_hopper_env import JaxHopperVecEnv

    eval_env = JaxHopperVecEnv(
        num_envs=n_episodes,
        use_guidance=True,
        max_tilt=0.25,
        max_episode_steps=max_steps,
        max_altitude=MAX_ALTITUDE,
        acceptance_radius=0.5,
        include_safety_obs=variant["include_safety_obs"],
        # Eval uses zero safety/reversal reward (pure task performance)
        safety_reward_weights={"throttle": 0.0, "gimbal": 0.0, "descent": 0.0},
        reversal_penalty_weight=0.0,
        seed=42,
        **env_kwargs,
    )

    obs = eval_env.reset()
    obs = jnp.array(obs)
    done_mask = np.zeros(n_episodes, dtype=bool)
    arrived = np.zeros(n_episodes, dtype=bool)
    landed = np.zeros(n_episodes, dtype=bool)
    crashed = np.zeros(n_episodes, dtype=bool)

    for _ in range(max_steps):
        actions = trainer.predict(obs, deterministic=True)
        obs_np, _, step_dones, infos = eval_env.step(actions)
        obs = jnp.array(obs_np)

        for i in range(n_episodes):
            if step_dones[i] and not done_mask[i]:
                done_mask[i] = True
                outcome = infos[i].get("outcome", "")
                if outcome == "arrived":
                    arrived[i] = True
                elif outcome == "landed":
                    landed[i] = True
                elif outcome == "crash":
                    crashed[i] = True

        if np.all(done_mask):
            break

    arrive_rate = float(np.sum(arrived)) / n_episodes
    land_rate = float(np.sum(landed)) / n_episodes
    crash_rate = float(np.sum(crashed)) / n_episodes
    total_ok = int(np.sum(arrived) + np.sum(landed))
    return {
        "arrive_rate": arrive_rate,
        "land_rate": land_rate,
        "crash_rate": crash_rate,
        "total_ok": total_ok,
        "total": n_episodes,
    }


def full_eval(trainer, variant, save_dir, stage_name):
    """Run all eval configs, print results, save to JSONL."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "stage": stage_name,
        "scenarios": {},
    }
    for name, kw in EVAL_CONFIGS.items():
        r = hopper_eval(trainer, variant, 200, **kw)
        results["scenarios"][name] = r
        pct = 100 * (r["arrive_rate"] + r["land_rate"])
        print(f"    {name}: {r['total_ok']}/{r['total']} "
              f"({pct:.0f}% ok, {100*r['crash_rate']:.0f}% crash)")

    with open(save_dir / "results.jsonl", "a") as f:
        f.write(json.dumps(results) + "\n")
    return results


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    global _shutdown

    parser = argparse.ArgumentParser(
        description="v13 ablation: systematic safety/smoothness penalty experiments")
    parser.add_argument("variant", type=str, nargs="?", default="C1",
                        choices=VARIANTS.keys(),
                        help="Ablation variant (C1-C5)")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--n-envs", type=int, default=512,
                        help="Parallel JAX envs (default: 512)")
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--total-steps", type=int, default=TOTAL_STEPS)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--max-tilt", type=float, default=0.25)
    parser.add_argument("--acceptance-radius", type=float, default=0.5)
    parser.add_argument("--smoothness-weight", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--tilt-penalty", type=float, default=0.20)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--lr-start", type=float, default=None)
    parser.add_argument("--lr-end", type=float, default=None)
    # Domain randomization
    parser.add_argument("--dr-wind", type=float, default=3.0)
    parser.add_argument("--dr-gps-noise", type=float, default=0.3)
    parser.add_argument("--dr-baro-noise", type=float, default=0.2)
    parser.add_argument("--dr-vel-noise", type=float, default=0.15)
    parser.add_argument("--dr-gyro-noise", type=float, default=0.02)
    args = parser.parse_args()

    variant_id = args.variant
    variant = VARIANTS[variant_id]
    run_name = f"v13_{variant_id}_{variant['name']}"

    from simulacra.jax_ppo import PPOTrainer, PPOConfig
    from simulacra.jax_hopper_env import JaxHopperVecEnv

    save_dir = Path(f"runs/{run_name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "checkpoints").mkdir(exist_ok=True)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    devices = jax.devices()
    backend = devices[0].platform if devices else "cpu"
    print(f"JAX backend: {backend} ({len(devices)} device(s))")
    for d in devices:
        print(f"  {d}")

    lr_s = args.lr_start if args.lr_start is not None else LR_START
    lr_e = args.lr_end if args.lr_end is not None else LR_END

    # ── Banner ──
    print()
    print("=" * 60)
    print(f"v13 ABLATION — Variant {variant_id}: {variant['name']}")
    print(f"  {variant['description']}")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Run name: {run_name}")
    print(f"Output: {save_dir}")
    print()
    print(f"Obs dim: {variant['obs_dim']} "
          f"({'22 base + 6 safety' if variant['include_safety_obs'] else '22 base, no safety obs'})")
    print(f"Safety reward weights: {variant['safety_reward_weights']}")
    print(f"Reversal penalty weight: {variant['reversal_penalty_weight']}")
    print()
    print(f"Network: {args.hidden_dims} (actor + critic)")
    print(f"Envs: {args.n_envs} (JAX vectorized)")
    print(f"Rollout: {args.n_steps} steps/env, batch {args.batch_size}")
    print(f"Gamma: {args.gamma} | Ent: {args.ent_coef}")
    print(f"Episode: {args.max_episode_steps} steps | LR: {lr_s} -> {lr_e}")
    print(f"Total steps: {args.total_steps/1e6:.0f}M")
    print(f"Acceptance: {args.acceptance_radius}m | max_tilt: {args.max_tilt}")
    print(f"Tilt penalty: -{args.tilt_penalty} | Smoothness: {args.smoothness_weight}x")
    print(f"DR: wind={args.dr_wind}N, gps={args.dr_gps_noise}m")
    print(f"Eval: every {EVAL_INTERVAL/1e6:.0f}M steps, {len(EVAL_CONFIGS)} scenarios x 200 episodes")
    print(f"Scenarios: {len(ALL_SCENARIOS)} types")
    print("=" * 60)

    # ── Config (saved to JSON + JSONL) ──
    config_dict = {
        "name": run_name,
        "experiment": "v13_ablation",
        "variant_id": variant_id,
        "variant": variant,
        "obs_dim": variant["obs_dim"],
        "flight_modes": ["TRACK", "LAND", "LIFTOFF"],
        "network": {"pi": args.hidden_dims, "vf": args.hidden_dims},
        "n_envs": args.n_envs,
        "ppo": {
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": 10,
            "gamma": args.gamma,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": args.ent_coef,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        },
        "lr_schedule": {"start": lr_s, "end": lr_e, "type": "cosine"},
        "training": {
            "total_steps": args.total_steps,
            "acceptance_radius": args.acceptance_radius,
            "max_episode_steps": args.max_episode_steps,
            "smoothness_weight": args.smoothness_weight,
            "max_tilt": args.max_tilt,
            "tilt_penalty": args.tilt_penalty,
        },
        "domain_randomization": {
            "wind_max": args.dr_wind,
            "gps_noise_max": args.dr_gps_noise,
            "baro_noise_max": args.dr_baro_noise,
            "vel_noise_max": args.dr_vel_noise,
            "gyro_noise_max": args.dr_gyro_noise,
        },
        "eval": {
            "interval_steps": EVAL_INTERVAL,
            "episodes_per_scenario": 200,
            "scenarios": list(EVAL_CONFIGS.keys()),
        },
        "backend": backend,
        "started": datetime.now().isoformat(),
    }
    with open(save_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # ── Environment ──
    env = JaxHopperVecEnv(
        num_envs=args.n_envs,
        scenarios=ALL_SCENARIOS,
        scenario_weights=SCENARIO_WEIGHTS,
        use_guidance=True,
        max_tilt=args.max_tilt,
        max_episode_steps=args.max_episode_steps,
        max_altitude=MAX_ALTITUDE,
        acceptance_radius=args.acceptance_radius,
        alt_range=(5.0, 80.0),
        horiz_range=(-30.0, 30.0),
        vel_range=(-5.0, 5.0),
        init_tilt_range=(-0.25, 0.25),
        init_fuel_range=(0.4, 1.0),
        smoothness_weight=args.smoothness_weight,
        tilt_penalty=args.tilt_penalty,
        # Variant-specific settings
        include_safety_obs=variant["include_safety_obs"],
        safety_reward_weights=variant["safety_reward_weights"],
        reversal_penalty_weight=variant["reversal_penalty_weight"],
        # DR
        dr_wind_max=args.dr_wind,
        dr_gps_noise_max=args.dr_gps_noise,
        dr_baro_noise_max=args.dr_baro_noise,
        dr_vel_noise_max=args.dr_vel_noise,
        dr_gyro_noise_max=args.dr_gyro_noise,
    )

    # ── PPO Trainer ──
    ppo_config = PPOConfig(
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=args.gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        lr=lr_s,
        hidden_dims=tuple(args.hidden_dims),
    )
    trainer = PPOTrainer(env, ppo_config, seed=0)
    print(f"Network params: {trainer.get_param_count():,}")

    if args.resume:
        print(f"\nResuming from {args.resume}")
        trainer.load(Path(args.resume))

    # ── Training ──
    train_start = time.time()
    print(f"\n{'='*60}")
    print(f"  Training: {args.total_steps/1e6:.0f}M steps")
    print(f"  Variant {variant_id}: {variant['name']}")
    print(f"{'='*60}")

    log_entries = []

    try:
        trainer.train(
            total_steps=args.total_steps,
            save_dir=save_dir,
            save_freq=50_000_000,  # save every 50M
            stage_name=f"v13 {variant_id}: {variant['name']}",
            log_entries=log_entries,
            shutdown_flag=[False],
            lr_start=lr_s,
            lr_end=lr_e,
            eval_fn=lambda t: _periodic_eval(t, variant_id, variant, save_dir, t.total_steps),
            eval_freq=EVAL_INTERVAL,
        )
    except KeyboardInterrupt:
        print("\n  Interrupted.")

    # Annotate log entries with variant info
    for entry in log_entries:
        entry["variant_id"] = variant_id
        entry["variant_name"] = variant["name"]

    with open(save_dir / "training_log.jsonl", "w") as f:
        for e in log_entries:
            f.write(json.dumps(e) + "\n")

    total_elapsed = (time.time() - train_start) / 60
    trainer.save(save_dir / "final_model")

    # ── Final Evaluation ──
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION (trained {total_elapsed:.0f} min)")
    print(f"Variant {variant_id}: {variant['name']}")
    print(f"{'='*60}")
    final = full_eval(trainer, variant, save_dir, "FINAL")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"SUMMARY — v13 {variant_id}: {variant['name']}")
    print(f"{'='*60}")
    print(f"  {variant['description']}")
    print(f"  Obs dim: {variant['obs_dim']}")
    print(f"  Safety weights: {variant['safety_reward_weights']}")
    print(f"  Reversal penalty: {variant['reversal_penalty_weight']}")
    print(f"  Total time: {total_elapsed:.0f} min")
    for name, r in final["scenarios"].items():
        pct = 100 * (r["arrive_rate"] + r["land_rate"])
        print(f"  {name}: {pct:.0f}% ok ({100*r['crash_rate']:.0f}% crash)")

    # ── Update config with final results ──
    config_dict["completed"] = datetime.now().isoformat()
    config_dict["elapsed_min"] = round(total_elapsed, 1)
    config_dict["final_results"] = final["scenarios"]
    with open(save_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"\nResults saved to {save_dir}")


def _periodic_eval(trainer, variant_id, variant, save_dir, steps):
    label = f"{steps/1e6:.0f}M steps"
    print(f"\n  --- Eval at {label} (variant {variant_id}) ---")
    results = full_eval(trainer, variant, save_dir, label)
    total_ok = sum(r["total_ok"] for r in results["scenarios"].values())
    total_n = sum(r["total"] for r in results["scenarios"].values())
    print(f"  Overall: {total_ok}/{total_n} ({100*total_ok/total_n:.0f}% ok)")


if __name__ == "__main__":
    main()
