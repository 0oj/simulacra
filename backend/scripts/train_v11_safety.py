#!/usr/bin/env python3
"""
v11 Safety Envelope — First experiment with actuator feedback in observation.

Key changes from v9/v10:
  1. Obs dim: 22 → 28 (6 safety envelope signals appended)
     - throttle_sat, gimbal_outer_sat, gimbal_inner_sat (actuator feedback)
     - speed_ratio, tilt_margin, descent_ratio (state envelope)
  2. r_safety reward term penalises actuator saturation + dangerous descent
  3. Hypothesis: agent learns smooth control by seeing when commands are clipped

Target: <50 thrust reversals per flight (v10c.4 baseline: 173)

Usage:
    cd backend
    uv run python scripts/train_v11_safety.py [--n-envs 512] [--total-steps 500000000]
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
# Configuration
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

TOTAL_STEPS = 500_000_000  # 500M for initial safety envelope test

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

EVAL_INTERVAL = 50_000_000  # eval every 50M (more frequent for shorter run)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

_shutdown = False


def _signal_handler(signum, frame):
    global _shutdown
    _shutdown = True
    print("\n[!] Shutdown requested, finishing current rollout...")


def hopper_eval(trainer, n_episodes=200, max_steps=750, **env_kwargs):
    """Fast batched evaluation using pure-JAX PPO inference."""
    from simulacra.jax_hopper_env import JaxHopperVecEnv

    eval_env = JaxHopperVecEnv(
        num_envs=n_episodes,
        use_guidance=True,
        max_tilt=0.25,
        max_episode_steps=max_steps,
        max_altitude=MAX_ALTITUDE,
        acceptance_radius=0.5,
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


def full_eval(trainer, save_dir, stage_name):
    """Run all eval configs, print results, save to JSONL."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "stage": stage_name,
        "scenarios": {},
    }
    for name, kw in EVAL_CONFIGS.items():
        r = hopper_eval(trainer, 200, **kw)
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
        description="v11 Safety Envelope — actuator feedback in obs")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--n-envs", type=int, default=512,
                        help="Parallel JAX envs (default: 512, conservative for 12GB)")
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--total-steps", type=int, default=TOTAL_STEPS)
    parser.add_argument("--save-dir", type=str, default="runs/v11_safety")
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

    from simulacra.jax_ppo import PPOTrainer, PPOConfig
    from simulacra.jax_hopper_env import JaxHopperVecEnv

    save_dir = Path(args.save_dir)
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

    print("=" * 60)
    print(f"v11 Safety Envelope — {args.total_steps/1e6:.0f}M Steps")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Network: {args.hidden_dims} (actor + critic)")
    print(f"Obs: 28-dim (22 base + 6 safety envelope)")
    print(f"  Safety signals: throttle_sat, gimbal_outer_sat, gimbal_inner_sat,")
    print(f"                  speed_ratio, tilt_margin, descent_ratio")
    print(f"Reward: R8 + r_safety (actuator saturation + descent excess)")
    print(f"Envs: {args.n_envs} (JAX vectorized)")
    print(f"Rollout: {args.n_steps} steps/env, batch {args.batch_size}")
    print(f"Gamma: {args.gamma} | Ent: {args.ent_coef}")
    print(f"Episode: {args.max_episode_steps} steps | LR: {lr_s} → {lr_e}")
    print(f"Acceptance: {args.acceptance_radius}m | max_tilt: {args.max_tilt}")
    print(f"Tilt penalty: -{args.tilt_penalty} | Smoothness: {args.smoothness_weight}x")
    print(f"DR: wind={args.dr_wind}N, gps={args.dr_gps_noise}m")
    print(f"Scenarios: {len(ALL_SCENARIOS)} types")
    print("=" * 60)

    config_dict = {
        "name": "v11_safety",
        "obs_dim": 28,
        "safety_envelope": {
            "signals": [
                "throttle_sat", "gimbal_outer_sat", "gimbal_inner_sat",
                "speed_ratio", "tilt_margin", "descent_ratio",
            ],
            "r_safety_weights": {
                "throttle_sat": 0.02,
                "gimbal_sat": 0.01,
                "descent_excess": 0.1,
            },
        },
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
        "backend": backend,
        "started": datetime.now().isoformat(),
    }
    with open(save_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

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
        dr_wind_max=args.dr_wind,
        dr_gps_noise_max=args.dr_gps_noise,
        dr_baro_noise_max=args.dr_baro_noise,
        dr_vel_noise_max=args.dr_vel_noise,
        dr_gyro_noise_max=args.dr_gyro_noise,
    )

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
    print(f"{'='*60}")

    log_entries = []

    try:
        trainer.train(
            total_steps=args.total_steps,
            save_dir=save_dir,
            save_freq=50_000_000,  # save every 50M
            stage_name="v11: safety envelope",
            log_entries=log_entries,
            shutdown_flag=[False],
            lr_start=lr_s,
            lr_end=lr_e,
            eval_fn=lambda t: _periodic_eval(t, save_dir, t.total_steps),
            eval_freq=EVAL_INTERVAL,
        )
    except KeyboardInterrupt:
        print("\n  Interrupted.")

    with open(save_dir / "training_log.jsonl", "w") as f:
        for e in log_entries:
            f.write(json.dumps(e) + "\n")

    total_elapsed = (time.time() - train_start) / 60
    trainer.save(save_dir / "final_model")

    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION (trained {total_elapsed:.0f} min)")
    print(f"{'='*60}")
    final = full_eval(trainer, save_dir, "FINAL")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Total time: {total_elapsed:.0f} min")
    for name, r in final["scenarios"].items():
        pct = 100 * (r["arrive_rate"] + r["land_rate"])
        print(f"  {name}: {pct:.0f}% ok ({100*r['crash_rate']:.0f}% crash)")

    config_dict["completed"] = datetime.now().isoformat()
    config_dict["elapsed_min"] = round(total_elapsed, 1)
    config_dict["final_results"] = final["scenarios"]
    with open(save_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"\nResults saved to {save_dir}")


def _periodic_eval(trainer, save_dir, steps):
    label = f"{steps/1e6:.0f}M steps"
    print(f"\n  --- Eval at {label} ---")
    results = full_eval(trainer, save_dir, label)
    total_ok = sum(r["total_ok"] for r in results["scenarios"].values())
    total_n = sum(r["total"] for r in results["scenarios"].values())
    print(f"  Overall: {total_ok}/{total_n} ({100*total_ok/total_n:.0f}% ok)")


if __name__ == "__main__":
    main()
