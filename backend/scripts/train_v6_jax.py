#!/usr/bin/env python3
"""
v6 Hopper Training — Pure-JAX PPO (no PyTorch, no SB3).

Entire training loop stays on GPU: rollout collection, GAE, minibatch
shuffling, gradient computation, and parameter updates all happen in
JAX arrays without device transfer.

Pipeline:
  Phase 1: (optional) Behavioral cloning warm-start
  Phase 2: Pure-JAX PPO on JaxHopperVecEnv (4 stages)

Key improvements over v5:
  1. No PyTorch/SB3 — eliminates JAX↔NumPy↔PyTorch transfer bottleneck
  2. Target: 1-5M sustained sps on RTX 3090 (vs 43K sps with SB3)
  3. All JAX arrays stay on device throughout training

Usage:
    cd backend
    uv run python scripts/train_v6_jax.py [--n-envs 1024] [--skip-bc] [--resume PATH]
"""

import sys
import json
import time
import signal
import argparse
from pathlib import Path
from datetime import datetime
from collections import deque

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
    1.0,   # landing_precision
    1.0,   # landing_high
    2.0,   # liftoff
    0.5,   # liftoff_fast
]

# 4-stage curriculum (matches v5)
STAGES = [
    {
        "name": "Phase 2a: Easy (all scenarios, small range)",
        "steps": 50_000_000,
        "lr": 3e-4,
        "max_episode_steps": 500,
        "alt_range": (5.0, 30.0),
        "horiz_range": (-15.0, 15.0),
        "vel_range": (-1.5, 1.5),
        "init_tilt_range": (-0.15, 0.15),
        "init_fuel_range": (0.6, 1.0),
        "scenarios": ALL_SCENARIOS,
        "scenario_weights": SCENARIO_WEIGHTS,
    },
    {
        "name": "Phase 2b: Medium (expanded range)",
        "steps": 100_000_000,
        "lr": 2e-4,
        "max_episode_steps": 500,
        "alt_range": (5.0, 50.0),
        "horiz_range": (-25.0, 25.0),
        "vel_range": (-3.0, 3.0),
        "init_tilt_range": (-0.20, 0.20),
        "init_fuel_range": (0.5, 1.0),
        "scenarios": ALL_SCENARIOS,
        "scenario_weights": SCENARIO_WEIGHTS,
    },
    {
        "name": "Phase 2c: Hard (full range)",
        "steps": 200_000_000,
        "lr": 1e-4,
        "max_episode_steps": 500,
        "alt_range": (5.0, 80.0),
        "horiz_range": (-30.0, 30.0),
        "vel_range": (-5.0, 5.0),
        "init_tilt_range": (-0.25, 0.25),
        "init_fuel_range": (0.4, 1.0),
        "scenarios": ALL_SCENARIOS,
        "scenario_weights": SCENARIO_WEIGHTS,
    },
    {
        "name": "Phase 2d: Consolidate + smoothness",
        "steps": 150_000_000,
        "lr": 5e-5,
        "max_episode_steps": 500,
        "alt_range": (5.0, 80.0),
        "horiz_range": (-30.0, 30.0),
        "vel_range": (-5.0, 5.0),
        "init_tilt_range": (-0.25, 0.25),
        "init_fuel_range": (0.4, 1.0),
        "scenarios": ALL_SCENARIOS,
        "scenario_weights": SCENARIO_WEIGHTS,
    },
]

# Evaluation configs
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


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

_shutdown = False


def _signal_handler(signum, frame):
    global _shutdown
    _shutdown = True
    print("\n[!] Shutdown requested, finishing current rollout...")


def update_env(env, trainer, stage):
    """Update the JAX hopper env's ranges and trigger recompile."""
    p = env.params
    p["hopper_alt_lo"] = stage["alt_range"][0]
    p["hopper_alt_hi"] = stage["alt_range"][1]
    p["hopper_horiz_lo"] = stage["horiz_range"][0]
    p["hopper_horiz_hi"] = stage["horiz_range"][1]
    p["hopper_vel_lo"] = stage["vel_range"][0]
    p["hopper_vel_hi"] = stage["vel_range"][1]
    p["init_tilt_lo"] = stage["init_tilt_range"][0]
    p["init_tilt_hi"] = stage["init_tilt_range"][1]
    p["init_fuel_lo"] = stage["init_fuel_range"][0]
    p["init_fuel_hi"] = stage["init_fuel_range"][1]
    p["max_episode_steps"] = stage["max_episode_steps"]

    # Recompile kernels with new params
    env._compile_kernels()


def hopper_eval(trainer, n_episodes=200, max_steps=500, **env_kwargs):
    """Fast batched evaluation using pure-JAX PPO inference."""
    from simulacra.jax_hopper_env import JaxHopperVecEnv

    eval_env = JaxHopperVecEnv(
        num_envs=n_episodes,
        use_guidance=True,
        max_tilt=0.25,
        max_episode_steps=max_steps,
        max_altitude=MAX_ALTITUDE,
        seed=42,
        **env_kwargs,
    )

    # Reset all eval envs
    obs = eval_env.reset()  # returns numpy
    obs = jnp.array(obs)
    done_mask = np.zeros(n_episodes, dtype=bool)
    arrived = np.zeros(n_episodes, dtype=bool)
    landed = np.zeros(n_episodes, dtype=bool)
    crashed = np.zeros(n_episodes, dtype=bool)

    for _ in range(max_steps):
        # Deterministic action
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
    parser = argparse.ArgumentParser(description="v6 Hopper training — pure JAX PPO")
    parser.add_argument("--skip-bc", action="store_true",
                        help="Skip behavioral cloning warm-start")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint directory")
    parser.add_argument("--n-envs", type=int, default=1024,
                        help="Number of parallel JAX envs (default: 1024)")
    parser.add_argument("--start-stage", type=int, default=0,
                        help="Skip to this stage index (0=2a, 1=2b, 2=2c, 3=2d)")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="Rollout length per env (default: 2048)")
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="Minibatch size (default: 4096)")
    parser.add_argument("--sb3-weights", type=str, default=None,
                        help="Load weights from SB3 model .zip (e.g. v5 checkpoint)")
    args = parser.parse_args()

    from simulacra.jax_ppo import PPOTrainer, PPOConfig
    from simulacra.jax_hopper_env import JaxHopperVecEnv

    save_dir = Path("runs/v6_hopper")
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "checkpoints").mkdir(exist_ok=True)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Report JAX backend
    devices = jax.devices()
    backend = devices[0].platform if devices else "cpu"
    print(f"JAX backend: {backend} ({len(devices)} device(s))")
    for d in devices:
        print(f"  {d}")

    total_steps = sum(s["steps"] for s in STAGES)
    print("=" * 60)
    print("v6 Hopper Training — Pure JAX PPO")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Network: [256, 256] (actor + critic)")
    print(f"Envs: {args.n_envs} (JAX vectorized)")
    print(f"Rollout: {args.n_steps} steps/env, batch {args.batch_size}")
    print(f"Total PPO: {total_steps / 1e6:.0f}M steps across {len(STAGES)} stages")
    print(f"Scenarios: {len(ALL_SCENARIOS)} types")
    print("=" * 60)

    # Save config
    config_dict = {
        "name": "v6_hopper",
        "description": "Pure JAX PPO (no SB3/PyTorch), R7 reward, 10 scenarios",
        "network": {"pi": [256, 256], "vf": [256, 256]},
        "n_envs": args.n_envs,
        "ppo": {
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        },
        "stages": [
            {"name": s["name"], "steps": s["steps"], "lr": s["lr"],
             "ep_steps": s["max_episode_steps"],
             "alt": list(s["alt_range"]),
             "horiz": list(s["horiz_range"])}
            for s in STAGES
        ],
        "total_steps": total_steps,
        "backend": backend,
        "started": datetime.now().isoformat(),
    }
    with open(save_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Create env from Stage 1 config
    s0 = STAGES[0]
    env = JaxHopperVecEnv(
        num_envs=args.n_envs,
        scenarios=s0["scenarios"],
        scenario_weights=s0["scenario_weights"],
        use_guidance=True,
        max_tilt=0.25,
        max_episode_steps=s0["max_episode_steps"],
        max_altitude=MAX_ALTITUDE,
        alt_range=s0["alt_range"],
        horiz_range=s0["horiz_range"],
        vel_range=s0["vel_range"],
        init_tilt_range=s0["init_tilt_range"],
        init_fuel_range=s0["init_fuel_range"],
    )

    # Create PPO trainer
    ppo_config = PPOConfig(
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        lr=s0["lr"],
    )
    trainer = PPOTrainer(env, ppo_config, seed=0)
    print(f"Network params: {trainer.get_param_count():,}")

    # Resume from checkpoint
    if args.resume:
        print(f"\nResuming from {args.resume}")
        trainer.load(Path(args.resume))

    # Load SB3 weights (e.g. from v5 training)
    if args.sb3_weights:
        print(f"\nLoading SB3 weights from {args.sb3_weights}")
        trainer.load_sb3_weights(Path(args.sb3_weights))

    # ── Phase 1: Behavioral cloning (optional) ──
    if not args.skip_bc and not args.resume and not args.sb3_weights:
        demo_path = Path("runs/pid_demos/demos.npz")
        if demo_path.exists():
            print(f"\n{'='*60}")
            print("  Phase 1: Behavioral Cloning from PID")
            print(f"{'='*60}")
            bc_start = time.time()
            _behavioral_cloning_jax(trainer, demo_path, n_epochs=10, lr=1e-3)
            bc_elapsed = (time.time() - bc_start) / 60
            print(f"  BC completed in {bc_elapsed:.1f} min")

            # Eval after BC
            print("\n  --- Eval after BC ---")
            full_eval(trainer, save_dir, "Phase 1: BC")
            trainer.save(save_dir / "bc_model")
        else:
            print(f"\n  No PID demos found at {demo_path}, skipping BC.")
            print(f"  Run: uv run python scripts/collect_pid_demos.py")

    # ── Phase 2: PPO training ──
    train_start = time.time()
    shutdown_flag = [False]

    for stage_idx, stage in enumerate(STAGES):
        if _shutdown or shutdown_flag[0]:
            break
        if stage_idx < args.start_stage:
            print(f"\n  Skipping {stage['name']} (--start-stage={args.start_stage})")
            continue

        stage_start = time.time()
        lr = stage["lr"]
        print(f"\n{'='*60}")
        print(f"  {stage['name']}")
        print(f"  Steps: {stage['steps']/1e6:.0f}M | LR: {lr} | "
              f"Ep: {stage['max_episode_steps']} steps")
        print(f"  Alt: {stage['alt_range']} | Horiz: {stage['horiz_range']}")
        print(f"{'='*60}")

        # Update env ranges (recompile kernels)
        if stage_idx > 0 or args.start_stage > 0:
            update_env(env, trainer, stage)

        # Set learning rate
        trainer.set_lr(lr)

        # Clear recent outcomes for fresh tracking
        trainer._recent_outcomes.clear()

        # Training log entries for this stage
        log_entries = []

        # Set shutdown flag
        shutdown_flag[0] = _shutdown

        try:
            trainer.train(
                total_steps=stage["steps"],
                save_dir=save_dir,
                save_freq=max(1, 10_000_000),
                stage_name=stage["name"],
                log_entries=log_entries,
                shutdown_flag=shutdown_flag,
            )
        except KeyboardInterrupt:
            print("\n  Interrupted.")
            _shutdown = True
            break

        # Save training log
        log_path = save_dir / f"training_log_stage{stage_idx}.jsonl"
        with open(log_path, "w") as f:
            for e in log_entries:
                f.write(json.dumps(e) + "\n")

        stage_elapsed = (time.time() - stage_start) / 60
        print(f"\n  Stage completed in {stage_elapsed:.1f} min")

        # Save stage checkpoint
        trainer.save(save_dir / f"stage{stage_idx}_model")

        # Evaluate
        print(f"\n  --- Eval after {stage['name']} ---")
        full_eval(trainer, save_dir, stage["name"])

    # Final
    total_elapsed = (time.time() - train_start) / 60
    trainer.save(save_dir / "final_model")
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION (trained {total_elapsed:.0f} min)")
    print(f"{'='*60}")
    final = full_eval(trainer, save_dir, "FINAL")

    # Summary
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


# ──────────────────────────────────────────────────────────────
# Behavioral cloning in JAX
# ──────────────────────────────────────────────────────────────

def _behavioral_cloning_jax(trainer, demo_path, n_epochs=10, lr=1e-3):
    """Pre-train policy on PID demonstrations using JAX (no PyTorch).

    Performs supervised learning: minimize MSE between policy mean and
    PID-recorded actions.
    """
    import optax

    data = np.load(demo_path)
    obs_all = jnp.array(data["obs"])
    act_all = jnp.array(data["actions"])
    n_samples = len(obs_all)

    print(f"  BC training: {n_samples:,} samples, {n_epochs} epochs")
    print(f"  Obs shape: {obs_all.shape}, Action shape: {act_all.shape}")

    # BC optimizer (separate from PPO optimizer)
    bc_optimizer = optax.adam(lr)
    bc_opt_state = bc_optimizer.init(trainer.train_state.params)

    @jax.jit
    def bc_step(params, opt_state, obs_batch, act_batch):
        def loss_fn(p):
            action_mean, _, _ = trainer.network.apply({"params": p}, obs_batch)
            return jnp.mean((action_mean - act_batch) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = bc_optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    batch_size = 256
    params = trainer.train_state.params
    rng = jax.random.PRNGKey(42)

    for epoch in range(n_epochs):
        rng, shuffle_key = jax.random.split(rng)
        indices = jax.random.permutation(shuffle_key, n_samples)

        total_loss = 0.0
        n_batches = 0
        for start in range(0, n_samples - batch_size + 1, batch_size):
            batch_idx = indices[start:start + batch_size]
            obs_b = obs_all[batch_idx]
            act_b = act_all[batch_idx]

            params, bc_opt_state, loss = bc_step(params, bc_opt_state, obs_b, act_b)
            total_loss += float(loss)
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        print(f"    Epoch {epoch + 1}/{n_epochs}: loss={avg_loss:.6f}")

    # Write BC params back to trainer
    trainer.train_state = trainer.train_state.replace(params=params)
    # Recompile with updated params
    trainer._compile()

    print(f"  BC training complete.")


if __name__ == "__main__":
    main()
