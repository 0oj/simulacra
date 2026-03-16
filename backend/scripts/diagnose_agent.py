"""
Diagnose trained RL agent behavior by recording full episode trajectories.

Runs N episodes with the best model and logs per-step data:
  altitude, velocity, throttle, tilt, outcome, etc.

Prints a summary table and detailed trajectory for each failure mode.

Usage:
    cd backend && uv run python scripts/diagnose_agent.py
"""

import sys
import math
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from simulacra.gym_env import RocketLandingEnvV2


class ThrottleOnlyWrapper(gym.Wrapper):
    def __init__(self, env, gimbal_gain=0.5, gimbal_clamp=0.3):
        super().__init__(env)
        self.gimbal_gain = gimbal_gain
        self.gimbal_clamp = gimbal_clamp
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
        )

    def step(self, action):
        throttle = float(action[0])
        obs = self._last_obs
        wy, wz = float(obs[11]), float(obs[12])
        go = np.clip(-self.gimbal_gain * wz, -self.gimbal_clamp, self.gimbal_clamp)
        gi = np.clip(-self.gimbal_gain * wy, -self.gimbal_clamp, self.gimbal_clamp)
        full = np.array([throttle, go, gi], dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(full)
        self._last_obs = obs
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info


def run_episodes(model, n=50, **env_kwargs):
    env = ThrottleOnlyWrapper(RocketLandingEnvV2(**env_kwargs))
    episodes = []

    for ep in range(n):
        obs, info = env.reset(seed=ep + 7000)
        init_alt = float(obs[1])
        init_vy = float(obs[4])

        trajectory = []
        done = False
        step = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, t, tr, info = env.step(action)
            done = t or tr
            step += 1

            # Get actual tilt from the underlying env's info dict
            # The ThrottleOnlyWrapper passes through to RocketLandingEnvV2.step()
            # which includes tilt_deg and fuel in info
            trajectory.append({
                "step": step,
                "time": step * 0.02,
                "alt": float(obs[1]),
                "vy": float(obs[4]),
                "vx": float(obs[3]),
                "vz": float(obs[5]),
                "throttle": float(action[0]),
                "tilt_deg": info.get("tilt_deg", 0),
                "fuel": info.get("fuel", -1),
                "horiz_dist": info.get("horiz_dist", math.sqrt(float(obs[0])**2 + float(obs[2])**2)),
                "on_ground": info.get("on_ground", False),
            })

        outcome = info.get("outcome", "timeout" if step >= 1000 else "terminated")
        # Get max tilt during episode
        max_tilt = max(s["tilt_deg"] for s in trajectory)
        min_fuel = min(s["fuel"] for s in trajectory)
        max_alt = max(s["alt"] for s in trajectory)

        episodes.append({
            "ep": ep,
            "init_alt": init_alt,
            "init_vy": init_vy,
            "outcome": outcome,
            "steps": step,
            "duration": step * 0.02,
            "final_alt": trajectory[-1]["alt"],
            "final_vy": trajectory[-1]["vy"],
            "final_tilt": trajectory[-1]["tilt_deg"],
            "max_tilt": max_tilt,
            "min_fuel": min_fuel,
            "max_alt": max_alt,
            "landing_vel": info.get("landing_vel"),
            "landing_tilt": info.get("landing_tilt"),
            "trajectory": trajectory,
        })

    env.close()
    return episodes


def print_summary(episodes):
    outcomes = defaultdict(list)
    for ep in episodes:
        outcomes[ep["outcome"]].append(ep)

    print(f"\n{'='*70}")
    print(f"OUTCOME SUMMARY ({len(episodes)} episodes)")
    print(f"{'='*70}")
    for outcome, eps in sorted(outcomes.items()):
        print(f"  {outcome:12s}: {len(eps):3d} ({100*len(eps)/len(episodes):.0f}%)")
    print()

    # Detail each outcome group
    for outcome, eps in sorted(outcomes.items()):
        print(f"\n--- {outcome.upper()} ({len(eps)} episodes) ---")

        if outcome == "landed":
            vels = [e["landing_vel"] for e in eps if e["landing_vel"] is not None]
            tilts = [e["landing_tilt"] for e in eps if e["landing_tilt"] is not None]
            durations = [e["duration"] for e in eps]
            if vels:
                print(f"  Landing vel:  mean={np.mean(vels):.2f} m/s, max={np.max(vels):.2f}")
            if tilts:
                print(f"  Landing tilt: mean={np.mean(tilts):.1f} deg, max={np.max(tilts):.1f}")
            print(f"  Duration:     mean={np.mean(durations):.1f}s, min={np.min(durations):.1f}s, max={np.max(durations):.1f}s")

        elif outcome == "crash":
            for e in eps[:5]:
                print(f"  ep{e['ep']:3d}: init_alt={e['init_alt']:.1f}m, init_vy={e['init_vy']:.1f}m/s, "
                      f"final_vy={e['final_vy']:.1f}m/s, tilt={e['final_tilt']:.1f}deg, "
                      f"steps={e['steps']}")

        elif outcome == "timeout":
            for e in eps[:5]:
                t = e["trajectory"]
                # Find min altitude reached
                min_alt = min(s["alt"] for s in t)
                # Average throttle in last 100 steps
                avg_throttle_end = np.mean([s["throttle"] for s in t[-100:]])
                avg_throttle_all = np.mean([s["throttle"] for s in t])
                print(f"  ep{e['ep']:3d}: init_alt={e['init_alt']:.1f}m, final_alt={e['final_alt']:.1f}m, "
                      f"min_alt={min_alt:.1f}m, avg_throttle={avg_throttle_all:.3f} "
                      f"(last100={avg_throttle_end:.3f})")

        else:  # terminated (tilt, fuel, etc.)
            # Classify termination reason
            tilt_term = [e for e in eps if e["max_tilt"] > 57]
            fuel_term = [e for e in eps if e["min_fuel"] <= 0.01]
            alt_term = [e for e in eps if e["max_alt"] > 99]
            print(f"  Breakdown: {len(tilt_term)} tilt, {len(fuel_term)} fuel, {len(alt_term)} altitude")
            for e in eps[:8]:
                reason = "TILT" if e["max_tilt"] > 57 else "FUEL" if e["min_fuel"] <= 0.01 else "ALT" if e["max_alt"] > 99 else "???"
                print(f"  ep{e['ep']:3d}: {reason:4s} init_alt={e['init_alt']:.1f}m, final_alt={e['final_alt']:.1f}m, "
                      f"tilt={e['final_tilt']:.1f}°, max_tilt={e['max_tilt']:.1f}°, fuel={e['min_fuel']:.2f}, "
                      f"steps={e['steps']}")


def print_trajectory_samples(episodes, outcome, n=3):
    """Print detailed trajectory for a few episodes of a given outcome."""
    eps = [e for e in episodes if e["outcome"] == outcome][:n]
    if not eps:
        return

    print(f"\n{'='*70}")
    print(f"SAMPLE TRAJECTORIES: {outcome.upper()}")
    print(f"{'='*70}")

    for e in eps:
        t = e["trajectory"]
        print(f"\nEpisode {e['ep']}: init_alt={e['init_alt']:.1f}m, init_vy={e['init_vy']:.1f}m/s → {e['outcome']}")
        print(f"  {'time':>5s}  {'alt':>6s}  {'vy':>6s}  {'thr':>5s}  {'tilt':>5s}  {'hdist':>5s}")
        print(f"  {'─'*5}  {'─'*6}  {'─'*6}  {'─'*5}  {'─'*5}  {'─'*5}")

        # Sample at key moments: every 1s, plus last 20 steps
        sample_steps = set()
        for s in t:
            if s["step"] % 50 == 0:  # every 1s
                sample_steps.add(s["step"])
        for s in t[-20:]:
            sample_steps.add(s["step"])
        # First 5 steps
        for s in t[:5]:
            sample_steps.add(s["step"])

        for s in t:
            if s["step"] in sample_steps:
                print(f"  {s['time']:5.1f}s  {s['alt']:6.1f}m  {s['vy']:+6.1f}  {s['throttle']:5.3f}  "
                      f"{s['tilt_deg']:5.1f}°  {s['horiz_dist']:5.1f}m")


def main():
    from stable_baselines3 import PPO

    # Load best model
    model_paths = [
        ("PPO finetune L3", Path("runs/ppo_finetune_l3/best_model/best_model.zip")),
        ("PPO curriculum", Path("runs/throttle_ppo/best_model/best_model.zip")),
    ]

    env_kwargs = dict(
        init_altitude_range=(10., 60.),
        init_velocity_range=(-8., 0.),
        init_horizontal_range=(-5., 5.),
        init_tilt_range=(-0.15, 0.15),
    )

    for name, path in model_paths:
        if not path.exists():
            print(f"Skipping {name}: {path} not found")
            continue

        print(f"\n{'#'*70}")
        print(f"# {name}")
        print(f"# Model: {path}")
        print(f"{'#'*70}")

        model = PPO.load(str(path))
        episodes = run_episodes(model, n=50, **env_kwargs)

        print_summary(episodes)
        print_trajectory_samples(episodes, "timeout", n=3)
        print_trajectory_samples(episodes, "crash", n=3)
        print_trajectory_samples(episodes, "landed", n=2)


if __name__ == "__main__":
    main()
