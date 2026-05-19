"""
Evaluate a trained model on RocketLandingEnv.

Usage:
    python scripts/eval_model.py runs/ppo_rajs/best_model/best_model.zip
    python scripts/eval_model.py runs/ppo_rajs/final_model.zip --episodes 100
"""

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from stable_baselines3 import PPO
from simulacra.gym_env import RocketLandingEnv


def evaluate(model_path: str, n_episodes: int = 50, render: bool = False):
    """Evaluate model and report statistics."""

    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)

    env = RocketLandingEnv()

    # Track metrics
    successes = 0
    total_rewards = []
    episode_lengths = []
    landing_velocities = []
    landing_positions = []

    print(f"Evaluating for {n_episodes} episodes...")

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        total_rewards.append(total_reward)
        episode_lengths.append(steps)

        # Check if successful (customize based on your reward structure)
        if info.get("is_success", False) or total_reward > 50:
            successes += 1

        # Track landing state if available
        if "final_velocity" in info:
            landing_velocities.append(info["final_velocity"])
        if "final_position" in info:
            landing_positions.append(info["final_position"])

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep + 1}/{n_episodes}: reward={total_reward:.2f}")

    env.close()

    # Report
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Episodes:     {n_episodes}")
    print(f"Success rate: {successes / n_episodes * 100:.1f}%")
    print(f"Mean reward:  {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Mean length:  {np.mean(episode_lengths):.1f} steps")

    if landing_velocities:
        print(f"Mean landing velocity: {np.mean(landing_velocities):.2f} m/s")
    if landing_positions:
        mean_pos = np.mean(landing_positions, axis=0)
        print(f"Mean landing position: x={mean_pos[0]:.2f}, z={mean_pos[2]:.2f}")

    return {
        "success_rate": successes / n_episodes,
        "mean_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "mean_length": np.mean(episode_lengths),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("model_path", type=str, help="Path to model .zip file")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes")
    parser.add_argument("--render", action="store_true", help="Render episodes")
    args = parser.parse_args()

    evaluate(args.model_path, args.episodes, args.render)


if __name__ == "__main__":
    main()
