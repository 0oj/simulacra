#!/usr/bin/env python3
"""
Demo script to run the RocketLandingEnv and visualize the results.
"""

import numpy as np
from simulacra.gym_env import RocketLandingEnv


def run_episode(env, policy="hover", seed=42, render_interval=50):
    """Run a single episode and print stats."""
    obs, info = env.reset(seed=seed)

    print(f"Initial state:")
    print(f"  Altitude: {info['init_altitude']:.1f} m")
    print(f"  Fuel: {info['init_fuel']*100:.0f}%")
    print()

    total_reward = 0.0
    step = 0

    while True:
        # Simple policy
        if policy == "hover":
            # Try to hover: ~50% throttle, no gimbal
            action = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        elif policy == "land":
            # Simple landing: adjust throttle based on velocity
            vel_y = obs[4]  # vertical velocity
            alt = obs[1]    # altitude

            if alt < 5.0:
                # Near ground: cut thrust
                throttle = 0.2
            elif vel_y < -5.0:
                # Falling fast: more thrust
                throttle = 0.8
            elif vel_y < -2.0:
                # Falling slow: medium thrust
                throttle = 0.6
            else:
                # Slow or rising: less thrust
                throttle = 0.4

            action = np.array([throttle, 0.0, 0.0], dtype=np.float32)
        else:
            # Random
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        if step % render_interval == 0:
            print(f"Step {step:4d}: alt={info['altitude']:6.1f}m, "
                  f"vel={info['velocity']:5.1f}m/s, "
                  f"tilt={info['tilt_deg']:5.1f}°, "
                  f"fuel={info['fuel']*100:4.0f}%, "
                  f"reward={reward:7.2f}")

        if terminated or truncated:
            break

    print()
    print(f"Episode finished after {step} steps")
    print(f"  Total reward: {total_reward:.1f}")
    print(f"  Final altitude: {info['altitude']:.2f} m")
    print(f"  Final velocity: {info['velocity']:.2f} m/s")
    print(f"  Final tilt: {info['tilt_deg']:.1f}°")
    print(f"  On ground: {info['on_ground']}")
    if 'outcome' in info:
        print(f"  Outcome: {info['outcome']}")

    return total_reward


def main():
    print("=" * 60)
    print("RocketLandingEnv Demo with AttitudeEstimator")
    print("=" * 60)
    print()

    # Create environment
    env = RocketLandingEnv(
        init_altitude_range=(30.0, 50.0),
        init_velocity_range=(-3.0, 0.0),
        init_tilt_range=(-0.1, 0.1),
        max_episode_steps=500,
    )

    print("Observation space:", env.observation_space.shape)
    print("Action space:", env.action_space.shape)
    print()

    # Run with simple landing policy
    print("-" * 60)
    print("Running with simple landing policy...")
    print("-" * 60)
    run_episode(env, policy="land", seed=42, render_interval=25)

    print()
    print("-" * 60)
    print("Running with hover policy...")
    print("-" * 60)
    run_episode(env, policy="hover", seed=123, render_interval=25)

    print()
    print("-" * 60)
    print("Running with random actions...")
    print("-" * 60)
    run_episode(env, policy="random", seed=456, render_interval=25)

    env.close()


if __name__ == "__main__":
    main()
