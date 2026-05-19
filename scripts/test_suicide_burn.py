"""
Test suicide burn landing controller from realistic altitudes.

A suicide burn works by:
1. Free fall until exactly the right altitude
2. Apply maximum braking thrust to stop exactly at ground level
3. The key is computing the correct "brake ignition altitude"

Physics:
- Falling freely: builds up speed v = sqrt(2*g*h)
- Braking at max thrust: deceleration a = (T_max - m*g) / m
- Stopping distance from velocity v: d = v^2 / (2*a)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from simulacra.gym_env import RocketLandingEnvV2


def suicide_burn_policy(obs: np.ndarray, env: RocketLandingEnvV2, mass: float = None) -> np.ndarray:
    """
    Glide path controller for rocket landing.

    Instead of a suicide burn, use a velocity profile that depends on altitude:
    - High altitude: allow faster descent
    - Low altitude: slow descent
    - The "glide slope" defines target_vy as a function of altitude

    This avoids the oscillation issues of the pure suicide burn approach.

    Observation: [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
    """
    # Extract state
    alt = obs[1]  # altitude
    vy = obs[4]   # vertical velocity (negative when falling)
    wx, wy, wz = obs[10], obs[11], obs[12]  # angular velocity

    # Get mass - use provided value or estimate
    if mass is None:
        mass = 112.0

    g = 9.81
    max_thrust = env.max_thrust  # 2000 N

    # Hover throttle = mg / max_thrust
    hover_throttle = (mass * g) / max_thrust  # ~0.55

    # Define glide slope: target velocity based on altitude
    # v_target = -k * sqrt(alt) gives decreasing speed as we approach ground
    # Calibrate k so that we're going slowly near touchdown
    k = 1.5  # Steeper glide slope for faster descent

    # Minimum descent speed (avoid hovering)
    min_descent_speed = -0.8

    # Maximum descent speed (safety limit)
    max_descent_speed = -15.0

    # Target velocity profile
    if alt > 0.5:
        target_vy = max(max_descent_speed, min(-k * np.sqrt(alt), min_descent_speed))
    else:
        # Very close to ground - gentle target
        target_vy = -1.0

    # Terminal landing: below 0.4m with slow speed, cut engine to let gravity bring us down
    if alt < 0.4 and abs(vy) < 2.5:
        throttle = 0.0
    else:
        # P control on velocity
        vy_error = target_vy - vy  # negative if we're too slow (need to descend faster)

        # Higher gains for more responsive control
        if alt < 2.0:
            Kp = 0.25  # Increased from 0.15
        elif alt < 10.0:
            Kp = 0.30  # Increased from 0.20
        else:
            Kp = 0.35  # Increased from 0.25

        throttle = hover_throttle + Kp * vy_error
        throttle = np.clip(throttle, 0.0, 1.0)

    # Angular rate damping for stability (original simple approach)
    gimbal_outer = np.clip(-0.5 * wz, -0.3, 0.3)
    gimbal_inner = np.clip(-0.5 * wy, -0.3, 0.3)

    return np.array([throttle, gimbal_outer, gimbal_inner], dtype=np.float32)


def debug_trajectory(n_steps: int = 500, altitude: float = 30.0):
    """Debug a single trajectory to understand the physics."""
    env = RocketLandingEnvV2(
        init_altitude_range=(altitude, altitude),
        init_velocity_range=(0.0, 0.0),  # Start stationary
        init_horizontal_range=(0.0, 0.0),
        init_tilt_range=(0.0, 0.0),
    )

    obs, info = env.reset(seed=42)

    # Get actual mass from kernel
    actual_mass = env._kernel.rocket.mass_model.total_mass
    g = 9.81
    max_thrust = env.max_thrust

    print(f"Actual mass: {actual_mass:.1f} kg")
    print(f"Max thrust: {max_thrust:.0f} N")
    print(f"Weight: {actual_mass * g:.0f} N")
    print(f"Net upward force at full thrust: {max_thrust - actual_mass * g:.0f} N")
    print(f"Max decel: {(max_thrust - actual_mass * g) / actual_mass:.2f} m/s^2")
    print()

    print(f"{'Step':>4} {'Alt':>7} {'Vy':>7} {'Target':>7} {'Throttle':>8}")
    print("-" * 50)

    for step in range(n_steps):
        alt = obs[1]
        vy = obs[4]

        # Compute target velocity (same as in policy)
        k = 1.5
        if alt > 0.5:
            target_vy = max(-15.0, min(-k * np.sqrt(alt), -0.8))
        else:
            target_vy = -1.0

        action = suicide_burn_policy(obs, env, mass=actual_mass)
        throttle = action[0]

        if step % 10 == 0 or alt < 5:
            print(f"{step:4d} {alt:7.2f} {vy:7.2f} {target_vy:7.2f} {throttle:8.2f}")

        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            outcome = info.get("outcome", "timeout")
            print(f"\nOutcome: {outcome}")
            if outcome == "crash":
                print(f"Impact velocity: {info.get('velocity', '?')}")
            break

    env.close()


def test_controller(n_episodes: int = 20, altitude_range=(20.0, 60.0)):
    """Test the suicide burn controller."""
    env = RocketLandingEnvV2(
        init_altitude_range=altitude_range,
        init_velocity_range=(-4.0, 0.0),
        init_horizontal_range=(-2.0, 2.0),
        init_tilt_range=(-0.05, 0.05),
    )

    successes = 0
    crashes = 0
    timeouts = 0

    print(f"Testing suicide burn controller: {altitude_range[0]}-{altitude_range[1]}m")
    print("=" * 60)

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        init_alt = info["init_altitude"]
        # Get actual mass for this episode
        mass = env._kernel.rocket.mass_model.total_mass
        done = False
        step = 0

        while not done:
            action = suicide_burn_policy(obs, env, mass=mass)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            step += 1
            # Update mass as fuel burns
            mass = env._kernel.rocket.mass_model.total_mass

        outcome = info.get("outcome", "timeout")
        if outcome == "landed":
            successes += 1
            print(f"Ep {ep+1:3d}: LANDED from {init_alt:5.1f}m (vel={info.get('landing_vel', 0):.2f} m/s)")
        elif outcome == "crash":
            crashes += 1
            print(f"Ep {ep+1:3d}: CRASH  from {init_alt:5.1f}m")
        else:
            timeouts += 1
            final_alt = obs[1]
            print(f"Ep {ep+1:3d}: TIMEOUT from {init_alt:5.1f}m (final alt={final_alt:.1f}m)")

    env.close()

    print("=" * 60)
    print(f"Success rate: {100*successes/n_episodes:.1f}%")
    print(f"Crashes: {crashes}, Timeouts: {timeouts}")

    return successes / n_episodes


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "debug":
        debug_trajectory(altitude=30.0)
    else:
        # Test at multiple altitude ranges
        ranges = [
            (10.0, 20.0),
            (20.0, 40.0),
            (40.0, 60.0),
            (60.0, 80.0),
        ]

        for r in ranges:
            print()
            test_controller(n_episodes=20, altitude_range=r)
