"""Diagnose what's happening in the landing environment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import math
import numpy as np
from simulacra.gym_env import RocketLandingEnvV2

env = RocketLandingEnvV2(
    init_altitude_range=(2.0, 5.0),
    init_velocity_range=(-1.0, 0.0),
    init_horizontal_range=(-0.5, 0.5),
    init_tilt_range=(-0.02, 0.02),
    init_fuel_range=(0.9, 1.0),
)

# Run episodes with constant hover thrust and track what terminates them
print("=== Hover thrust episodes ===")
for trial in range(10):
    obs, _ = env.reset(seed=trial)
    done = False
    steps = 0
    mass = 60 + 80 * 0.8
    hover = mass * 9.81 / 2000.0

    while not done:
        action = np.array([hover, 0.0, 0.0], dtype=np.float32)
        obs, r, term, trunc, info = env.step(action)
        done = term or trunc
        steps += 1

    state = env._kernel.rocket.state
    ground = env._kernel.ground_contact
    tilt = 2 * math.acos(min(1, abs(obs[6])))
    outcome = info.get("outcome", "")
    cause = ""
    if ground.on_ground:
        cause = f"ground (crashed={ground.crashed}, tipped={ground.tipped_over})"
    elif tilt > 1.0:
        cause = f"tilt={math.degrees(tilt):.1f}deg"
    elif state.position.y > 100:
        cause = f"too_high (alt={state.position.y:.1f})"
    elif env._kernel.rocket.mass_model.fuel_fraction <= 0:
        cause = "no_fuel"
    elif steps >= 1000:
        cause = "timeout"
    else:
        cause = f"unknown (alt={state.position.y:.1f}, tilt={math.degrees(tilt):.1f})"

    print(f"  Trial {trial}: steps={steps}, alt={state.position.y:.2f}m, "
          f"vel={state.velocity.norm:.2f}m/s, outcome={outcome}, cause={cause}")

# Run with max thrust
print("\n=== Max thrust episodes ===")
for trial in range(5):
    obs, _ = env.reset(seed=trial)
    done = False
    steps = 0

    while not done:
        action = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        obs, r, term, trunc, info = env.step(action)
        done = term or trunc
        steps += 1

    state = env._kernel.rocket.state
    print(f"  Trial {trial}: steps={steps}, alt={state.position.y:.1f}m, "
          f"outcome={info.get('outcome','')}")

# Run with zero thrust (free fall)
print("\n=== Free fall episodes ===")
for trial in range(5):
    obs, _ = env.reset(seed=trial)
    done = False
    steps = 0

    while not done:
        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        obs, r, term, trunc, info = env.step(action)
        done = term or trunc
        steps += 1

    state = env._kernel.rocket.state
    ground = env._kernel.ground_contact
    print(f"  Trial {trial}: steps={steps}, vel={state.velocity.norm:.2f}m/s, "
          f"outcome={info.get('outcome','')}, on_ground={ground.on_ground}, "
          f"crashed={ground.crashed}")

# What about the landing speed threshold?
print(f"\n=== Landing thresholds ===")
print(f"  max_landing_speed: {env.max_landing_speed} m/s")
print(f"  max_landing_tilt: {math.degrees(env.max_landing_tilt):.1f} deg")
print(f"  max_tilt (termination): {math.degrees(env.max_tilt):.1f} deg")

# Try to manually land: free fall then brake
print("\n=== Manual brake landing ===")
for trial in range(10):
    obs, _ = env.reset(seed=trial)
    done = False
    steps = 0

    while not done:
        alt = obs[1]
        vy = obs[4]

        # Simple policy: brake when close to ground
        if alt < 2.0 and vy < -0.5:
            throttle = 0.8
        elif alt < 0.5:
            throttle = 0.4  # gentle near ground
        else:
            throttle = 0.0  # free fall

        action = np.array([throttle, 0.0, 0.0], dtype=np.float32)
        obs, r, term, trunc, info = env.step(action)
        done = term or trunc
        steps += 1

    outcome = info.get("outcome", "other")
    vel = info.get("velocity", 0) if isinstance(info.get("velocity"), (int, float)) else 0
    print(f"  Trial {trial}: steps={steps}, outcome={outcome}, "
          f"vel={info.get('landing_vel', info.get('velocity', '?'))}")

env.close()
