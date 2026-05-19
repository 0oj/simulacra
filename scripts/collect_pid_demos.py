#!/usr/bin/env python3
"""
Collect behavioral cloning demonstrations from the PID hopper controller.

Runs the PID controller through diverse waypoint scenarios, recording
(observation, action) pairs in the same format as HopperTrackingEnv / GuidanceWrapper.

Observations: 19-dim [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz,
                       dx, dy, dz, dist, vel_toward, brake_margin]
Actions:      3-dim  [throttle (0-1), tilt_x (-1 to 1), tilt_z (-1 to 1)]

Usage:
    cd backend
    uv run python scripts/collect_pid_demos.py
    uv run python scripts/collect_pid_demos.py --episodes 20000 --max-tilt 0.25
"""

import sys
import math
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from simulacra.noumena import (
    Rocket, RocketState, Environment, Kernel,
    MassModel, get_upright_orientation,
)
from simulacra.noumena.mass_model import MassComponent, FuelComponent
from simulacra.phenomena.actuators import Actuators, GimbalActuator, ThrottleActuator
from simulacra.phenomena.sensors import Sensors
from simulacra.physics import RocketVec3, WorldVec3
from simulacra.onboard.hopper_controller import HopperController, HopperConfig, Waypoint


# ──────────────────────────────────────────────────────────────
# Constants (matching HopperTrackingEnv and GuidanceWrapper)
# ──────────────────────────────────────────────────────────────

DT = 0.02
DRY_MASS = 60.0
FUEL_MASS = 80.0
LENGTH = 4.0
MAX_THRUST = 2000.0
BURN_RATE = 0.68
A_MAX = 4.0  # matching HopperTrackingEnv.A_MAX for brake_margin
MAX_EPISODE_STEPS = 1000


# ──────────────────────────────────────────────────────────────
# Simulation factory
# ──────────────────────────────────────────────────────────────

def make_sim(init_pos, init_vel, init_fuel_frac=1.0):
    """Create simulation components with the given initial state."""
    mass_model = MassModel(components=[
        MassComponent("dry", DRY_MASS, RocketVec3(LENGTH / 2, 0, 0)),
        FuelComponent("fuel", mass=FUEL_MASS * init_fuel_frac,
                      position=RocketVec3(LENGTH * 0.3, 0, 0),
                      full_mass=FUEL_MASS, burn_rate=BURN_RATE),
    ])
    props = mass_model.to_rocket_props(
        length=LENGTH,
        thrust_application_offset=RocketVec3(0, 0, 0),
    )
    state = RocketState(
        position=WorldVec3(float(init_pos[0]), float(init_pos[1]),
                           float(init_pos[2])),
        velocity=WorldVec3(float(init_vel[0]), float(init_vel[1]),
                           float(init_vel[2])),
        orientation=get_upright_orientation(),
        angular_velocity_body=RocketVec3(0.0, 0.0, 0.0),
        thrust_body=RocketVec3(0.0, 0.0, 0.0),
    )
    rocket = Rocket(props=props, state=state, mass_model=mass_model)
    environment = Environment(
        gravity=WorldVec3(0.0, -9.81, 0.0),
        wind=WorldVec3(0.0, 0.0, 0.0),
        air_density=1.225,
    )
    kernel = Kernel(rocket, environment)

    # Set ground contact for on-ground starts
    if init_pos[1] <= 0.0:
        kernel.ground_contact.on_ground = True

    gimbal = GimbalActuator(rocket=rocket, max_outer_angle=0.26,
                            max_inner_angle=0.26, max_rate=1.0, tau=0.05)
    throttle = ThrottleActuator(rocket=rocket, max_thrust_rate=MAX_THRUST * 2,
                                min_thrust=0.0)
    actuators = Actuators(gimbal=gimbal, throttle=throttle)
    sensors = Sensors(kernel)

    config = HopperConfig(
        dry_mass=DRY_MASS,
        full_fuel_mass=FUEL_MASS,
        max_thrust=MAX_THRUST,
        gravity=9.81,
    )

    return kernel, actuators, sensors, config


# ──────────────────────────────────────────────────────────────
# Rotation helpers (matching hopper_controller.py)
# ──────────────────────────────────────────────────────────────

def _rotate_body_to_world(quat, vec_body):
    """quat = (w, x, y, z)"""
    w, x, y, z = quat
    qv = np.array([x, y, z])
    v = np.asarray(vec_body, dtype=np.float64)
    c1 = np.cross(qv, v)
    c2 = np.cross(qv, c1)
    return v + 2.0 * w * c1 + 2.0 * c2


# ──────────────────────────────────────────────────────────────
# Observation builder (matching HopperTrackingEnv._get_obs)
# ──────────────────────────────────────────────────────────────

def build_obs(sensors, target):
    """Build 19-dim observation matching HopperTrackingEnv._get_obs."""
    gps = sensors.GPS.read()
    alt = sensors.Barometer.read()
    vel = sensors.VelocitySensor.read()
    orient = sensors.OrientationSensor.read()
    omega = sensors.GyroSensor.read()

    qw, qx, qy, qz = orient.as_array()
    pos = np.array([gps.x, alt, gps.z])
    vel_vec = np.array([vel.x, vel.y, vel.z])

    # Goal-relative
    goal_rel = target - pos

    # Features
    dist = np.linalg.norm(goal_rel)
    if dist > 0.01:
        dir_to_wp = goal_rel / dist
        vel_toward = float(np.dot(vel_vec, dir_to_wp))
    else:
        vel_toward = 0.0
    speed = np.linalg.norm(vel_vec)
    brake_margin = dist - speed ** 2 / (2 * A_MAX)

    return np.array([
        gps.x, alt, gps.z,
        vel.x, vel.y, vel.z,
        qw, qx, qy, qz,
        omega.x, omega.y, omega.z,
        goal_rel[0], goal_rel[1], goal_rel[2],
        dist, vel_toward, brake_margin,
    ], dtype=np.float32)


# ──────────────────────────────────────────────────────────────
# Action extraction (reverse-map PID commands to guidance format)
# ──────────────────────────────────────────────────────────────

def extract_action(kernel, max_tilt):
    """
    Extract [throttle_norm, tilt_x_norm, tilt_z_norm] from current rocket state.

    - throttle_norm = thrust_magnitude / MAX_THRUST, clipped to [0, 1]
    - tilt_x, tilt_z: derived from nose world direction
      If nose_world = [nx, ny, nz] and ny > 0.1:
        tilt_x = nx / ny, tilt_z = nz / ny (pre-normalization tilt components)
      Divided by max_tilt and clamped to [-1, 1]
    """
    s = kernel.rocket.state

    # Throttle
    thrust_mag = s.thrust_body.norm
    throttle_norm = np.clip(thrust_mag / MAX_THRUST, 0.0, 1.0)

    # Tilt from nose direction
    quat = tuple(s.orientation.as_array())  # (w, x, y, z)
    nose_world = _rotate_body_to_world(quat, np.array([1.0, 0.0, 0.0]))

    if nose_world[1] > 0.1:
        # GuidanceWrapper desired = normalize([tilt_x * max_tilt, 1, tilt_z * max_tilt])
        # If the rocket is pointing in direction nose_world, the "tilt" that would produce
        # this direction is: tilt_x_raw = nx/ny, tilt_z_raw = nz/ny
        # Then tilt_x_norm = tilt_x_raw / max_tilt
        tilt_x_raw = nose_world[0] / nose_world[1]
        tilt_z_raw = nose_world[2] / nose_world[1]
        tilt_x_norm = np.clip(tilt_x_raw / max_tilt, -1.0, 1.0)
        tilt_z_norm = np.clip(tilt_z_raw / max_tilt, -1.0, 1.0)
    else:
        # Rocket is tilted far from vertical; clamp to max
        tilt_x_norm = np.clip(np.sign(nose_world[0]), -1.0, 1.0)
        tilt_z_norm = np.clip(np.sign(nose_world[2]), -1.0, 1.0)

    return np.array([throttle_norm, tilt_x_norm, tilt_z_norm], dtype=np.float32)


# ──────────────────────────────────────────────────────────────
# Scenario generators (single-waypoint episodes)
# ──────────────────────────────────────────────────────────────

def gen_vertical(rng):
    """Vertical transition: liftoff or altitude change."""
    start_y = rng.uniform(0.0, 30.0)
    target_y = rng.uniform(5.0, 30.0)
    while abs(target_y - start_y) < 3.0:
        target_y = rng.uniform(5.0, 30.0)
    start_vel_y = rng.uniform(-1.0, 1.0)
    # On-ground starts: zero velocity
    if start_y < 0.5:
        start_y = 0.0
        start_vel_y = 0.0
    return (
        np.array([0.0, start_y, 0.0]),
        np.array([0.0, start_vel_y, 0.0]),
        np.array([0.0, target_y, 0.0]),
    )


def gen_lateral(rng):
    """Lateral move at roughly constant altitude."""
    alt = rng.uniform(10.0, 30.0)
    start_x = rng.uniform(-7.0, 7.0)
    start_z = rng.uniform(-7.0, 7.0)
    target_x = rng.uniform(-15.0, 15.0)
    target_z = rng.uniform(-15.0, 15.0)
    while math.sqrt((target_x - start_x)**2 + (target_z - start_z)**2) < 3.0:
        target_x = rng.uniform(-15.0, 15.0)
        target_z = rng.uniform(-15.0, 15.0)
    vx = rng.uniform(-1.0, 1.0)
    vz = rng.uniform(-1.0, 1.0)
    return (
        np.array([start_x, alt, start_z]),
        np.array([vx, 0.0, vz]),
        np.array([target_x, alt, target_z]),
    )


def gen_diagonal(rng):
    """Full 3D transition."""
    start = np.array([
        rng.uniform(-7.0, 7.0),
        rng.uniform(2.0, 40.0),
        rng.uniform(-7.0, 7.0),
    ])
    target = np.array([
        rng.uniform(-15.0, 15.0),
        rng.uniform(5.0, 40.0),
        rng.uniform(-15.0, 15.0),
    ])
    while np.linalg.norm(target - start) < 5.0:
        target[1] = rng.uniform(5.0, 40.0)
    vel = np.array([rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5),
                    rng.uniform(-1.5, 1.5)])
    return start, vel, target


def gen_landing(rng):
    """Approach to ground (target y=0)."""
    start = np.array([
        rng.uniform(-10.0, 10.0),
        rng.uniform(10.0, 40.0),
        rng.uniform(-10.0, 10.0),
    ])
    target = np.array([0.0, 0.0, 0.0])
    vy = rng.uniform(-2.0, 0.0)
    return start, np.array([0.0, vy, 0.0]), target


def gen_liftoff(rng):
    """Take off from ground to target altitude."""
    target = np.array([
        rng.uniform(-5.0, 5.0),
        rng.uniform(5.0, 25.0),
        rng.uniform(-5.0, 5.0),
    ])
    return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), target


SCENARIO_GENERATORS = {
    "vertical": gen_vertical,
    "lateral": gen_lateral,
    "diagonal": gen_diagonal,
    "landing": gen_landing,
    "liftoff": gen_liftoff,
}


# ──────────────────────────────────────────────────────────────
# Episode runner
# ──────────────────────────────────────────────────────────────

def run_episode(start_pos, start_vel, target, max_tilt, max_steps=None):
    """
    Run one PID episode from start_pos to target waypoint.

    Returns (obs_list, act_list, success) where obs/act are lists of arrays,
    and success indicates whether the PID reached the target.
    """
    # For landing targets (y~0), use a generous hold time and more steps;
    # for aerial, use shorter settings
    is_landing = target[1] < 1.0
    if is_landing:
        hold_time = 60.0
        max_speed = 1.0
        acceptance_radius = 0.5
        if max_steps is None:
            # Landing from up to 40m needs ~30-60s; allow up to 100s
            max_steps = 5000
    else:
        hold_time = 3.0
        max_speed = 5.0
        acceptance_radius = 1.5
        if max_steps is None:
            max_steps = MAX_EPISODE_STEPS

    waypoints = [
        Waypoint(target.copy(), hold_time=hold_time,
                 max_speed=max_speed, acceptance_radius=acceptance_radius),
    ]

    fuel_frac = np.random.uniform(0.6, 1.0)
    kernel, actuators, sensors, config = make_sim(start_pos, start_vel,
                                                  init_fuel_frac=fuel_frac)

    controller = HopperController(
        throttle=actuators._throttle,
        gimbal=actuators._gimbal,
        waypoints=waypoints,
        config=config,
    )
    controller = controller.with_sensors(sensors)

    obs_list = []
    act_list = []
    t = 0.0

    for step in range(max_steps):
        sensors.set_time(t)

        # Build observation BEFORE the controller acts (this is what the RL agent sees)
        obs = build_obs(sensors, target)

        # Run PID controller
        controller.run()

        # Step actuators and physics
        actuators.set_time(t)
        actuators.step(DT)
        kernel.step(DT)
        t += DT

        # Extract action from the state AFTER the controller has applied commands
        # but BEFORE the next physics step alters state further.
        # The thrust is applied by the controller; the nose direction reflects
        # what the PD attitude loop is tracking.
        action = extract_action(kernel, max_tilt)

        obs_list.append(obs)
        act_list.append(action)

        # Check termination
        s = kernel.rocket.state
        speed = s.velocity.norm

        # Success: controller finished (waypoint reached)
        if controller.finished:
            if is_landing:
                # PID's landing detection: alt < 0.3 and speed < 0.5
                # The rocket may still be settling to ground; accept this as success
                return obs_list, act_list, True
            else:
                return obs_list, act_list, True

        # Failure: crash or excessive tilt
        if kernel.ground_contact.crashed or kernel.ground_contact.tipped_over:
            return obs_list, act_list, False

        nose_body = RocketVec3(1.0, 0.0, 0.0)
        nose_world = nose_body.to_world(s.orientation)
        cos_tilt = max(-1.0, min(1.0, nose_world.y))
        tilt = math.acos(cos_tilt)
        if tilt > 1.2:
            return obs_list, act_list, False

        # Out of fuel
        if kernel.rocket.mass_model.fuel_fraction <= 0:
            return obs_list, act_list, False

    # Timeout: check if we at least got close
    pos = np.array([kernel.rocket.state.position.x,
                    kernel.rocket.state.position.y,
                    kernel.rocket.state.position.z])
    dist = np.linalg.norm(target - pos)
    speed = kernel.rocket.state.velocity.norm

    # For landing, check ground contact; for aerial, check proximity
    if is_landing and kernel.ground_contact.on_ground and not kernel.ground_contact.crashed:
        return obs_list, act_list, True
    if not is_landing and dist < 3.0 and speed < 3.0:
        return obs_list, act_list, True

    return obs_list, act_list, False


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Collect PID demonstrations for behavioral cloning")
    parser.add_argument("--episodes", type=int, default=10000,
                        help="Total episodes to attempt (default: 10000)")
    parser.add_argument("--max-tilt", type=float, default=0.25,
                        help="GuidanceWrapper max_tilt for action normalization")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output", type=str, default="runs/pid_demos/demos.npz",
                        help="Output path for saved demos")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)  # for fuel_frac sampling in run_episode

    # Episode distribution: 2K each of 5 scenario types for 10K total
    episodes_per_type = args.episodes // 5
    scenario_schedule = []
    for name in ["vertical", "lateral", "diagonal", "landing", "liftoff"]:
        scenario_schedule.extend([name] * episodes_per_type)
    # Fill remaining with random
    remaining = args.episodes - len(scenario_schedule)
    types = list(SCENARIO_GENERATORS.keys())
    for _ in range(remaining):
        scenario_schedule.append(rng.choice(types))
    rng.shuffle(scenario_schedule)

    all_obs = []
    all_act = []

    success_counts = {k: 0 for k in SCENARIO_GENERATORS}
    attempt_counts = {k: 0 for k in SCENARIO_GENERATORS}
    total_steps_collected = 0

    t_start = time.time()
    print(f"Collecting PID demos: {args.episodes} episodes, max_tilt={args.max_tilt}")
    print(f"Distribution: {episodes_per_type} per scenario type")
    print()

    for i, scenario_name in enumerate(scenario_schedule):
        gen = SCENARIO_GENERATORS[scenario_name]
        start_pos, start_vel, target = gen(rng)
        attempt_counts[scenario_name] += 1

        obs_list, act_list, success = run_episode(
            start_pos, start_vel, target, args.max_tilt)

        if success and len(obs_list) > 0:
            success_counts[scenario_name] += 1
            all_obs.extend(obs_list)
            all_act.extend(act_list)
            total_steps_collected += len(obs_list)

        # Progress every 500 episodes
        if (i + 1) % 500 == 0 or (i + 1) == len(scenario_schedule):
            elapsed = time.time() - t_start
            eps_per_sec = (i + 1) / elapsed
            total_success = sum(success_counts.values())
            total_attempts = sum(attempt_counts.values())
            rate = total_success / max(1, total_attempts) * 100
            print(f"[{i+1:>6d}/{len(scenario_schedule)}] "
                  f"{total_steps_collected:>8d} steps | "
                  f"success {total_success}/{total_attempts} ({rate:.0f}%) | "
                  f"{eps_per_sec:.0f} ep/s | "
                  f"{elapsed:.0f}s")

    # Summary
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Collection complete in {elapsed:.1f}s")
    print(f"\n{'Scenario':<12s} {'Attempts':>8s} {'Success':>8s} {'Rate':>7s}")
    print("-" * 40)
    for name in SCENARIO_GENERATORS:
        att = attempt_counts[name]
        suc = success_counts[name]
        rate = suc / max(1, att) * 100
        print(f"{name:<12s} {att:>8d} {suc:>8d} {rate:>6.1f}%")

    total_success = sum(success_counts.values())
    total_attempts = sum(attempt_counts.values())
    print("-" * 40)
    print(f"{'Total':<12s} {total_attempts:>8d} {total_success:>8d} "
          f"{total_success / max(1, total_attempts) * 100:>6.1f}%")

    if total_steps_collected == 0:
        print("\nNo successful demonstrations collected. Exiting.")
        return

    # Stack into arrays
    obs_array = np.array(all_obs, dtype=np.float32)  # (N, 19)
    act_array = np.array(all_act, dtype=np.float32)   # (N, 3)

    print(f"\nDataset: {obs_array.shape[0]} timesteps")
    print(f"  obs shape: {obs_array.shape}")
    print(f"  act shape: {act_array.shape}")

    # Action statistics
    print(f"\nAction statistics:")
    print(f"  throttle:  mean={act_array[:, 0].mean():.3f}  "
          f"std={act_array[:, 0].std():.3f}  "
          f"range=[{act_array[:, 0].min():.3f}, {act_array[:, 0].max():.3f}]")
    print(f"  tilt_x:    mean={act_array[:, 1].mean():.3f}  "
          f"std={act_array[:, 1].std():.3f}  "
          f"range=[{act_array[:, 1].min():.3f}, {act_array[:, 1].max():.3f}]")
    print(f"  tilt_z:    mean={act_array[:, 2].mean():.3f}  "
          f"std={act_array[:, 2].std():.3f}  "
          f"range=[{act_array[:, 2].min():.3f}, {act_array[:, 2].max():.3f}]")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_path), obs=obs_array, actions=act_array)
    print(f"\nSaved to {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
