"""
JAX-vectorized hopper environment for goal-conditioned waypoint tracking.

Port of HopperTrackingEnv (hopper_env.py) to JAX for massive parallel training.
Runs thousands of hopper environments in parallel via jax.vmap.

Architecture:
    SB3JaxHopperVecEnv (SB3-compatible VecEnv adapter)
      └─ JaxHopperVecEnv (core JAX env)
           └─ hopper_physics_step  ← jit + vmap
                └─ jax_kernel.py physics primitives (actuators, mass, quat, etc.)

Observation (19-dim, matches NumPy HopperTrackingEnv):
    [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz,   # base state (13)
     dx, dy, dz,                                            # goal-relative (3)
     dist_to_wp, vel_toward, brake_margin_wp]               # features (3)

Action (3-dim): [throttle, gimbal_outer, gimbal_inner]
    Designed to be used with guidance wrapper (PD inner loop).

Scenarios (10 types, integer-encoded for branchless JAX):
    0=vertical, 1=vertical_micro, 2=lateral, 3=lateral_short, 4=diagonal,
    5=landing, 6=landing_precision, 7=landing_high, 8=liftoff, 9=liftoff_fast

Usage:
    from simulacra.jax_hopper_env import SB3JaxHopperVecEnv
    env = SB3JaxHopperVecEnv(num_envs=4096, scenarios=["vertical", "lateral", "landing"])
    model = PPO("MlpPolicy", env, ...)
    model.learn(total_timesteps=10_000_000)
"""

import math
import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional

from .jax_kernel import (
    quat_rotate,
    quat_rotate_inverse,
    quat_multiply,
    quat_normalize,
    quat_from_rotvec,
    nozzle_direction_body,
    compute_mass_properties,
    advance_gimbal_angle,
    advance_throttle,
    integrate_orientation,
    upright_quat,
    default_params,
)


# ──────────────────────────────────────────────────────────────
# Scenario encoding
# ──────────────────────────────────────────────────────────────

SCENARIO_NAMES = [
    "vertical",           # 0
    "vertical_micro",     # 1
    "lateral",            # 2
    "lateral_short",      # 3
    "diagonal",           # 4
    "landing",            # 5
    "landing_precision",  # 6
    "landing_high",       # 7
    "liftoff",            # 8
    "liftoff_fast",       # 9
]

SCENARIO_NAME_TO_ID = {name: i for i, name in enumerate(SCENARIO_NAMES)}
NUM_SCENARIOS = len(SCENARIO_NAMES)


# ──────────────────────────────────────────────────────────────
# Branchless scenario sampling (pure JAX)
# ──────────────────────────────────────────────────────────────

def _sample_scenario(scenario_id, rng_key, params):
    """Sample start_pos, start_vel, target for a given scenario ID.

    All branching uses jnp.where chains — fully branchless on GPU.
    Returns: (start_pos (3,), start_vel (3,), target (3,))
    """
    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11 = jax.random.split(rng_key, 11)

    alt_lo = params["hopper_alt_lo"]
    alt_hi = params["hopper_alt_hi"]
    horiz_lo = params["hopper_horiz_lo"]
    horiz_hi = params["hopper_horiz_hi"]
    vel_lo = params["hopper_vel_lo"]
    vel_hi = params["hopper_vel_hi"]

    # Pre-sample all random values we might need
    r_alt1 = jax.random.uniform(k1, minval=0.0, maxval=alt_hi)
    r_alt2 = jax.random.uniform(k2, minval=alt_lo, maxval=alt_hi)
    r_alt3 = jax.random.uniform(k3, minval=alt_lo, maxval=alt_hi)
    r_hx1 = jax.random.uniform(k4, minval=horiz_lo * 0.5, maxval=horiz_hi * 0.5)
    r_hz1 = jax.random.uniform(k5, minval=horiz_lo * 0.5, maxval=horiz_hi * 0.5)
    r_hx2 = jax.random.uniform(k6, minval=horiz_lo, maxval=horiz_hi)
    r_hz2 = jax.random.uniform(k7, minval=horiz_lo, maxval=horiz_hi)
    r_vx = jax.random.uniform(k8, minval=vel_lo, maxval=vel_hi)
    r_vy = jax.random.uniform(k9, minval=vel_lo, maxval=vel_hi)
    r_vz = jax.random.uniform(k10, minval=vel_lo, maxval=vel_hi)
    r_alt4 = jax.random.uniform(k11, minval=2.0, maxval=alt_hi)

    # Default values (will be overwritten by jnp.where chains)
    start_pos = jnp.zeros(3)
    start_vel = jnp.zeros(3)
    target = jnp.zeros(3)

    # ── Scenario 0: vertical ──
    # Start at random alt, target at different alt, no horizontal
    s0_start = jnp.array([0.0, r_alt1, 0.0])
    s0_vel = jnp.array([0.0, r_vy * 0.67, 0.0])  # scaled vel
    s0_target = jnp.array([0.0, r_alt2, 0.0])
    is_0 = scenario_id == 0

    # ── Scenario 1: vertical_micro ──
    # Precision micro vertical (1-5m range)
    micro_alt_lo = 1.0
    micro_alt_hi = 5.0
    r_micro1 = micro_alt_lo + (micro_alt_hi - micro_alt_lo) * jax.random.uniform(k1)
    r_micro2 = micro_alt_lo + (micro_alt_hi - micro_alt_lo) * jax.random.uniform(k2)
    s1_start = jnp.array([0.0, r_micro1, 0.0])
    s1_vel = jnp.array([0.0, r_vy * 0.33, 0.0])
    s1_target = jnp.array([0.0, r_micro2, 0.0])
    is_1 = scenario_id == 1

    # ── Scenario 2: lateral ──
    # Horizontal move at roughly constant altitude
    s2_start = jnp.array([r_hx1, r_alt3, r_hz1])
    s2_vel = jnp.array([r_vx * 0.67, 0.0, r_vz * 0.67])
    s2_target = jnp.array([r_hx2, r_alt3, r_hz2])
    is_2 = scenario_id == 2

    # ── Scenario 3: lateral_short ──
    # Precision short lateral (tighter ranges)
    short_horiz_lo = -8.0
    short_horiz_hi = 8.0
    short_alt_lo = 10.0
    short_alt_hi = 20.0
    r_sh_alt = short_alt_lo + (short_alt_hi - short_alt_lo) * jax.random.uniform(k3)
    r_sh_x1 = short_horiz_lo * 0.5 + (short_horiz_hi - short_horiz_lo) * 0.5 * jax.random.uniform(k4)
    r_sh_z1 = short_horiz_lo * 0.5 + (short_horiz_hi - short_horiz_lo) * 0.5 * jax.random.uniform(k5)
    r_sh_x2 = short_horiz_lo + (short_horiz_hi - short_horiz_lo) * jax.random.uniform(k6)
    r_sh_z2 = short_horiz_lo + (short_horiz_hi - short_horiz_lo) * jax.random.uniform(k7)
    s3_start = jnp.array([r_sh_x1, r_sh_alt, r_sh_z1])
    s3_vel = jnp.array([r_vx * 0.33, 0.0, r_vz * 0.33])
    s3_target = jnp.array([r_sh_x2, r_sh_alt, r_sh_z2])
    is_3 = scenario_id == 3

    # ── Scenario 4: diagonal ──
    # Combined 3D transition
    s4_start = jnp.array([r_hx1, r_alt4, r_hz1])
    s4_vel = jnp.array([r_vx, r_vy, r_vz])
    s4_target = jnp.array([r_hx2, r_alt2, r_hz2])
    is_4 = scenario_id == 4

    # ── Scenario 5: landing ──
    # Approach to ground from moderate altitude
    r_land_alt = alt_lo + (alt_hi - alt_lo) * jax.random.uniform(k1)
    r_land_hx = horiz_lo * 0.67 + (horiz_hi - horiz_lo) * 0.67 * jax.random.uniform(k4)
    r_land_hz = horiz_lo * 0.67 + (horiz_hi - horiz_lo) * 0.67 * jax.random.uniform(k5)
    r_land_vy = -2.0 * jax.random.uniform(k9)  # downward only
    s5_start = jnp.array([r_land_hx, r_land_alt, r_land_hz])
    s5_vel = jnp.array([0.0, r_land_vy, 0.0])
    s5_target = jnp.array([0.0, 0.0, 0.0])
    is_5 = scenario_id == 5

    # ── Scenario 6: landing_precision ──
    # Tight precision landing from low altitude
    prec_alt_lo = 5.0
    prec_alt_hi = 15.0
    prec_horiz = 3.0
    r_prec_alt = prec_alt_lo + (prec_alt_hi - prec_alt_lo) * jax.random.uniform(k1)
    r_prec_hx = -prec_horiz + 2 * prec_horiz * jax.random.uniform(k4)
    r_prec_hz = -prec_horiz + 2 * prec_horiz * jax.random.uniform(k5)
    r_prec_vy = -1.0 * jax.random.uniform(k9)
    s6_start = jnp.array([r_prec_hx, r_prec_alt, r_prec_hz])
    s6_vel = jnp.array([0.0, r_prec_vy, 0.0])
    s6_target = jnp.array([0.0, 0.0, 0.0])
    is_6 = scenario_id == 6

    # ── Scenario 7: landing_high ──
    # Extended landing from high altitude
    high_alt_lo = 50.0
    high_alt_hi = 100.0
    high_horiz = 15.0
    r_high_alt = high_alt_lo + (high_alt_hi - high_alt_lo) * jax.random.uniform(k1)
    r_high_hx = -high_horiz + 2 * high_horiz * jax.random.uniform(k4)
    r_high_hz = -high_horiz + 2 * high_horiz * jax.random.uniform(k5)
    r_high_vy = -5.0 * jax.random.uniform(k9)
    s7_start = jnp.array([r_high_hx, r_high_alt, r_high_hz])
    s7_vel = jnp.array([0.0, r_high_vy, 0.0])
    s7_target = jnp.array([0.0, 0.0, 0.0])
    is_7 = scenario_id == 7

    # ── Scenario 8: liftoff ──
    # Take off from ground to target altitude
    r_lift_alt = alt_lo + (alt_hi - alt_lo) * jax.random.uniform(k2)
    r_lift_hx = -5.0 + 10.0 * jax.random.uniform(k6)
    r_lift_hz = -5.0 + 10.0 * jax.random.uniform(k7)
    s8_start = jnp.array([0.0, 0.0, 0.0])
    s8_vel = jnp.zeros(3)
    s8_target = jnp.array([r_lift_hx, r_lift_alt, r_lift_hz])
    is_8 = scenario_id == 8

    # ── Scenario 9: liftoff_fast ──
    # Aggressive liftoff with lateral component
    fast_alt_lo = 50.0
    fast_alt_hi = 100.0
    fast_horiz = 10.0
    r_fast_alt = fast_alt_lo + (fast_alt_hi - fast_alt_lo) * jax.random.uniform(k2)
    r_fast_hx = -fast_horiz + 2 * fast_horiz * jax.random.uniform(k6)
    r_fast_hz = -fast_horiz + 2 * fast_horiz * jax.random.uniform(k7)
    s9_start = jnp.array([0.0, 0.0, 0.0])
    s9_vel = jnp.zeros(3)
    s9_target = jnp.array([r_fast_hx, r_fast_alt, r_fast_hz])
    is_9 = scenario_id == 9

    # ── Compose via jnp.where chain ──
    start_pos = jnp.where(is_0, s0_start, start_pos)
    start_pos = jnp.where(is_1, s1_start, start_pos)
    start_pos = jnp.where(is_2, s2_start, start_pos)
    start_pos = jnp.where(is_3, s3_start, start_pos)
    start_pos = jnp.where(is_4, s4_start, start_pos)
    start_pos = jnp.where(is_5, s5_start, start_pos)
    start_pos = jnp.where(is_6, s6_start, start_pos)
    start_pos = jnp.where(is_7, s7_start, start_pos)
    start_pos = jnp.where(is_8, s8_start, start_pos)
    start_pos = jnp.where(is_9, s9_start, start_pos)

    start_vel = jnp.where(is_0, s0_vel, start_vel)
    start_vel = jnp.where(is_1, s1_vel, start_vel)
    start_vel = jnp.where(is_2, s2_vel, start_vel)
    start_vel = jnp.where(is_3, s3_vel, start_vel)
    start_vel = jnp.where(is_4, s4_vel, start_vel)
    start_vel = jnp.where(is_5, s5_vel, start_vel)
    start_vel = jnp.where(is_6, s6_vel, start_vel)
    start_vel = jnp.where(is_7, s7_vel, start_vel)
    start_vel = jnp.where(is_8, s8_vel, start_vel)
    start_vel = jnp.where(is_9, s9_vel, start_vel)

    target = jnp.where(is_0, s0_target, target)
    target = jnp.where(is_1, s1_target, target)
    target = jnp.where(is_2, s2_target, target)
    target = jnp.where(is_3, s3_target, target)
    target = jnp.where(is_4, s4_target, target)
    target = jnp.where(is_5, s5_target, target)
    target = jnp.where(is_6, s6_target, target)
    target = jnp.where(is_7, s7_target, target)
    target = jnp.where(is_8, s8_target, target)
    target = jnp.where(is_9, s9_target, target)

    # On-ground starts: clamp position to y=0 and zero velocity
    on_ground = start_pos[1] < 0.5
    start_pos = jnp.where(on_ground, start_pos.at[1].set(0.0), start_pos)
    start_vel = jnp.where(on_ground, jnp.zeros(3), start_vel)

    return start_pos, start_vel, target, on_ground


# ──────────────────────────────────────────────────────────────
# Guidance wrapper (PD attitude controller, runs in JAX)
# ──────────────────────────────────────────────────────────────

ATT_KP = 500.0
ATT_KD = 200.0
MOMENT_ARM = 2.0


def hopper_guidance_to_gimbal(obs_13, guidance_action, max_tilt, max_thrust, max_gimbal):
    """Convert [throttle, tilt_x, tilt_z] -> [throttle, gimbal_outer, gimbal_inner].

    PD attitude controller matching GuidanceWrapper from hopper_env.py.
    Uses the first 13 dims of obs (base state).
    """
    throttle = guidance_action[0]
    tilt_x = guidance_action[1] * max_tilt
    tilt_z = guidance_action[2] * max_tilt

    # Desired nose direction (world frame)
    desired = jnp.array([tilt_x, 1.0, tilt_z])
    desired = desired / jnp.linalg.norm(desired)

    # Current orientation from obs
    quat = jnp.array([obs_13[6], obs_13[7], obs_13[8], obs_13[9]])
    omega = jnp.array([obs_13[10], obs_13[11], obs_13[12]])

    # PD attitude control
    nose_world = quat_rotate(quat, jnp.array([1.0, 0.0, 0.0]))
    error_world = jnp.cross(nose_world, desired)
    error_body = quat_rotate_inverse(quat, error_world)
    tau = ATT_KP * error_body - ATT_KD * omega

    # Convert torque to gimbal angles
    thrust_n = throttle * max_thrust
    scale = jnp.where(thrust_n > 100.0, 1.0 / (thrust_n * MOMENT_ARM), 0.0)
    outer = jnp.clip(-tau[2] * scale, -max_gimbal, max_gimbal)
    inner = jnp.clip(-tau[1] * scale, -max_gimbal, max_gimbal)

    return jnp.array([throttle, outer / max_gimbal, inner / max_gimbal])


# ──────────────────────────────────────────────────────────────
# Hopper physics step (reuses jax_kernel primitives)
# ──────────────────────────────────────────────────────────────

A_MAX = 4.0     # conservative max deceleration for brake margin feature
MAX_DECEL = 8.0  # mid-fuel TWR for braking profile in reward


def hopper_physics_step(state, action, params, rng_key):
    """Pure-function hopper physics step.

    Reuses actuator, mass, thrust, drag, and rotation physics from jax_kernel.py.
    Adds goal-conditioned observation (19-dim), R7 reward, and hopper-specific
    termination (arrived/crash/truncated).

    Args:
        state: dict with keys:
            pos (3,), vel (3,), quat (4,), omega (3,),
            gimbal_outer, gimbal_inner, actual_thrust,
            fuel_mass, on_ground, step_count,
            target (3,), prev_dist, prev_throttle, prev_omega_mag
        action: (3,) array [throttle_norm, gimbal_outer_norm, gimbal_inner_norm]
        params: dict of static parameters
        rng_key: JAX PRNG key

    Returns:
        new_state, obs (19,), reward, done, info
    """
    dt = params["dt"]
    pos = state["pos"]
    vel = state["vel"]
    quat = state["quat"]
    omega = state["omega"]
    target = state["target"]

    # ── Decode action ──
    throttle_cmd = action[0] * params["max_thrust"]
    gimbal_outer_cmd = action[1] * params["max_gimbal"]
    gimbal_inner_cmd = action[2] * params["max_gimbal"]

    # ── Actuator dynamics ──
    gimbal_outer = advance_gimbal_angle(
        state["gimbal_outer"], gimbal_outer_cmd,
        params["max_gimbal"], params["gimbal_rate"], params["gimbal_tau"], dt,
    )
    gimbal_inner = advance_gimbal_angle(
        state["gimbal_inner"], gimbal_inner_cmd,
        params["max_gimbal"], params["gimbal_rate"], params["gimbal_tau"], dt,
    )
    actual_thrust = advance_throttle(
        state["actual_thrust"], throttle_cmd,
        params["max_thrust"], params["max_thrust_rate"], params["min_thrust"], dt,
    )

    # ── Fuel burn ──
    is_thrusting = actual_thrust > 0.0
    fuel_burned = jnp.where(
        is_thrusting,
        jnp.minimum(params["burn_rate"] * dt, state["fuel_mass"]),
        0.0,
    )
    fuel_mass = state["fuel_mass"] - fuel_burned
    actual_thrust = jnp.where(fuel_mass <= 0.0, 0.0, actual_thrust)

    # ── Mass properties ──
    total_mass, com_x, Ix, Iy, Iz = compute_mass_properties(fuel_mass, params)

    # ── Thrust vector in body frame ──
    nozzle_dir = nozzle_direction_body(gimbal_outer, gimbal_inner)
    thrust_body = actual_thrust * nozzle_dir

    # ── Kill thrust if on ground (except for liftoff — handled by allowing
    #    thrust when target is above) ──
    # For hopper: allow thrust on ground (liftoff scenarios need it)
    # Ground contact freezes velocity but thrust is still applied
    # to allow liftoff on next step

    # ── Translational dynamics (world frame) ──
    thrust_world = quat_rotate(quat, thrust_body)
    gravity_force = jnp.array([0.0, params["gravity"] * total_mass, 0.0])
    wind_force = params["wind"]

    # Drag
    speed = jnp.linalg.norm(vel)
    drag_coeff = -0.5 * params["air_density"] * params["Cd"] * params["ref_area"]
    drag_force = jnp.where(speed > 1e-6, drag_coeff * speed * vel, jnp.zeros(3))

    net_force = thrust_world + gravity_force + wind_force + drag_force

    # On ground: only allow upward net force (normal force from ground)
    net_force_y_clamped = jnp.where(
        state["on_ground"] & (net_force[1] < 0.0),
        0.0,
        net_force[1],
    )
    net_force = jnp.where(
        state["on_ground"],
        jnp.array([0.0, net_force_y_clamped, 0.0]),
        net_force,
    )

    accel = net_force / total_mass

    # Semi-implicit Euler
    vel_new = vel + accel * dt
    pos_new = pos + vel_new * dt

    # ── Rotational dynamics (body frame, Euler's equations) ──
    thrust_offset = jnp.array([0.0, 0.0, 0.0]) - jnp.array([com_x, 0.0, 0.0])
    torque = jnp.cross(thrust_offset, thrust_body)

    omega_dot = jnp.array([
        (torque[0] + (Iy - Iz) * omega[1] * omega[2]) / Ix,
        (torque[1] + (Iz - Ix) * omega[2] * omega[0]) / Iy,
        (torque[2] + (Ix - Iy) * omega[0] * omega[1]) / Iz,
    ])

    # On ground: zero rotational dynamics
    omega_dot = jnp.where(state["on_ground"], jnp.zeros(3), omega_dot)

    omega_new = omega + omega_dot * dt
    quat_new = integrate_orientation(quat, omega_new, dt)

    # ── Ground contact ──
    touching = pos_new[1] <= 0.0
    was_airborne = jnp.logical_not(state["on_ground"])
    just_landed = was_airborne & touching

    # Impact metrics
    impact_speed = jnp.linalg.norm(vel_new)
    nose_world = quat_rotate(quat_new, jnp.array([1.0, 0.0, 0.0]))
    cos_tilt = jnp.clip(nose_world[1], -1.0, 1.0)
    tilt_angle = jnp.arccos(cos_tilt)

    crashed_impact = just_landed & (impact_speed > params["max_landing_speed"])
    tipped = just_landed & (tilt_angle > params["max_landing_tilt"])

    # Freeze state on ground
    pos_new = jnp.where(touching, pos_new.at[1].set(0.0), pos_new)
    vel_new = jnp.where(touching, jnp.zeros(3), vel_new)
    omega_new = jnp.where(touching, jnp.zeros(3), omega_new)

    # ── Observation (19-dim with sensor noise) ──
    k1, k2, k3, k4 = jax.random.split(rng_key, 4)
    gps_noise = params["gps_noise_std"] * jax.random.normal(k1, (2,))
    baro_noise = params["baro_noise_std"] * jax.random.normal(k2, (1,))
    vel_noise = params["vel_noise_std"] * jax.random.normal(k3, (3,))
    gyro_noise = params["gyro_noise_std"] * jax.random.normal(k4, (3,))

    obs_pos = jnp.array([
        pos_new[0] + gps_noise[0],
        pos_new[1] + baro_noise[0],
        pos_new[2] + gps_noise[1],
    ])
    obs_vel = vel_new + vel_noise
    obs_omega = omega_new + gyro_noise

    # Goal-relative
    goal_rel = target - obs_pos
    dist = jnp.linalg.norm(goal_rel)
    dir_to_wp = goal_rel / jnp.maximum(dist, 0.01)
    vel_toward = jnp.dot(obs_vel, dir_to_wp)
    obs_speed = jnp.linalg.norm(obs_vel)
    brake_margin = dist - obs_speed ** 2 / (2.0 * A_MAX)

    obs = jnp.concatenate([
        obs_pos,                          # 3: x, y, z
        obs_vel,                          # 3: vx, vy, vz
        quat_new,                         # 4: qw, qx, qy, qz
        obs_omega,                        # 3: wx, wy, wz
        goal_rel,                         # 3: dx, dy, dz
        jnp.array([dist, vel_toward, brake_margin]),  # 3: features
    ])

    # ── Reward R8 (7 unified terms, stronger smoothness) ──
    true_speed = jnp.linalg.norm(vel_new)
    omega_mag = jnp.linalg.norm(omega_new)
    throttle_norm = action[0]
    fuel_frac = fuel_mass / jnp.maximum(params["fuel_capacity"], 1e-6)
    prev_dist = state["prev_dist"]
    prev_throttle = state["prev_throttle"]
    prev_omega_mag = state["prev_omega_mag"]
    prev_gimbal_outer = state["prev_gimbal_outer"]
    prev_gimbal_inner = state["prev_gimbal_inner"]

    # Smoothness weight (curriculum: ramps from 0.7 → 1.5 during training)
    smooth_w = params.get("smoothness_weight", 1.0)

    # 1. Distance potential (unchanged)
    r_approach = 0.15 * (prev_dist - dist)

    # 2. Braking profile (unchanged)
    v_safe = jnp.sqrt(2.0 * MAX_DECEL * jnp.maximum(0.1, dist))
    excess = jnp.maximum(0.0, true_speed - v_safe)
    r_brake = -0.3 * excess ** 2

    # 3. Hover quality (unchanged)
    r_hover = jnp.where(dist < 3.0, 0.15 * jnp.maximum(0.0, 1.0 - true_speed / 1.5), 0.0)

    # 4. Uprightness (strengthened: 0.05 → 0.08)
    r_tilt = -0.08 * tilt_angle ** 2

    # 5. Smoothness (all penalties 2-3x stronger)
    r_smooth = smooth_w * (
        -0.06 * omega_mag                                    # angular rate (was -0.025)
        - 0.025 * (throttle_norm - prev_throttle) ** 2       # throttle jerk (was -0.01)
        - 0.015 * (omega_mag - prev_omega_mag) ** 2          # angular accel (was -0.005)
    )

    # 6. Gimbal rate penalty (NEW — penalises rapid gimbal oscillation)
    gimbal_delta_sq = (
        (gimbal_outer - prev_gimbal_outer) ** 2
        + (gimbal_inner - prev_gimbal_inner) ** 2
    )
    r_gimbal = smooth_w * (-0.02 * gimbal_delta_sq)

    # 7. Fuel efficiency (NEW — tiny dense penalty for running engine)
    r_fuel = -0.001 * throttle_norm

    dense_reward = r_approach + r_brake + r_hover + r_tilt + r_smooth + r_gimbal + r_fuel

    # ── Termination ──
    step_count = state["step_count"] + 1
    truncated = step_count >= params["max_episode_steps"]
    tilt_exceeded = tilt_angle > params["max_tilt"]
    alt_exceeded = pos_new[1] > params["max_altitude"]
    fuel_empty = fuel_mass <= 0.0

    is_landing_wp = target[1] < 1.0
    is_aerial_wp = jnp.logical_not(is_landing_wp)

    # Arrived at aerial waypoint
    arrived = is_aerial_wp & (dist < params["acceptance_radius"]) & (true_speed < params["max_arrival_speed"])

    # Landed at ground waypoint
    landed_ok = is_landing_wp & just_landed & ~crashed_impact & ~tipped
    landed_crash = is_landing_wp & just_landed & (crashed_impact | tipped)

    # Crash: ground contact when target is in air (after grace period for liftoff)
    air_target_ground_crash = is_aerial_wp & touching & (pos_new[1] <= 0.01) & (step_count > 5)

    # Kinematic early termination (RAJS/Jiang 2024)
    brake_dist = true_speed ** 2 / (2.0 * A_MAX)
    kinematic_crash = (brake_dist > dist * 2.0) & (dist > 5.0) & (step_count > 10)

    # Aggregate
    success = arrived | landed_ok
    crash = landed_crash | air_target_ground_crash | tilt_exceeded | alt_exceeded | fuel_empty | kinematic_crash
    terminated = success | crash
    done = terminated | truncated

    # ── Terminal rewards ──
    success_reward = (
        200.0
        + 50.0 * jnp.maximum(0.0, 1.0 - true_speed / 2.0)
        + 30.0 * jnp.maximum(0.0, 1.0 - dist / params["acceptance_radius"])
        + 10.0 * fuel_frac
    )
    crash_reward = -200.0
    truncation_reward = -50.0 - 0.5 * jnp.minimum(dist, 50.0)

    # Select reward
    reward = dense_reward
    reward = jnp.where(success, success_reward, reward)
    reward = jnp.where(crash & ~success, crash_reward, reward)
    reward = jnp.where(truncated & ~terminated, truncation_reward, reward)

    # ── Assemble new state ──
    new_state = {
        "pos": pos_new,
        "vel": vel_new,
        "quat": quat_new,
        "omega": omega_new,
        "gimbal_outer": gimbal_outer,
        "gimbal_inner": gimbal_inner,
        "actual_thrust": actual_thrust,
        "fuel_mass": fuel_mass,
        "on_ground": touching,
        "step_count": step_count,
        "target": target,
        "prev_dist": dist,
        "prev_throttle": throttle_norm,
        "prev_omega_mag": omega_mag,
        "prev_gimbal_outer": gimbal_outer,
        "prev_gimbal_inner": gimbal_inner,
        "scenario_id": state["scenario_id"],
    }

    info = {
        "arrived": arrived,
        "landed": landed_ok,
        "crashed": crash & ~success,
        "truncated": truncated & ~terminated,
        "dist": dist,
        "speed": true_speed,
        "tilt_angle": tilt_angle,
        "impact_speed": jnp.where(just_landed, impact_speed, 0.0),
        "scenario_id": state.get("scenario_id", jnp.int32(0)),
    }

    return new_state, obs, reward, done, info


# ──────────────────────────────────────────────────────────────
# Hopper initial state
# ──────────────────────────────────────────────────────────────

def make_hopper_initial_state(
    start_pos,
    start_vel,
    target,
    on_ground,
    tilt_y=0.0,
    tilt_z=0.0,
    fuel_fraction=1.0,
    fuel_capacity=80.0,
    scenario_id=0,
):
    """Create an initial hopper state dict from scenario outputs."""
    # Orientation: upright with small random tilt
    base_quat = upright_quat()
    tilt_rotvec_y = quat_from_rotvec(jnp.array([0.0, tilt_y, 0.0]))
    tilt_rotvec_z = quat_from_rotvec(jnp.array([0.0, 0.0, tilt_z]))
    quat = quat_normalize(quat_multiply(quat_multiply(tilt_rotvec_z, tilt_rotvec_y), base_quat))

    dist = jnp.linalg.norm(target - start_pos)

    return {
        "pos": start_pos,
        "vel": start_vel,
        "quat": quat,
        "omega": jnp.zeros(3),
        "gimbal_outer": 0.0,
        "gimbal_inner": 0.0,
        "actual_thrust": 0.0,
        "fuel_mass": fuel_fraction * fuel_capacity,
        "on_ground": on_ground,
        "step_count": jnp.int32(0),
        "target": target,
        "prev_dist": dist,
        "prev_throttle": 0.0,
        "prev_omega_mag": 0.0,
        "prev_gimbal_outer": 0.0,
        "prev_gimbal_inner": 0.0,
        "scenario_id": jnp.int32(scenario_id),
    }


# ──────────────────────────────────────────────────────────────
# Hopper default parameters
# ──────────────────────────────────────────────────────────────

def hopper_default_params(
    dt=0.02,
    max_episode_steps=500,
    max_altitude=100.0,
    # Rocket
    dry_mass=60.0,
    fuel_capacity=80.0,
    length=4.0,
    max_thrust=2000.0,
    burn_rate=0.68,
    max_gimbal=0.26,
    gimbal_rate=1.0,
    gimbal_tau=0.05,
    max_thrust_rate=4000.0,
    min_thrust=0.0,
    # Landing thresholds
    max_landing_speed=3.0,
    max_landing_tilt=0.26,
    max_tilt=1.0,
    # Tracking
    acceptance_radius=1.5,
    max_arrival_speed=1.0,
    # Environment
    gravity=-9.81,
    air_density=1.225,
    Cd=0.5,
    ref_area=0.0314,
    # Sensor noise
    gps_noise_std=0.0,
    baro_noise_std=0.0,
    vel_noise_std=0.0,
    gyro_noise_std=0.0,
    # Hopper scenario ranges (defaults)
    hopper_alt_lo=5.0,
    hopper_alt_hi=30.0,
    hopper_horiz_lo=-15.0,
    hopper_horiz_hi=15.0,
    hopper_vel_lo=-1.5,
    hopper_vel_hi=1.5,
    # Init perturbation ranges
    init_tilt_lo=-0.15,
    init_tilt_hi=0.15,
    init_fuel_lo=0.6,
    init_fuel_hi=1.0,
    # Smoothness curriculum weight (ramps during training)
    smoothness_weight=1.0,
):
    """Create parameter dict for hopper environment."""
    return {
        "dt": dt,
        "max_episode_steps": max_episode_steps,
        "max_altitude": max_altitude,
        "dry_mass": dry_mass,
        "fuel_capacity": fuel_capacity,
        "length": length,
        "max_thrust": max_thrust,
        "burn_rate": burn_rate,
        "max_gimbal": max_gimbal,
        "gimbal_rate": gimbal_rate,
        "gimbal_tau": gimbal_tau,
        "max_thrust_rate": max_thrust_rate,
        "min_thrust": min_thrust,
        "max_landing_speed": max_landing_speed,
        "max_landing_tilt": max_landing_tilt,
        "max_tilt": max_tilt,
        "acceptance_radius": acceptance_radius,
        "max_arrival_speed": max_arrival_speed,
        "gravity": gravity,
        "air_density": air_density,
        "Cd": Cd,
        "ref_area": ref_area,
        "wind": jnp.zeros(3),
        "gps_noise_std": gps_noise_std,
        "baro_noise_std": baro_noise_std,
        "vel_noise_std": vel_noise_std,
        "gyro_noise_std": gyro_noise_std,
        "hopper_alt_lo": hopper_alt_lo,
        "hopper_alt_hi": hopper_alt_hi,
        "hopper_horiz_lo": hopper_horiz_lo,
        "hopper_horiz_hi": hopper_horiz_hi,
        "hopper_vel_lo": hopper_vel_lo,
        "hopper_vel_hi": hopper_vel_hi,
        "init_tilt_lo": init_tilt_lo,
        "init_tilt_hi": init_tilt_hi,
        "init_fuel_lo": init_fuel_lo,
        "init_fuel_hi": init_fuel_hi,
        "smoothness_weight": smoothness_weight,
    }


# ──────────────────────────────────────────────────────────────
# Core JAX vectorized hopper environment
# ──────────────────────────────────────────────────────────────

class JaxHopperVecEnv:
    """Vectorized hopper tracking environment running entirely via JAX.

    Args:
        num_envs: Number of parallel environments.
        scenarios: List of scenario name strings (default: all 10).
        scenario_weights: Per-scenario sampling weights (default: uniform).
        use_guidance: If True, action is [throttle, tilt_x, tilt_z] with PD inner loop.
        max_tilt: Maximum tilt for guidance wrapper (radians).
        seed: Random seed.
        **kwargs: Parameter overrides for hopper_default_params.
    """

    def __init__(
        self,
        num_envs: int = 4096,
        scenarios: Optional[list[str]] = None,
        scenario_weights: Optional[list[float]] = None,
        use_guidance: bool = True,
        max_tilt: float = 0.25,
        acceptance_radius: float = 1.5,
        max_arrival_speed: float = 1.0,
        alt_range: tuple = (5.0, 30.0),
        horiz_range: tuple = (-15.0, 15.0),
        vel_range: tuple = (-1.5, 1.5),
        init_tilt_range: tuple = (-0.15, 0.15),
        init_fuel_range: tuple = (0.6, 1.0),
        max_episode_steps: int = 500,
        max_altitude: float = 100.0,
        seed: int = 0,
        **param_overrides,
    ):
        self.num_envs = num_envs
        self.use_guidance = use_guidance
        self.max_tilt = max_tilt

        # Scenario setup
        if scenarios is None:
            scenarios = list(SCENARIO_NAMES)
        self.scenario_names = scenarios
        self._scenario_ids = jnp.array(
            [SCENARIO_NAME_TO_ID[s] for s in scenarios], dtype=jnp.int32,
        )
        num_active = len(scenarios)
        if scenario_weights is None:
            self._scenario_weights = jnp.ones(num_active) / num_active
        else:
            w = jnp.array(scenario_weights)
            self._scenario_weights = w / jnp.sum(w)

        # Build params
        self.params = hopper_default_params(
            max_episode_steps=max_episode_steps,
            max_altitude=max_altitude,
            acceptance_radius=acceptance_radius,
            max_arrival_speed=max_arrival_speed,
            hopper_alt_lo=alt_range[0],
            hopper_alt_hi=alt_range[1],
            hopper_horiz_lo=horiz_range[0],
            hopper_horiz_hi=horiz_range[1],
            hopper_vel_lo=vel_range[0],
            hopper_vel_hi=vel_range[1],
            init_tilt_lo=init_tilt_range[0],
            init_tilt_hi=init_tilt_range[1],
            init_fuel_lo=init_fuel_range[0],
            init_fuel_hi=init_fuel_range[1],
            **param_overrides,
        )

        # PRNG
        self.rng = jax.random.PRNGKey(seed)

        # Observation space: 19-dim
        self._obs_dim = 19
        vel_bound = max(50.0, math.sqrt(2 * 9.81 * max_altitude) * 1.2)
        alt_bound = max(150.0, max_altitude * 1.2)
        dist_bound = 200.0

        self.obs_low = np.array([
            -100, -10, -100,                           # position
            -vel_bound, -vel_bound, -vel_bound,        # velocity
            -1, -1, -1, -1,                            # quaternion
            -10, -10, -10,                             # angular velocity
            -dist_bound, -dist_bound, -dist_bound,     # goal relative
            0.0, -vel_bound, -dist_bound,              # features
        ], dtype=np.float32)

        self.obs_high = np.array([
            100, alt_bound, 100,
            vel_bound, vel_bound, vel_bound,
            1, 1, 1, 1,
            10, 10, 10,
            dist_bound, dist_bound, dist_bound,
            dist_bound, vel_bound, dist_bound,
        ], dtype=np.float32)

        # State
        self._state = None

        # Pre-compile kernels
        self._compile_kernels()

    def _compile_kernels(self):
        """JIT-compile the batched step and reset functions."""
        params = self.params
        use_guidance = self.use_guidance
        max_tilt = self.max_tilt
        scenario_ids = self._scenario_ids
        scenario_weights = self._scenario_weights

        def _step_single(state, action, rng_key):
            # Guidance wrapper: convert tilt -> gimbal
            obs_13 = jnp.concatenate([
                state["pos"], state["vel"], state["quat"], state["omega"],
            ])

            raw_action = jnp.where(
                use_guidance,
                hopper_guidance_to_gimbal(
                    obs_13, action, max_tilt,
                    params["max_thrust"], params["max_gimbal"],
                ),
                action,
            )

            return hopper_physics_step(state, raw_action, params, rng_key)

        self._step_fn = jax.jit(jax.vmap(
            _step_single,
            in_axes=(0, 0, 0),
        ))

        def _reset_single(rng_key):
            k1, k2, k3, k4 = jax.random.split(rng_key, 4)

            # Sample scenario type
            scenario_idx = jax.random.choice(k1, len(scenario_ids), p=scenario_weights)
            scenario_id = scenario_ids[scenario_idx]

            # Sample scenario
            start_pos, start_vel, target, on_ground = _sample_scenario(
                scenario_id, k2, params,
            )

            # Random tilt perturbation
            tilt_y = jax.random.uniform(
                k3, minval=params["init_tilt_lo"], maxval=params["init_tilt_hi"],
            )
            tilt_z = jax.random.uniform(
                k4, minval=params["init_tilt_lo"], maxval=params["init_tilt_hi"],
            )

            # Random fuel fraction
            k5 = jax.random.fold_in(k4, 1)
            fuel_frac = jax.random.uniform(
                k5, minval=params["init_fuel_lo"], maxval=params["init_fuel_hi"],
            )

            return make_hopper_initial_state(
                start_pos=start_pos,
                start_vel=start_vel,
                target=target,
                on_ground=on_ground,
                tilt_y=tilt_y,
                tilt_z=tilt_z,
                fuel_fraction=fuel_frac,
                fuel_capacity=params["fuel_capacity"],
                scenario_id=scenario_id,
            )

        self._reset_fn = jax.jit(jax.vmap(_reset_single))

    def _state_to_obs(self, state):
        """Extract 19-dim observation from batched state."""
        pos = state["pos"]      # (N, 3)
        vel = state["vel"]      # (N, 3)
        quat = state["quat"]    # (N, 4)
        omega = state["omega"]  # (N, 3)
        target = state["target"]  # (N, 3)

        goal_rel = target - pos
        dist = jnp.linalg.norm(goal_rel, axis=-1, keepdims=True)  # (N, 1)
        dir_to_wp = goal_rel / jnp.maximum(dist, 0.01)
        vel_toward = jnp.sum(vel * dir_to_wp, axis=-1, keepdims=True)  # (N, 1)
        speed = jnp.linalg.norm(vel, axis=-1, keepdims=True)  # (N, 1)
        brake_margin = dist - speed ** 2 / (2.0 * A_MAX)

        return jnp.concatenate([
            pos, vel, quat, omega,
            goal_rel,
            dist, vel_toward, brake_margin,
        ], axis=-1)

    def reset(self, seed: Optional[int] = None):
        """Reset all environments. Returns obs (num_envs, 19) as NumPy."""
        if seed is not None:
            self.rng = jax.random.PRNGKey(seed)

        self.rng, reset_key = jax.random.split(self.rng)
        keys = jax.random.split(reset_key, self.num_envs)
        self._state = self._reset_fn(keys)

        obs = np.array(self._state_to_obs(self._state))
        return obs

    def step(self, actions):
        """Step all environments.

        Args:
            actions: (num_envs, 3) NumPy array

        Returns:
            obs: (num_envs, 19) NumPy
            rewards: (num_envs,) NumPy
            dones: (num_envs,) NumPy bool
            infos: list of dicts (SB3-compatible)
        """
        actions_jax = jnp.array(actions)

        # Split PRNG
        self.rng, step_key = jax.random.split(self.rng)
        keys = jax.random.split(step_key, self.num_envs)

        # Batched physics step
        new_state, obs, rewards, dones, info = self._step_fn(
            self._state, actions_jax, keys,
        )

        # Auto-reset: sample fresh states for done envs
        self.rng, reset_key = jax.random.split(self.rng)
        reset_keys = jax.random.split(reset_key, self.num_envs)
        fresh_state = self._reset_fn(reset_keys)

        # Done envs get fresh state; continuing envs keep new_state
        self._state = jax.tree.map(
            lambda fresh, cont: jnp.where(
                dones[:, None] if fresh.ndim > 1 else dones,
                fresh, cont,
            ),
            fresh_state, new_state,
        )

        # Convert to NumPy
        obs_np = np.asarray(obs)
        rewards_np = np.asarray(rewards).astype(np.float32)
        dones_np = np.asarray(dones)

        # Build SB3-compatible infos
        arrived_np = np.asarray(info["arrived"])
        landed_np = np.asarray(info["landed"])
        crashed_np = np.asarray(info["crashed"])
        truncated_np = np.asarray(info["truncated"])
        dist_np = np.asarray(info["dist"])
        speed_np = np.asarray(info["speed"])
        impact_np = np.asarray(info["impact_speed"])
        scenario_np = np.asarray(info["scenario_id"])

        infos = []
        for i in range(self.num_envs):
            d = {}
            if dones_np[i]:
                if arrived_np[i]:
                    d["outcome"] = "arrived"
                    d["arrival_dist"] = float(dist_np[i])
                    d["arrival_speed"] = float(speed_np[i])
                elif landed_np[i]:
                    d["outcome"] = "landed"
                    d["landing_vel"] = float(impact_np[i])
                    d["landing_dist"] = float(dist_np[i])
                elif crashed_np[i]:
                    d["outcome"] = "crash"
                else:
                    d["outcome"] = "truncated"
                d["terminal_observation"] = obs_np[i]
                d["episode"] = {"r": float(rewards_np[i]), "l": 0}
                d["scenario_id"] = int(scenario_np[i])
                if truncated_np[i]:
                    d["TimeLimit.truncated"] = True
            infos.append(d)

        # For done envs, return the RESET observation (SB3 convention)
        reset_obs = np.asarray(self._state_to_obs(self._state))
        obs_out = np.where(dones_np[:, None], reset_obs, obs_np)

        return np.array(obs_out), np.array(rewards_np), np.array(dones_np), infos

    def close(self):
        pass

    @property
    def observation_space(self):
        import gymnasium as gym
        return gym.spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.float32)

    @property
    def action_space(self):
        import gymnasium as gym
        return gym.spaces.Box(
            low=np.array([0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )


# ──────────────────────────────────────────────────────────────
# SB3 VecEnv adapter
# ──────────────────────────────────────────────────────────────

class SB3JaxHopperVecEnv:
    """Stable-Baselines3 compatible VecEnv wrapping JaxHopperVecEnv.

    Usage:
        env = SB3JaxHopperVecEnv(num_envs=4096, scenarios=["vertical", "lateral", "landing"])
        model = PPO("MlpPolicy", env, ...)
        model.learn(total_timesteps=10_000_000)
    """

    def __new__(cls, **kwargs):
        from stable_baselines3.common.vec_env import VecEnv

        jax_env = JaxHopperVecEnv(**kwargs)

        class _SB3Adapter(VecEnv):
            def __init__(self, jax_env):
                self.jax_env = jax_env
                super().__init__(
                    num_envs=jax_env.num_envs,
                    observation_space=jax_env.observation_space,
                    action_space=jax_env.action_space,
                )
                self._actions = None

            def reset(self):
                return self.jax_env.reset()

            def step_async(self, actions):
                self._actions = actions

            def step_wait(self):
                return self.jax_env.step(self._actions)

            def close(self):
                self.jax_env.close()

            def env_method(self, method_name, *args, indices=None, **kwargs):
                return [getattr(self.jax_env, method_name)(*args, **kwargs)]

            def get_attr(self, attr_name, indices=None):
                val = getattr(self.jax_env, attr_name)
                n = len(indices) if indices else self.num_envs
                return [val] * n

            def set_attr(self, attr_name, value, indices=None):
                setattr(self.jax_env, attr_name, value)

            def env_is_wrapped(self, wrapper_class, indices=None):
                n = len(indices) if indices else self.num_envs
                return [False] * n

            def seed(self, seed=None):
                if seed is not None:
                    self.jax_env.rng = jax.random.PRNGKey(seed)

        return _SB3Adapter(jax_env)


# ──────────────────────────────────────────────────────────────
# Benchmark
# ──────────────────────────────────────────────────────────────

def benchmark(num_envs=4096, num_steps=500, use_guidance=True,
              scenarios=None):
    """Measure steps/sec for the JAX vectorized hopper env."""
    import time

    if scenarios is None:
        scenarios = ["vertical", "lateral", "diagonal", "landing", "liftoff"]

    env = JaxHopperVecEnv(
        num_envs=num_envs, seed=42,
        scenarios=scenarios,
        use_guidance=use_guidance,
    )
    obs = env.reset()

    # Warmup (JIT compilation)
    actions = np.random.uniform(0, 1, (num_envs, 3)).astype(np.float32)
    actions[:, 1:] = actions[:, 1:] * 2 - 1
    for _ in range(10):
        obs, _, _, _ = env.step(actions)

    # Timed run
    start = time.perf_counter()
    for _ in range(num_steps):
        obs, _, _, _ = env.step(actions)
    jax.block_until_ready(obs)
    elapsed = time.perf_counter() - start

    total_steps = num_envs * num_steps
    rate = total_steps / elapsed
    mode = "guidance" if use_guidance else "raw"
    print(f"JAX hopper env benchmark ({mode}, {len(scenarios)} scenarios):")
    print(f"  {num_envs} envs x {num_steps} steps = {total_steps:,} total")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Rate: {rate:,.0f} steps/sec")
    print(f"  Speedup over 8-env CPU: ~{rate / 1500:.0f}x")
    return rate


if __name__ == "__main__":
    benchmark(num_envs=4096, num_steps=500)
