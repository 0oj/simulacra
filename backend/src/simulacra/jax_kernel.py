"""
JAX-accelerated physics kernel for Simulacra.

Pure-function implementation of the rocket physics, designed for:
  - jax.vmap: vectorise across thousands of parallel environments
  - jax.jit: fuse the entire step into a single compiled kernel
  - GPU execution: entire simulation runs on-device, no host transfer

All state is represented as flat JAX arrays in a pytree-compatible dict.
All branching uses jnp.where (branchless on GPU).

Physics matches the NumPy kernel in noumena/kernel.py exactly:
  - Semi-implicit Euler integration
  - Euler's rigid body equations (diagonal inertia)
  - Quaternion orientation (w, x, y, z) with body-frame angular velocity
  - Gimbal actuator: rate-limited + first-order lag
  - Throttle actuator: rate-limited with deadband
  - Ground contact with crash/landing detection
"""

import jax
import jax.numpy as jnp
from functools import partial


# ──────────────────────────────────────────────────────────────
# Quaternion math (pure JAX, scalar-first [w, x, y, z])
# ──────────────────────────────────────────────────────────────

def quat_multiply(q1, q2):
    """Hamilton product: q1 ⊗ q2. Both (4,) arrays [w, x, y, z]."""
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return jnp.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_normalize(q):
    """Normalize quaternion to unit length."""
    return q / jnp.linalg.norm(q)


def quat_conjugate(q):
    """Conjugate (inverse for unit quaternions): [w, -x, -y, -z]."""
    return q * jnp.array([1.0, -1.0, -1.0, -1.0])


def quat_rotate(q, v):
    """Rotate vector v by quaternion q (body → world). v is (3,)."""
    # q ⊗ [0, v] ⊗ q*
    qv = jnp.array([0.0, v[0], v[1], v[2]])
    return quat_multiply(quat_multiply(q, qv), quat_conjugate(q))[1:]


def quat_rotate_inverse(q, v):
    """Rotate vector v by q* (world → body). v is (3,)."""
    return quat_rotate(quat_conjugate(q), v)


def quat_from_rotvec(rotvec):
    """Quaternion from rotation vector (axis * angle). rotvec is (3,)."""
    angle = jnp.linalg.norm(rotvec)
    # For tiny angles, use first-order Taylor expansion
    half = angle / 2.0
    # sin(half)/angle ≈ 0.5 for small angles
    scale = jnp.where(angle > 1e-8, jnp.sin(half) / angle, 0.5)
    w = jnp.where(angle > 1e-8, jnp.cos(half), 1.0)
    return jnp.array([w, rotvec[0] * scale, rotvec[1] * scale, rotvec[2] * scale])


def integrate_orientation(quat, omega_body, dt):
    """Integrate orientation using body-frame angular velocity.

    Matches noumena/kernel.py: R_new = R_current ⊗ R_delta
    where R_delta = exp(ω_body × dt).
    """
    dq = quat_from_rotvec(omega_body * dt)
    return quat_normalize(quat_multiply(quat, dq))


# ──────────────────────────────────────────────────────────────
# Nozzle direction (gimbal kinematics)
# ──────────────────────────────────────────────────────────────

def _rotation_matrix_z(angle):
    """Rotation matrix about Z axis."""
    c, s = jnp.cos(angle), jnp.sin(angle)
    return jnp.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def _rotation_about_axis(axis, angle):
    """Rodrigues' rotation formula: rotation matrix about arbitrary axis."""
    axis = axis / jnp.linalg.norm(axis)
    c, s = jnp.cos(angle), jnp.sin(angle)
    ux, uy, uz = axis[0], axis[1], axis[2]
    return jnp.array([
        [c + ux*ux*(1-c),    ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
        [uy*ux*(1-c) + uz*s, c + uy*uy*(1-c),     uy*uz*(1-c) - ux*s],
        [uz*ux*(1-c) - uy*s, uz*uy*(1-c) + ux*s,  c + uz*uz*(1-c)],
    ])


def nozzle_direction_body(outer_angle, inner_angle):
    """Compute thrust direction in body frame from gimbal angles.

    Matches phenomena/actuators.py GimbalActuator._nozzle_direction_body.
    """
    # Outer rotation about body +Z
    r_outer = _rotation_matrix_z(outer_angle)

    # Inner rotation axis: +Y in outer frame, expressed in body frame
    axis_inner_body = r_outer @ jnp.array([0.0, 1.0, 0.0])
    r_inner = _rotation_about_axis(axis_inner_body, inner_angle)

    # Compose: nozzle = inner(outer([1,0,0]))
    direction = r_inner @ (r_outer @ jnp.array([1.0, 0.0, 0.0]))
    return direction / jnp.linalg.norm(direction)


# ──────────────────────────────────────────────────────────────
# Mass model (analytical, matching noumena/mass_model.py)
# ──────────────────────────────────────────────────────────────

def compute_mass_properties(fuel_mass, params):
    """Compute total mass, CoM, and inertia from current fuel mass.

    2-component model matching NumPy kernel (gym_env.py / hopper_env.py):
      dry:  dry_mass at length/2     (centre of rocket)
      fuel: fuel_mass at 0.3*length  (forward of engine)

    Returns: (total_mass, com_x, Ix, Iy, Iz)
    """
    dry = params["dry_mass"]
    length = params["length"]

    # 2-component model — matches NumPy MassModel exactly
    masses = jnp.array([dry, fuel_mass])
    positions_x = jnp.array([length / 2, 0.3 * length])

    total_mass = jnp.sum(masses)
    com_x = jnp.sum(masses * positions_x) / total_mass

    # Parallel-axis theorem (all components on X-axis, so dy=dz=0)
    dx = positions_x - com_x
    Iy = jnp.sum(masses * dx**2)
    Iz = Iy  # axial symmetry

    # Minimum inertia floor — matches NumPy inertia.py:
    # min_ix_fraction * total_m * max(1.0, max_dist_sq)
    max_dist_sq = jnp.max(dx**2)
    min_inertia = 0.01 * total_mass * jnp.maximum(1.0, max_dist_sq)
    Ix = min_inertia  # roll axis — always at floor for slender rocket
    Iy = jnp.maximum(Iy, min_inertia)
    Iz = jnp.maximum(Iz, min_inertia)

    return total_mass, com_x, Ix, Iy, Iz


# ──────────────────────────────────────────────────────────────
# Actuator dynamics (pure functions)
# ──────────────────────────────────────────────────────────────

def advance_gimbal_angle(actual, commanded, max_angle, max_rate, tau, dt):
    """Advance a single gimbal angle: first-order lag + rate limit.

    Matches phenomena/actuators.py GimbalActuator._advance_angle.
    """
    commanded = jnp.clip(commanded, -max_angle, max_angle)

    # First-order lag
    alpha = jnp.minimum(1.0, dt / jnp.maximum(tau, 1e-6))
    target = actual + alpha * (commanded - actual)

    # Rate limit
    max_delta = max_rate * dt
    delta = jnp.clip(target - actual, -max_delta, max_delta)

    return actual + delta


def advance_throttle(actual, commanded, max_thrust, max_rate, min_thrust, dt):
    """Advance throttle with rate limit and deadband.

    Matches phenomena/actuators.py ThrottleActuator.step.
    """
    commanded = jnp.clip(commanded, 0.0, max_thrust)

    # Rate limit
    max_delta = max_rate * dt
    delta = jnp.clip(commanded - actual, -max_delta, max_delta)
    new_thrust = actual + delta

    # Deadband: snap to 0 below minimum
    new_thrust = jnp.where(new_thrust < min_thrust, 0.0, new_thrust)
    return new_thrust


# ──────────────────────────────────────────────────────────────
# Physics step (the core kernel)
# ──────────────────────────────────────────────────────────────

def physics_step(state, action, params, rng_key):
    """Pure-function physics step.

    Args:
        state: dict with keys:
            pos (3,), vel (3,), quat (4,), omega (3,),
            gimbal_outer, gimbal_inner, actual_thrust,
            fuel_mass, on_ground, step_count
        action: (3,) array [throttle_norm, gimbal_outer_norm, gimbal_inner_norm]
            throttle_norm ∈ [0, 1], gimbal norms ∈ [-1, 1]
        params: dict of static environment parameters
        rng_key: JAX PRNG key (for sensor noise)

    Returns:
        new_state, obs (13,), reward, terminated, info
    """
    dt = params["dt"]
    pos = state["pos"]
    vel = state["vel"]
    quat = state["quat"]
    omega = state["omega"]

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

    # Cut thrust if out of fuel
    actual_thrust = jnp.where(fuel_mass <= 0.0, 0.0, actual_thrust)

    # ── Mass properties (depend on current fuel) ──
    total_mass, com_x, Ix, Iy, Iz = compute_mass_properties(fuel_mass, params)

    # ── Thrust vector in body frame ──
    nozzle_dir = nozzle_direction_body(gimbal_outer, gimbal_inner)
    thrust_body = actual_thrust * nozzle_dir

    # ── Kill thrust if on ground ──
    thrust_body = jnp.where(state["on_ground"], jnp.zeros(3), thrust_body)
    actual_thrust = jnp.where(state["on_ground"], 0.0, actual_thrust)

    # ── Translational dynamics (world frame) ──
    thrust_world = quat_rotate(quat, thrust_body)
    gravity_force = jnp.array([0.0, params["gravity"] * total_mass, 0.0])
    wind_force = params["wind"]  # already in world frame, pre-scaled

    # Drag
    speed = jnp.linalg.norm(vel)
    drag_coeff = -0.5 * params["air_density"] * params["Cd"] * params["ref_area"]
    drag_force = jnp.where(speed > 1e-6, drag_coeff * speed * vel, jnp.zeros(3))

    net_force = thrust_world + gravity_force + wind_force + drag_force
    accel = net_force / total_mass

    # Semi-implicit Euler (velocity first, then position)
    vel_new = vel + accel * dt
    pos_new = pos + vel_new * dt

    # ── Rotational dynamics (body frame, Euler's equations) ──
    # Torque from off-axis thrust
    thrust_offset = jnp.array([0.0, 0.0, 0.0]) - jnp.array([com_x, 0.0, 0.0])
    # thrust_application_offset is at [0,0,0] (engine base)
    torque = jnp.cross(thrust_offset, thrust_body)

    # Euler's equations for diagonal inertia tensor
    omega_dot = jnp.array([
        (torque[0] + (Iy - Iz) * omega[1] * omega[2]) / Ix,
        (torque[1] + (Iz - Ix) * omega[2] * omega[0]) / Iy,
        (torque[2] + (Ix - Iy) * omega[0] * omega[1]) / Iz,
    ])
    omega_new = omega + omega_dot * dt

    # Integrate orientation
    quat_new = integrate_orientation(quat, omega_new, dt)

    # ── Ground contact ──
    touching = pos_new[1] <= 0.0

    # Impact metrics (latched at moment of contact)
    was_airborne = jnp.logical_not(state["on_ground"])
    just_landed = was_airborne & touching

    impact_speed = jnp.linalg.norm(vel_new)
    nose_world = quat_rotate(quat_new, jnp.array([1.0, 0.0, 0.0]))
    cos_tilt = jnp.clip(nose_world[1], -1.0, 1.0)
    tilt_angle = jnp.arccos(cos_tilt)
    horiz_dist = jnp.sqrt(pos_new[0]**2 + pos_new[2]**2)

    crashed = just_landed & (impact_speed > params["max_landing_speed"])
    tipped = just_landed & (tilt_angle > params["max_landing_tilt"])
    landed_ok = just_landed & ~crashed & ~tipped

    # Freeze state on ground
    pos_new = jnp.where(touching, pos_new.at[1].set(0.0), pos_new)
    vel_new = jnp.where(touching, jnp.zeros(3), vel_new)
    omega_new = jnp.where(touching, jnp.zeros(3), omega_new)

    # ── Observation (with sensor noise) ──
    k1, k2, k3, k4 = jax.random.split(rng_key, 4)
    gps_noise = params["gps_noise_std"] * jax.random.normal(k1, (2,))
    baro_noise = params["baro_noise_std"] * jax.random.normal(k2, (1,))
    vel_noise = params["vel_noise_std"] * jax.random.normal(k3, (3,))
    gyro_noise = params["gyro_noise_std"] * jax.random.normal(k4, (3,))

    obs = jnp.array([
        pos_new[0] + gps_noise[0],    # x (GPS)
        pos_new[1] + baro_noise[0],    # y (baro)
        pos_new[2] + gps_noise[1],     # z (GPS)
        vel_new[0] + vel_noise[0],     # vx
        vel_new[1] + vel_noise[1],     # vy
        vel_new[2] + vel_noise[2],     # vz
        quat_new[0],                   # qw (orientation sensor — low noise)
        quat_new[1],                   # qx
        quat_new[2],                   # qy
        quat_new[3],                   # qz
        omega_new[0] + gyro_noise[0],  # wx
        omega_new[1] + gyro_noise[1],  # wy
        omega_new[2] + gyro_noise[2],  # wz
    ])

    # ── Reward (matches GuidanceReward from train_guidance.py) ──
    alt = pos_new[1]
    vy = vel_new[1]
    v_horiz = jnp.sqrt(vel_new[0]**2 + vel_new[2]**2)
    omega_mag = jnp.linalg.norm(omega_new)

    # Terminal rewards
    terminal_reward = jnp.where(
        landed_ok,
        200.0
        + 50.0 * jnp.maximum(0.0, 1.0 - impact_speed / 3.0)
        + 20.0 * jnp.maximum(0.0, 1.0 - tilt_angle / 0.26)
        + 80.0 * jnp.maximum(0.0, 1.0 - horiz_dist / 3.0),
        0.0,
    )
    terminal_reward = jnp.where(crashed | tipped, -200.0, terminal_reward)

    # Dense shaping (flight phase)
    max_decel = 8.0
    max_safe_vy = jnp.sqrt(2.0 * max_decel * jnp.maximum(0.1, alt))
    excess = jnp.maximum(0.0, jnp.abs(vy) - max_safe_vy)

    descent_reward = jnp.where(vy < 0.0, 0.03 * jnp.minimum(jnp.abs(vy), max_safe_vy), 0.0)
    # Near-ground bonus/penalty
    near_ground = alt < 2.0
    descent_reward = jnp.where(
        near_ground & (vy > -1.5) & (vy < 0.0),
        descent_reward + 0.15, descent_reward,
    )
    descent_reward = jnp.where(
        near_ground & (vy < -3.0),
        descent_reward - 0.5, descent_reward,
    )

    # Potential-based altitude shaping: reward descent progress
    prev_alt = state["pos"][1]
    alt_shaping = 0.005 * (prev_alt - alt)

    # Potential-based horizontal approach
    prev_horiz = jnp.sqrt(state["pos"][0]**2 + state["pos"][2]**2)
    approach_reward = 0.15 * (prev_horiz - horiz_dist)

    flight_reward = (
        alt_shaping
        + descent_reward
        - 0.5 * excess**2
        - 0.05 * tilt_angle**2
        - 0.03 * omega_mag
        - 0.01 * v_horiz
        - 0.02 * horiz_dist
        + approach_reward
    )

    # Use terminal reward on ground, flight reward otherwise
    reward = jnp.where(just_landed, terminal_reward, flight_reward)

    # ── Termination ──
    step_count = state["step_count"] + 1
    truncated = step_count >= params["max_episode_steps"]
    tilt_exceeded = tilt_angle > params["max_tilt"]
    alt_exceeded = pos_new[1] > params["max_altitude"]
    terminated = touching | tilt_exceeded | alt_exceeded

    # Truncation penalty
    reward = jnp.where(
        truncated & ~terminated,
        -50.0 - 0.5 * alt,
        reward,
    )
    # Tilt/altitude termination penalty
    reward = jnp.where(
        (tilt_exceeded | alt_exceeded) & ~touching,
        -100.0,
        reward,
    )

    done = terminated | truncated

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
    }

    info = {
        "landed": landed_ok,
        "crashed": crashed | tipped,
        "truncated": truncated & ~terminated,  # for SB3 TimeLimit handling
        "impact_speed": jnp.where(just_landed, impact_speed, 0.0),
        "tilt_angle": tilt_angle,
        "horiz_dist": horiz_dist,
    }

    return new_state, obs, reward, done, info


# ──────────────────────────────────────────────────────────────
# Default parameters
# ──────────────────────────────────────────────────────────────

def default_params(
    dt=0.02,
    max_episode_steps=1000,
    max_altitude=750.0,
    # Rocket
    dry_mass=60.0,
    fuel_capacity=80.0,
    length=4.0,
    max_thrust=2000.0,
    burn_rate=4.0,
    max_gimbal=0.26,
    gimbal_rate=1.0,
    gimbal_tau=0.05,
    max_thrust_rate=4000.0,   # max_thrust * 2 — matches NumPy env (0→100% in 0.5s)
    min_thrust=0.0,           # no deadband — matches NumPy env
    # Landing
    max_landing_speed=3.0,
    max_landing_tilt=0.26,
    max_tilt=1.0,
    # Environment
    gravity=-9.81,
    air_density=1.225,
    Cd=0.5,
    ref_area=0.0314,  # π × 0.1²
    # Sensor noise
    gps_noise_std=0.0,
    baro_noise_std=0.0,
    vel_noise_std=0.0,
    gyro_noise_std=0.0,
):
    """Create parameter dict. All values are JAX-compatible scalars."""
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
        "gravity": gravity,
        "air_density": air_density,
        "Cd": Cd,
        "ref_area": ref_area,
        "wind": jnp.zeros(3),
        "gps_noise_std": gps_noise_std,
        "baro_noise_std": baro_noise_std,
        "vel_noise_std": vel_noise_std,
        "gyro_noise_std": gyro_noise_std,
    }


# ──────────────────────────────────────────────────────────────
# Upright orientation
# ──────────────────────────────────────────────────────────────

def upright_quat():
    """Upright quaternion: body +X → world +Y (90° about Z)."""
    return quat_from_rotvec(jnp.array([0.0, 0.0, jnp.pi / 2.0]))


def make_initial_state(
    altitude=50.0,
    velocity_y=0.0,
    horizontal_x=0.0,
    horizontal_z=0.0,
    tilt_angle=0.0,
    fuel_fraction=1.0,
    fuel_capacity=80.0,
):
    """Create an initial state dict."""
    # Apply tilt perturbation to upright quaternion
    tilt_rotvec = jnp.array([0.0, tilt_angle, 0.0])  # tilt about Y
    base_quat = upright_quat()
    tilt_quat = quat_from_rotvec(tilt_rotvec)
    quat = quat_normalize(quat_multiply(base_quat, tilt_quat))

    return {
        "pos": jnp.array([horizontal_x, altitude, horizontal_z]),
        "vel": jnp.array([0.0, velocity_y, 0.0]),
        "quat": quat,
        "omega": jnp.zeros(3),
        "gimbal_outer": 0.0,
        "gimbal_inner": 0.0,
        "actual_thrust": 0.0,
        "fuel_mass": fuel_fraction * fuel_capacity,
        "on_ground": False,
        "step_count": 0,
    }
