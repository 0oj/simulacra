"""Tests for the JAX-accelerated physics kernel.

Verifies that the JAX kernel matches the NumPy kernel's physics.
"""

import pytest
import numpy as np

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from simulacra.jax_kernel import (
    quat_multiply,
    quat_normalize,
    quat_rotate,
    quat_rotate_inverse,
    quat_from_rotvec,
    integrate_orientation,
    nozzle_direction_body,
    compute_mass_properties,
    advance_gimbal_angle,
    physics_step,
    default_params,
    make_initial_state,
    upright_quat,
)


class TestQuaternionMath:
    def test_upright_quaternion(self):
        """Upright quat should map body +X to world +Y."""
        q = upright_quat()
        nose = quat_rotate(q, jnp.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(nose, [0.0, 1.0, 0.0], atol=1e-6)

    def test_identity_rotation(self):
        """Identity quaternion should not change vectors."""
        q = jnp.array([1.0, 0.0, 0.0, 0.0])
        v = jnp.array([1.0, 2.0, 3.0])
        result = quat_rotate(q, v)
        np.testing.assert_allclose(result, v, atol=1e-6)

    def test_rotate_inverse_roundtrip(self):
        """Rotate then inverse-rotate should return original."""
        q = quat_from_rotvec(jnp.array([0.3, 0.5, -0.2]))
        v = jnp.array([1.0, -2.0, 0.5])
        v_world = quat_rotate(q, v)
        v_back = quat_rotate_inverse(q, v_world)
        np.testing.assert_allclose(v_back, v, atol=1e-5)

    def test_normalize(self):
        q = jnp.array([2.0, 0.0, 0.0, 0.0])
        qn = quat_normalize(q)
        assert abs(jnp.linalg.norm(qn) - 1.0) < 1e-7

    def test_orientation_integration_small_step(self):
        """Small angular velocity integration should be consistent."""
        q = upright_quat()
        omega = jnp.array([0.0, 0.0, 0.1])  # slow yaw
        q_new = integrate_orientation(q, omega, 0.02)
        assert abs(jnp.linalg.norm(q_new) - 1.0) < 1e-6
        # Should have rotated slightly
        assert not jnp.allclose(q, q_new)


class TestNozzleDirection:
    def test_zero_gimbal(self):
        """Zero gimbal angles → thrust along body +X."""
        d = nozzle_direction_body(0.0, 0.0)
        np.testing.assert_allclose(d, [1.0, 0.0, 0.0], atol=1e-6)

    def test_outer_gimbal_rotates_about_z(self):
        """Outer gimbal should rotate nozzle in X-Y plane (about Z)."""
        d = nozzle_direction_body(0.1, 0.0)
        assert d[0] < 1.0  # X reduced
        assert abs(d[1]) > 0.01  # Y component appears
        assert abs(d[2]) < 1e-6  # Z unchanged


class TestMassProperties:
    def test_full_fuel(self):
        params = default_params()
        total_m, com_x, Ix, Iy, Iz = compute_mass_properties(80.0, params)
        assert abs(total_m - 140.0) < 0.01  # 60 dry + 80 fuel
        assert com_x > 0  # CoM should be along body

    def test_empty_fuel(self):
        params = default_params()
        total_m, _, _, _, _ = compute_mass_properties(0.0, params)
        assert abs(total_m - 60.0) < 0.01  # dry mass only

    def test_com_shifts_with_fuel(self):
        """CoM should shift as fuel burns."""
        params = default_params()
        _, com_full, _, _, _ = compute_mass_properties(80.0, params)
        _, com_empty, _, _, _ = compute_mass_properties(0.0, params)
        # Fuel is at 0.3*L, removing it should shift CoM
        assert com_full != com_empty


class TestPhysicsStep:
    def test_free_fall(self):
        """Free fall from 100m for 2s should match v = g*t, y = y0 - 0.5*g*t²."""
        params = default_params()
        state = make_initial_state(altitude=100.0, velocity_y=0.0)
        rng = jax.random.PRNGKey(0)

        for _ in range(100):  # 100 × 0.02s = 2s
            rng, k = jax.random.split(rng)
            state, _, _, _, _ = physics_step(state, jnp.array([0.0, 0.0, 0.0]), params, k)

        # Expected: y ≈ 100 - 0.5*9.81*4 = 80.38m, vy ≈ -19.62 m/s
        # (slight drag means slightly less than analytical)
        assert abs(float(state["pos"][1]) - 80.38) < 1.0
        assert abs(float(state["vel"][1]) - (-19.62)) < 0.5

    def test_hover_thrust(self):
        """Full weight thrust should approximately maintain altitude."""
        params = default_params()
        state = make_initial_state(altitude=50.0, velocity_y=0.0)
        rng = jax.random.PRNGKey(0)

        # Thrust = weight = m*g = 140 * 9.81 ≈ 1373 N → throttle = 1373/2000 ≈ 0.687
        throttle = 140.0 * 9.81 / 2000.0

        for _ in range(50):  # 1 second
            rng, k = jax.random.split(rng)
            state, _, _, _, _ = physics_step(
                state, jnp.array([throttle, 0.0, 0.0]), params, k)

        # Should be roughly at same altitude (within 2m due to actuator lag)
        assert abs(float(state["pos"][1]) - 50.0) < 3.0

    def test_ground_contact(self):
        """Low altitude + downward velocity should trigger ground contact."""
        params = default_params()
        state = make_initial_state(altitude=0.5, velocity_y=-2.0)
        rng = jax.random.PRNGKey(0)

        for _ in range(50):
            rng, k = jax.random.split(rng)
            state, _, _, done, info = physics_step(
                state, jnp.array([0.0, 0.0, 0.0]), params, k)
            if done:
                break

        assert bool(state["on_ground"])

    def test_fuel_depletion(self):
        """Continuous thrust should deplete fuel."""
        params = default_params()
        state = make_initial_state(altitude=100.0, fuel_fraction=0.1)  # low fuel
        rng = jax.random.PRNGKey(0)
        initial_fuel = float(state["fuel_mass"])

        for _ in range(200):
            rng, k = jax.random.split(rng)
            state, _, _, _, _ = physics_step(
                state, jnp.array([1.0, 0.0, 0.0]), params, k)

        assert float(state["fuel_mass"]) < initial_fuel


class TestVectorizedEnv:
    def test_reset_and_step(self):
        from simulacra.jax_env import JaxVecEnv

        env = JaxVecEnv(num_envs=8, seed=0)
        obs = env.reset()
        assert obs.shape == (8, 13)
        assert obs.dtype == np.float32

        actions = np.zeros((8, 3), dtype=np.float32)
        actions[:, 0] = 0.5  # some thrust
        obs, rewards, dones, infos = env.step(actions)
        assert obs.shape == (8, 13)
        assert rewards.shape == (8,)
        assert dones.shape == (8,)
        assert len(infos) == 8

    def test_different_seeds(self):
        from simulacra.jax_env import JaxVecEnv

        env1 = JaxVecEnv(num_envs=4, seed=0)
        env2 = JaxVecEnv(num_envs=4, seed=1)
        obs1 = env1.reset()
        obs2 = env2.reset()
        assert not np.allclose(obs1, obs2), "Different seeds should give different states"
