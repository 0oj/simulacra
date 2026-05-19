"""Tests for safety envelope: checks, actuator saturation, and sensor."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from simulacra.onboard.safety_envelope import (
    SafetyCheck,
    SafetyEnvelopeConfig,
    SafetyEnvelope,
    ThrottleSaturationCheck,
    GimbalSaturationCheck,
    SpeedEnvelopeCheck,
    TiltMarginCheck,
    DescentRateCheck,
    default_checks,
)


# ──────────────────────────────────────────────────────────────
# SafetyEnvelopeConfig
# ──────────────────────────────────────────────────────────────

class TestSafetyEnvelopeConfig:
    def test_default_checks_signal_dim(self):
        config = SafetyEnvelopeConfig(default_checks())
        assert config.total_signal_dim == 6

    def test_obs_bounds_shape(self):
        config = SafetyEnvelopeConfig(default_checks())
        low, high = config.obs_bounds()
        assert low.shape == (6,)
        assert high.shape == (6,)
        assert np.all(low < high)

    def test_compute_all_signals_shape(self):
        config = SafetyEnvelopeConfig(default_checks())
        ctx = _make_ctx()
        signals = config.compute_all_signals(ctx)
        assert signals.shape == (6,)

    def test_compute_total_penalty_nonpositive(self):
        config = SafetyEnvelopeConfig(default_checks())
        ctx = _make_ctx(throttle_saturation=1.0)
        penalty = config.compute_total_penalty(ctx)
        assert penalty <= 0.0


# ──────────────────────────────────────────────────────────────
# Individual checks
# ──────────────────────────────────────────────────────────────

class TestThrottleSaturationCheck:
    def test_zero_saturation(self):
        check = ThrottleSaturationCheck()
        ctx = _make_ctx(throttle_saturation=0.0)
        assert check.compute_signal(ctx) == pytest.approx([0.0])
        assert check.compute_penalty(ctx) == 0.0

    def test_full_saturation(self):
        check = ThrottleSaturationCheck()
        ctx = _make_ctx(throttle_saturation=1.0)
        signal = check.compute_signal(ctx)
        assert signal == pytest.approx([1.0])
        assert check.compute_penalty(ctx) < 0.0

    def test_clips_extreme(self):
        check = ThrottleSaturationCheck()
        ctx = _make_ctx(throttle_saturation=10.0)
        signal = check.compute_signal(ctx)
        assert signal == pytest.approx([3.0])


class TestGimbalSaturationCheck:
    def test_zero_saturation(self):
        check = GimbalSaturationCheck()
        ctx = _make_ctx()
        signal = check.compute_signal(ctx)
        np.testing.assert_array_almost_equal(signal, [0.0, 0.0])

    def test_signal_dim(self):
        check = GimbalSaturationCheck()
        assert check.signal_dim == 2


class TestSpeedEnvelopeCheck:
    def test_safe_speed(self):
        check = SpeedEnvelopeCheck()
        ctx = _make_ctx(speed=5.0, v_safe=20.0)
        signal = check.compute_signal(ctx)
        assert 0.0 < signal[0] < 1.0

    def test_excess_speed(self):
        check = SpeedEnvelopeCheck()
        ctx = _make_ctx(speed=30.0, v_safe=10.0)
        signal = check.compute_signal(ctx)
        assert signal[0] == pytest.approx(3.0)  # clipped

    def test_no_penalty(self):
        """SpeedEnvelopeCheck should never penalize (avoids double-counting)."""
        check = SpeedEnvelopeCheck()
        ctx = _make_ctx(speed=100.0, v_safe=1.0)
        assert check.compute_penalty(ctx) == 0.0


class TestTiltMarginCheck:
    def test_upright(self):
        check = TiltMarginCheck()
        ctx = _make_ctx(tilt=0.0, max_tilt=1.0)
        signal = check.compute_signal(ctx)
        assert signal[0] == pytest.approx(1.0)

    def test_at_limit(self):
        check = TiltMarginCheck()
        ctx = _make_ctx(tilt=1.0, max_tilt=1.0)
        signal = check.compute_signal(ctx)
        assert signal[0] == pytest.approx(0.0)


class TestDescentRateCheck:
    def test_ascending(self):
        check = DescentRateCheck()
        ctx = _make_ctx(vy=5.0, alt=50.0)
        signal = check.compute_signal(ctx)
        assert signal[0] == pytest.approx(0.0)  # ascending, no descent

    def test_safe_descent(self):
        check = DescentRateCheck()
        ctx = _make_ctx(vy=-2.0, alt=50.0)
        signal = check.compute_signal(ctx)
        assert 0.0 < signal[0] < 1.0

    def test_dangerous_descent_penalizes(self):
        check = DescentRateCheck()
        ctx = _make_ctx(vy=-20.0, alt=2.0)
        penalty = check.compute_penalty(ctx)
        assert penalty < 0.0


# ──────────────────────────────────────────────────────────────
# Actuator saturation properties
# ──────────────────────────────────────────────────────────────

class TestActuatorSaturation:
    def test_throttle_saturation_zero_when_honored(self):
        """When command equals actual, saturation should be ~0."""
        from simulacra.noumena.rocket import Rocket, RocketProps, RocketState
        from simulacra.phenomena.actuators import ThrottleActuator, ThrottleCommand
        from simulacra.physics import RocketVec3

        rocket = _make_rocket()
        actuator = ThrottleActuator(rocket, max_thrust_rate=4000.0)
        # Command a small change that can be honored in one step
        actuator.apply(ThrottleCommand(thrust_N=50.0))
        actuator.step(0.02)  # max_delta = 80N, so 50N is honored
        assert abs(actuator.saturation_ratio) < 0.01

    def test_throttle_saturation_nonzero_when_clamped(self):
        """When command exceeds rate limit, saturation should be nonzero."""
        from simulacra.noumena.rocket import Rocket, RocketProps, RocketState
        from simulacra.phenomena.actuators import ThrottleActuator, ThrottleCommand

        rocket = _make_rocket()
        actuator = ThrottleActuator(rocket, max_thrust_rate=4000.0)
        # Command a large change: 2000N, max_delta = 80N/step
        actuator.apply(ThrottleCommand(thrust_N=2000.0))
        actuator.step(0.02)
        # After step, actual should be ~80N, commanded is 2000N
        # saturation = (2000 - 80) / 80 ≈ 24
        assert actuator.saturation_ratio > 1.0

    def test_gimbal_saturation_properties_exist(self):
        from simulacra.noumena.rocket import Rocket, RocketProps, RocketState
        from simulacra.phenomena.actuators import GimbalActuator, GimbalCommand

        rocket = _make_rocket()
        gimbal = GimbalActuator(rocket, max_outer_angle=0.26, max_inner_angle=0.26,
                                max_rate=1.0, tau=0.05)
        gimbal.apply(GimbalCommand(desired_outer_angle=0.2, desired_inner_angle=0.1))
        gimbal.step(0.02)
        # Properties should exist and be floats
        assert isinstance(gimbal.outer_saturation, float)
        assert isinstance(gimbal.inner_saturation, float)


# ──────────────────────────────────────────────────────────────
# JAX env obs dim
# ──────────────────────────────────────────────────────────────

class TestJaxEnvObsDim:
    def test_obs_shape_28(self):
        from simulacra.jax_hopper_env import JaxHopperVecEnv
        env = JaxHopperVecEnv(num_envs=4, scenarios=["vertical"], seed=42)
        obs = env.reset()
        assert obs.shape == (4, 28), f"Expected (4, 28), got {obs.shape}"

    def test_obs_in_bounds(self):
        from simulacra.jax_hopper_env import JaxHopperVecEnv
        env = JaxHopperVecEnv(num_envs=4, scenarios=["vertical"], seed=42)
        obs = env.reset()
        for i in range(4):
            assert env.observation_space.contains(obs[i]), (
                f"Env {i}: obs not in space"
            )

    def test_step_produces_28_dim(self):
        from simulacra.jax_hopper_env import JaxHopperVecEnv
        env = JaxHopperVecEnv(num_envs=4, scenarios=["vertical"], seed=42)
        env.reset()
        actions = np.array([[0.7, 0.0, 0.0]] * 4, dtype=np.float32)
        obs, rewards, dones, infos = env.step(actions)
        assert obs.shape == (4, 28)

    def test_safety_signals_vary_after_steps(self):
        """After several steps with aggressive actions, saturation signals should be nonzero."""
        from simulacra.jax_hopper_env import JaxHopperVecEnv
        env = JaxHopperVecEnv(num_envs=8, scenarios=["vertical"], seed=42)
        env.reset()
        # Alternate between full throttle and zero to trigger saturation
        for i in range(10):
            throttle = 1.0 if i % 2 == 0 else 0.0
            actions = np.array([[throttle, 0.5, -0.5]] * 8, dtype=np.float32)
            obs, _, _, _ = env.step(actions)
        # Safety signals are obs[:, 22:28]
        safety = obs[:, 22:28]
        # At least some signals should be nonzero
        assert np.any(np.abs(safety) > 0.01), f"Safety signals all zero: {safety}"


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _make_ctx(**overrides):
    """Create a default safety check context dict."""
    ctx = {
        "throttle_saturation": 0.0,
        "gimbal_outer_saturation": 0.0,
        "gimbal_inner_saturation": 0.0,
        "speed": 5.0,
        "v_safe": 20.0,
        "tilt": 0.1,
        "max_tilt": 1.0,
        "vy": -2.0,
        "alt": 50.0,
    }
    ctx.update(overrides)
    return ctx


def _make_rocket():
    """Create a minimal rocket for actuator tests."""
    from simulacra.noumena.rocket import Rocket, RocketProps, RocketState
    from simulacra.physics import RocketVec3, WorldVec3, Orientation

    props = RocketProps(
        length=4.0,
        mass=140.0,
        center_of_mass_offset=RocketVec3(2.0, 0.0, 0.0),
        inertia_diag=(50.0, 300.0, 300.0),
        thrust_application_offset=RocketVec3(0.0, 0.0, 0.0),
        drag_coefficient=0.5,
        reference_area=0.0314,
    )
    state = RocketState(
        position=WorldVec3(0.0, 50.0, 0.0),
        velocity=WorldVec3(0.0, 0.0, 0.0),
        orientation=Orientation.from_wxyz(0.707, 0.0, 0.0, 0.707),
        angular_velocity_body=RocketVec3(0.0, 0.0, 0.0),
        thrust_body=RocketVec3(0.0, 0.0, 0.0),
    )
    return Rocket(props=props, state=state)
