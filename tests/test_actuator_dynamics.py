"""
Tests for actuator dynamics (rate limiting, servo lag, deadband).

Verifies that:
- Default actuators are instantaneous (infinite rates)
- Rate limiting clamps changes per timestep
- First-order lag converges over time
- Deadband snaps thrust to zero below threshold
"""

import pytest
import numpy as np

from simulacra.noumena import Kernel, Environment
from simulacra.noumena.rocket import Rocket, RocketProps, RocketState
from simulacra.phenomena.actuators import (
    GimbalActuator,
    GimbalCommand,
    ThrottleActuator,
    ThrottleCommand,
    Actuators,
)
from simulacra.physics import Orientation, RocketVec3, WorldVec3


@pytest.fixture
def rocket():
    """Create a rocket for testing."""
    props = RocketProps(
        length=10.0,
        mass=1000.0,
        center_of_mass_offset=RocketVec3(5.0, 0.0, 0.0),
        inertia_diag=(100.0, 1000.0, 1000.0),
        thrust_application_offset=RocketVec3(0.0, 0.0, 0.0),
    )
    state = RocketState(
        position=WorldVec3(0.0, 0.0, 0.0),
        velocity=WorldVec3(0.0, 0.0, 0.0),
        orientation=Orientation.identity(),
        angular_velocity_body=RocketVec3(0.0, 0.0, 0.0),
        thrust_body=RocketVec3(5000.0, 0.0, 0.0),
    )
    return Rocket(props, state)


class TestGimbalActuatorInstantaneous:
    """Tests for GimbalActuator with default (instantaneous) dynamics."""

    def test_instantaneous_response(self, rocket):
        """With infinite rate and no lag, angles reach commanded instantly."""
        gimbal = GimbalActuator(
            rocket=rocket,
            max_outer_angle=np.deg2rad(15.0),
            max_inner_angle=np.deg2rad(15.0),
        )

        # Command 10 degrees
        gimbal.apply(GimbalCommand(
            desired_outer_angle=np.deg2rad(10.0),
            desired_inner_angle=np.deg2rad(5.0),
        ))
        gimbal.step(0.01)

        assert gimbal.outer_angle == pytest.approx(np.deg2rad(10.0))
        assert gimbal.inner_angle == pytest.approx(np.deg2rad(5.0))

    def test_saturation(self, rocket):
        """Angles are clamped to max values."""
        gimbal = GimbalActuator(
            rocket=rocket,
            max_outer_angle=np.deg2rad(15.0),
            max_inner_angle=np.deg2rad(15.0),
        )

        # Command 30 degrees (exceeds max)
        gimbal.apply(GimbalCommand(
            desired_outer_angle=np.deg2rad(30.0),
            desired_inner_angle=np.deg2rad(30.0),
        ))
        gimbal.step(0.01)

        assert gimbal.outer_angle == pytest.approx(np.deg2rad(15.0))
        assert gimbal.inner_angle == pytest.approx(np.deg2rad(15.0))


class TestGimbalActuatorRateLimiting:
    """Tests for GimbalActuator rate limiting."""

    def test_rate_limit_clamps_change(self, rocket):
        """With rate limit, angle changes at most max_rate * dt per step."""
        max_rate = np.deg2rad(5.0)  # 5 deg/s
        gimbal = GimbalActuator(
            rocket=rocket,
            max_outer_angle=np.deg2rad(15.0),
            max_inner_angle=np.deg2rad(15.0),
            max_rate=max_rate,
        )

        # Command instant 15-deg change
        gimbal.apply(GimbalCommand(
            desired_outer_angle=np.deg2rad(15.0),
            desired_inner_angle=0.0,
        ))

        # After 1 second at 5 deg/s, should be at 5 degrees
        for _ in range(100):  # 100 * 0.01 = 1 second
            gimbal.step(0.01)

        assert gimbal.outer_angle == pytest.approx(np.deg2rad(5.0), abs=1e-6)

    def test_rate_limit_converges(self, rocket):
        """With rate limit, angle eventually reaches commanded value."""
        max_rate = np.deg2rad(10.0)  # 10 deg/s
        gimbal = GimbalActuator(
            rocket=rocket,
            max_outer_angle=np.deg2rad(15.0),
            max_inner_angle=np.deg2rad(15.0),
            max_rate=max_rate,
        )

        # Command 10 degrees
        gimbal.apply(GimbalCommand(
            desired_outer_angle=np.deg2rad(10.0),
            desired_inner_angle=0.0,
        ))

        # After 2 seconds at 10 deg/s, should have reached 10 degrees
        for _ in range(200):
            gimbal.step(0.01)

        assert gimbal.outer_angle == pytest.approx(np.deg2rad(10.0), abs=1e-6)


class TestGimbalActuatorLag:
    """Tests for GimbalActuator first-order lag."""

    def test_lag_slows_response(self, rocket):
        """With tau > 0, angle approaches commanded asymptotically."""
        tau = 0.5  # 0.5 second time constant
        gimbal = GimbalActuator(
            rocket=rocket,
            max_outer_angle=np.deg2rad(15.0),
            max_inner_angle=np.deg2rad(15.0),
            tau=tau,
        )

        # Command 10 degrees
        target = np.deg2rad(10.0)
        gimbal.apply(GimbalCommand(
            desired_outer_angle=target,
            desired_inner_angle=0.0,
        ))

        # After one time constant, should be at ~63% of target
        for _ in range(50):  # 0.5 seconds
            gimbal.step(0.01)

        expected = target * (1 - np.exp(-1))  # ~63%
        assert gimbal.outer_angle == pytest.approx(expected, rel=0.05)

    def test_lag_converges(self, rocket):
        """With tau > 0, angle eventually reaches commanded value."""
        tau = 0.1
        gimbal = GimbalActuator(
            rocket=rocket,
            max_outer_angle=np.deg2rad(15.0),
            max_inner_angle=np.deg2rad(15.0),
            tau=tau,
        )

        target = np.deg2rad(10.0)
        gimbal.apply(GimbalCommand(
            desired_outer_angle=target,
            desired_inner_angle=0.0,
        ))

        # After 10 time constants, should be very close
        for _ in range(100):  # 1 second = 10*tau
            gimbal.step(0.01)

        assert gimbal.outer_angle == pytest.approx(target, rel=0.01)


class TestThrottleActuatorInstantaneous:
    """Tests for ThrottleActuator with default (instantaneous) dynamics."""

    def test_instantaneous_response(self, rocket):
        """With infinite rate, thrust reaches commanded instantly."""
        throttle = ThrottleActuator(rocket=rocket)

        throttle.apply(ThrottleCommand(thrust_N=8000.0))
        throttle.step(0.01)

        assert rocket.state.thrust_body.norm == pytest.approx(8000.0)


class TestThrottleActuatorRateLimiting:
    """Tests for ThrottleActuator rate limiting."""

    def test_rate_limit_clamps_change(self, rocket):
        """With rate limit, thrust changes at most max_rate * dt per step."""
        max_rate = 1000.0  # 1000 N/s
        throttle = ThrottleActuator(
            rocket=rocket,
            max_thrust_rate=max_rate,
        )

        # Initial thrust is 5000N, command 10000N
        throttle.apply(ThrottleCommand(thrust_N=10000.0))

        # After 1 second at 1000 N/s, should increase by 1000N
        for _ in range(100):
            throttle.step(0.01)

        assert rocket.state.thrust_body.norm == pytest.approx(6000.0, abs=1.0)

    def test_rate_limit_decrease(self, rocket):
        """Rate limiting also applies to thrust decrease."""
        max_rate = 2000.0  # 2000 N/s
        throttle = ThrottleActuator(
            rocket=rocket,
            max_thrust_rate=max_rate,
        )

        # Initial thrust is 5000N, command 0N
        throttle.apply(ThrottleCommand(thrust_N=0.0))

        # After 1 second at 2000 N/s, should decrease by 2000N
        for _ in range(100):
            throttle.step(0.01)

        assert rocket.state.thrust_body.norm == pytest.approx(3000.0, abs=1.0)


class TestThrottleActuatorDeadband:
    """Tests for ThrottleActuator deadband."""

    def test_deadband_snaps_to_zero(self, rocket):
        """Thrust below min_thrust snaps to zero."""
        throttle = ThrottleActuator(
            rocket=rocket,
            min_thrust=100.0,  # 100N deadband
        )

        # Command 50N (below deadband)
        throttle.apply(ThrottleCommand(thrust_N=50.0))
        throttle.step(0.01)

        assert rocket.state.thrust_body.norm == pytest.approx(0.0)

    def test_above_deadband_works(self, rocket):
        """Thrust above min_thrust is applied normally."""
        throttle = ThrottleActuator(
            rocket=rocket,
            min_thrust=100.0,
        )

        # Command 150N (above deadband)
        throttle.apply(ThrottleCommand(thrust_N=150.0))
        throttle.step(0.01)

        assert rocket.state.thrust_body.norm == pytest.approx(150.0)


class TestActuatorsBundle:
    """Tests for the Actuators bundle."""

    def test_step_advances_all(self, rocket):
        """Actuators.step() advances all contained actuators."""
        gimbal = GimbalActuator(
            rocket=rocket,
            max_outer_angle=np.deg2rad(15.0),
            max_inner_angle=np.deg2rad(15.0),
            max_rate=np.deg2rad(10.0),  # 10 deg/s
        )
        throttle = ThrottleActuator(
            rocket=rocket,
            max_thrust_rate=1000.0,
        )

        actuators = Actuators(gimbal=gimbal, throttle=throttle)

        # Issue commands
        gimbal.apply(GimbalCommand(
            desired_outer_angle=np.deg2rad(15.0),
            desired_inner_angle=0.0,
        ))
        throttle.apply(ThrottleCommand(thrust_N=10000.0))

        # Step through bundle
        for _ in range(100):
            actuators.step(0.01)

        # Gimbal: 10 deg/s for 1s = 10 deg
        assert gimbal.outer_angle == pytest.approx(np.deg2rad(10.0), abs=1e-6)

        # Throttle: 5000 + 1000 = 6000N
        assert rocket.state.thrust_body.norm == pytest.approx(6000.0, abs=1.0)

    def test_properties_expose_state(self, rocket):
        """Bundle properties expose actuator state for logging."""
        gimbal = GimbalActuator(
            rocket=rocket,
            max_outer_angle=np.deg2rad(15.0),
            max_inner_angle=np.deg2rad(15.0),
        )

        actuators = Actuators(gimbal=gimbal)

        gimbal.apply(GimbalCommand(
            desired_outer_angle=np.deg2rad(5.0),
            desired_inner_angle=np.deg2rad(3.0),
        ))
        gimbal.step(0.01)

        assert actuators.gimbal_outer_angle == pytest.approx(np.deg2rad(5.0))
        assert actuators.gimbal_inner_angle == pytest.approx(np.deg2rad(3.0))

    def test_throttle_command_tracking(self, rocket):
        """Bundle tracks throttle commands for logging."""
        throttle = ThrottleActuator(rocket=rocket)
        actuators = Actuators(throttle=throttle)

        actuators.set_throttle_command(7500.0)

        assert actuators.throttle_command == pytest.approx(7500.0)


class TestBackwardCompatibility:
    """Tests ensuring existing code without dynamics still works."""

    def test_gimbal_without_step_still_works(self, rocket):
        """Old code that calls apply() without step() should still work.

        Note: This tests the case where apply() is called but step() isn't.
        The actual angles won't change until step() is called, but there
        should be no errors.
        """
        gimbal = GimbalActuator(
            rocket=rocket,
            max_outer_angle=np.deg2rad(15.0),
            max_inner_angle=np.deg2rad(15.0),
        )

        # Old pattern: just apply
        gimbal.apply(GimbalCommand(
            desired_outer_angle=np.deg2rad(10.0),
            desired_inner_angle=np.deg2rad(5.0),
        ))

        # Angles should be zero until step is called
        assert gimbal.outer_angle == 0.0
        assert gimbal.inner_angle == 0.0

        # After step, angles update
        gimbal.step(0.01)
        assert gimbal.outer_angle == pytest.approx(np.deg2rad(10.0))
