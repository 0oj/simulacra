"""
Tests for the full TVC hover controller with altitude and attitude control.

The HoverController uses:
- Outer loop: PD altitude control → thrust magnitude
- Inner loop: Quaternion PD + gravity feedforward → gimbal angles

Key feature: gravity feedforward allows recovery from large tilts that
pure PD feedback cannot handle.
"""

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from simulacra.noumena import (
    Rocket,
    RocketProps,
    RocketState,
    Environment,
    Kernel,
    Scenario,
    get_upright_orientation,
)
from simulacra.phenomena.actuators import (
    GimbalActuator,
    ThrottleActuator,
    Actuators,
)
from simulacra.phenomena.sensors import Sensors
from simulacra.phenomena.errors import GaussianNoise
from simulacra.physics import Orientation, RocketVec3, WorldVec3
from simulacra.onboard.hover_controller import HoverController, HoverControllerConfig


def make_tilted_orientation(tilt_deg: float, axis: str = "z") -> Orientation:
    """Create an orientation tilted from vertical by given degrees."""
    base = get_upright_orientation()._as_scipy_rotation()
    tilt = R.from_euler(axis, math.radians(tilt_deg))
    combined = tilt * base
    x, y, z, w = combined.as_quat()
    return Orientation.from_wxyz(w, x, y, z)


def make_hover_scenario(
    start_alt: float = 10.0,
    target_alt: float = 10.0,
    tilt_deg: float = 0.0,
    noisy: bool = False,
    dt: float = 0.02,
    config: HoverControllerConfig | None = None,
) -> tuple[Scenario, Kernel]:
    """Create a hover test scenario with full TVC control."""
    mass = 100.0
    gravity = 9.81
    weight = mass * gravity
    length = 4.0
    thrust_moment_arm = 2.0  # CoM to thrust point

    if tilt_deg != 0.0:
        orientation = make_tilted_orientation(tilt_deg)
    else:
        orientation = get_upright_orientation()

    rocket = Rocket(
        props=RocketProps(
            length=length,
            mass=mass,
            center_of_mass_offset=RocketVec3(thrust_moment_arm, 0.0, 0.0),
            inertia_diag=(50.0, 50.0, 10.0),
            thrust_application_offset=RocketVec3(0.0, 0.0, 0.0),
        ),
        state=RocketState(
            position=WorldVec3(0.0, start_alt, 0.0),
            velocity=WorldVec3(0.0, 0.0, 0.0),
            orientation=orientation,
            angular_velocity_body=RocketVec3(0.0, 0.0, 0.0),
            thrust_body=RocketVec3(weight, 0.0, 0.0),
        ),
    )

    env = Environment(gravity=WorldVec3(0.0, -gravity, 0.0))
    kernel = Kernel(rocket=rocket, environment=env)

    if noisy:
        sensors = Sensors(
            kernel,
            gps_errors={"y": GaussianNoise(std=0.3, seed=42)},
            velocity_errors={"y": GaussianNoise(std=0.1, seed=43)},
            gyro_errors={
                "x": GaussianNoise(std=math.radians(0.5), seed=44),
                "y": GaussianNoise(std=math.radians(0.5), seed=45),
                "z": GaussianNoise(std=math.radians(0.5), seed=46),
            },
        )
    else:
        sensors = Sensors(kernel)

    gimbal = GimbalActuator(
        rocket=rocket,
        max_outer_angle=math.radians(15.0),
        max_inner_angle=math.radians(15.0),
        max_rate=math.radians(60.0),  # 60 deg/s
        tau=0.05,
    )
    throttle = ThrottleActuator(
        rocket=rocket,
        max_thrust_rate=weight * 4,  # can ramp quickly
    )
    actuators = Actuators(gimbal=gimbal, throttle=throttle)

    cfg = config or HoverControllerConfig(
        target_alt=target_alt,
        weight=weight,
        max_thrust=2000.0,
        gravity=gravity,
        thrust_moment_arm=thrust_moment_arm,
    )

    controller = HoverController(
        throttle=throttle,
        gimbal=gimbal,
        config=cfg,
    )

    scenario = Scenario(
        kernel=kernel,
        controller=controller,
        sensors=sensors,
        actuators=actuators,
        dt=dt,
    )

    return scenario, kernel


def get_tilt_angle(kernel: Kernel) -> float:
    """Compute tilt angle from vertical in degrees."""
    orient = kernel.rocket.state.orientation
    nose_body = RocketVec3(1.0, 0.0, 0.0)
    nose_world = nose_body.to_world(orient)
    cos_angle = max(-1.0, min(1.0, nose_world.y))
    return math.degrees(math.acos(cos_angle))


class TestHoverControllerAltitude:
    """Tests for altitude control (outer loop)."""

    def test_holds_altitude_when_at_target(self):
        """Starting at target altitude, controller maintains it."""
        scenario, kernel = make_hover_scenario(
            start_alt=10.0,
            target_alt=10.0,
            tilt_deg=0.0,
        )

        for _ in range(150):  # 3 seconds
            scenario.step()

        final_alt = kernel.rocket.state.position.y
        assert final_alt == pytest.approx(10.0, abs=0.1)

    def test_converges_from_below(self):
        """Starting below target, controller ascends."""
        scenario, kernel = make_hover_scenario(
            start_alt=5.0,
            target_alt=10.0,
            tilt_deg=0.0,
        )

        for _ in range(300):  # 6 seconds
            scenario.step()

        final_alt = kernel.rocket.state.position.y
        assert final_alt == pytest.approx(10.0, abs=0.5)

    def test_converges_from_above(self):
        """Starting above target, controller descends."""
        scenario, kernel = make_hover_scenario(
            start_alt=15.0,
            target_alt=10.0,
            tilt_deg=0.0,
        )

        for _ in range(300):  # 6 seconds
            scenario.step()

        final_alt = kernel.rocket.state.position.y
        assert final_alt == pytest.approx(10.0, abs=0.5)


class TestHoverControllerAttitude:
    """Tests for attitude control (inner loop)."""

    def test_maintains_upright_when_vertical(self):
        """Starting upright, controller keeps rocket vertical."""
        scenario, kernel = make_hover_scenario(
            start_alt=10.0,
            target_alt=10.0,
            tilt_deg=0.0,
        )

        for _ in range(150):
            scenario.step()

        tilt = get_tilt_angle(kernel)
        assert tilt < 1.0  # Less than 1 degree

    def test_recovers_from_small_tilt(self):
        """Controller recovers from 5-degree initial tilt."""
        scenario, kernel = make_hover_scenario(
            start_alt=10.0,
            target_alt=10.0,
            tilt_deg=5.0,
        )

        initial_tilt = get_tilt_angle(kernel)
        assert initial_tilt == pytest.approx(5.0, abs=0.5)

        for _ in range(150):  # 3 seconds
            scenario.step()

        final_tilt = get_tilt_angle(kernel)
        assert final_tilt < 2.0  # Recovered to near vertical

    def test_recovers_from_large_tilt(self):
        """Controller recovers from 15-degree initial tilt (requires feedforward)."""
        scenario, kernel = make_hover_scenario(
            start_alt=10.0,
            target_alt=10.0,
            tilt_deg=15.0,
        )

        initial_tilt = get_tilt_angle(kernel)
        assert initial_tilt == pytest.approx(15.0, abs=1.0)

        for _ in range(200):  # 4 seconds
            scenario.step()

        final_tilt = get_tilt_angle(kernel)
        # With gravity feedforward, should recover even from large tilts
        assert final_tilt < 3.0

    def test_angular_velocity_settles(self):
        """After recovery, angular velocity is near zero."""
        scenario, kernel = make_hover_scenario(
            start_alt=10.0,
            target_alt=10.0,
            tilt_deg=10.0,
        )

        for _ in range(200):
            scenario.step()

        omega = kernel.rocket.state.angular_velocity_body
        omega_mag = math.sqrt(omega.x**2 + omega.y**2 + omega.z**2)
        assert omega_mag < 0.1  # rad/s


class TestHoverControllerCombined:
    """Tests for combined altitude and attitude control."""

    def test_altitude_and_tilt_together(self):
        """Controller handles both altitude error and tilt simultaneously."""
        scenario, kernel = make_hover_scenario(
            start_alt=7.0,
            target_alt=10.0,
            tilt_deg=10.0,
        )

        for _ in range(300):  # 6 seconds
            scenario.step()

        final_alt = kernel.rocket.state.position.y
        final_tilt = get_tilt_angle(kernel)

        assert final_alt == pytest.approx(10.0, abs=0.5)
        assert final_tilt < 2.0

    def test_velocity_settles_to_zero(self):
        """Velocity converges to near zero at hover."""
        scenario, kernel = make_hover_scenario(
            start_alt=8.0,
            target_alt=10.0,
            tilt_deg=5.0,
        )

        for _ in range(300):
            scenario.step()

        vel = kernel.rocket.state.velocity
        vel_mag = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        assert vel_mag < 0.3  # m/s


class TestHoverControllerWithNoise:
    """Tests with sensor noise."""

    def test_maintains_hover_with_noise(self):
        """Controller maintains approximate hover despite noisy sensors."""
        scenario, kernel = make_hover_scenario(
            start_alt=10.0,
            target_alt=10.0,
            tilt_deg=0.0,
            noisy=True,
        )

        altitudes = []
        tilts = []

        for _ in range(250):  # 5 seconds
            scenario.step()
            altitudes.append(kernel.rocket.state.position.y)
            tilts.append(get_tilt_angle(kernel))

        # Should stay near target (larger tolerance for noise)
        mean_alt = np.mean(altitudes[-50:])  # Last second
        mean_tilt = np.mean(tilts[-50:])

        assert mean_alt == pytest.approx(10.0, abs=1.0)
        assert mean_tilt < 3.0

    def test_recovers_from_tilt_with_noise(self):
        """Controller recovers from tilt even with sensor noise."""
        scenario, kernel = make_hover_scenario(
            start_alt=10.0,
            target_alt=10.0,
            tilt_deg=10.0,
            noisy=True,
        )

        for _ in range(250):
            scenario.step()

        final_tilt = get_tilt_angle(kernel)
        assert final_tilt < 5.0  # Larger tolerance with noise


class TestHoverControllerConfig:
    """Tests for configuration options."""

    def test_custom_gains(self):
        """Controller works with custom gain configuration."""
        config = HoverControllerConfig(
            target_alt=15.0,
            weight=981.0,
            max_thrust=3000.0,
            alt_Kp=150.0,
            alt_Kd=250.0,
            att_Kp=600.0,
            att_Kd=250.0,
        )

        scenario, kernel = make_hover_scenario(
            start_alt=10.0,
            target_alt=15.0,
            config=config,
        )

        for _ in range(300):
            scenario.step()

        final_alt = kernel.rocket.state.position.y
        assert final_alt == pytest.approx(15.0, abs=0.5)

    def test_requires_sensors(self):
        """Controller raises error if run without sensors."""
        mass = 100.0
        rocket = Rocket(
            props=RocketProps(
                length=4.0,
                mass=mass,
                center_of_mass_offset=RocketVec3(2.0, 0.0, 0.0),
                inertia_diag=(50.0, 50.0, 10.0),
                thrust_application_offset=RocketVec3(0.0, 0.0, 0.0),
            ),
            state=RocketState(
                position=WorldVec3(0.0, 10.0, 0.0),
                velocity=WorldVec3(0.0, 0.0, 0.0),
                orientation=get_upright_orientation(),
                angular_velocity_body=RocketVec3(0.0, 0.0, 0.0),
                thrust_body=RocketVec3(981.0, 0.0, 0.0),
            ),
        )

        gimbal = GimbalActuator(
            rocket=rocket,
            max_outer_angle=math.radians(15.0),
            max_inner_angle=math.radians(15.0),
        )
        throttle = ThrottleActuator(rocket=rocket)

        controller = HoverController(throttle=throttle, gimbal=gimbal)

        with pytest.raises(ValueError, match="Sensors not attached"):
            controller.run()


class TestHoverControllerHelpers:
    """Tests for internal helper methods."""

    def test_tilt_factor_upright(self):
        """Tilt factor is 1.0 when upright."""
        scenario, kernel = make_hover_scenario(tilt_deg=0.0)

        # Access controller through scenario
        controller = scenario.controller
        orient = kernel.rocket.state.orientation
        quat = tuple(orient.as_array())

        tilt_factor = controller._compute_tilt_factor(quat)
        assert tilt_factor == pytest.approx(1.0, abs=0.01)

    def test_tilt_factor_tilted(self):
        """Tilt factor decreases with tilt."""
        scenario, kernel = make_hover_scenario(tilt_deg=30.0)

        controller = scenario.controller
        orient = kernel.rocket.state.orientation
        quat = tuple(orient.as_array())

        tilt_factor = controller._compute_tilt_factor(quat)
        # cos(30°) ≈ 0.866
        assert tilt_factor == pytest.approx(math.cos(math.radians(30.0)), abs=0.05)

    def test_quaternion_error_identity(self):
        """Quaternion error is identity when current equals desired."""
        scenario, _ = make_hover_scenario()
        controller = scenario.controller

        quat = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))  # upright
        quat_desired = quat

        error = controller._quaternion_error(quat, quat_desired)

        # Identity quaternion: (1, 0, 0, 0)
        assert error[0] == pytest.approx(1.0, abs=0.01)
        assert abs(error[1]) < 0.01
        assert abs(error[2]) < 0.01
        assert abs(error[3]) < 0.01
