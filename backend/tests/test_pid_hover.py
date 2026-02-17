"""
Tests for PID hover controller.
"""

import pytest

from simulacra.noumena import (
    Rocket,
    RocketProps,
    RocketState,
    Environment,
    Kernel,
    Scenario,
    get_upright_orientation,
)
from simulacra.phenomena.actuators import ThrottleActuator, Actuators
from simulacra.phenomena.sensors import Sensors
from simulacra.phenomena.errors import GaussianNoise
from simulacra.physics import RocketVec3, WorldVec3
from simulacra.onboard.pid_hover import PIDHoverController


def make_hover_scenario(
    start_alt: float = 5.0,
    target_alt: float = 10.0,
    noisy: bool = False,
    dt: float = 0.02,
) -> tuple[Scenario, Kernel]:
    """Create a hover test scenario."""
    mass = 100.0
    gravity = 9.81
    weight = mass * gravity

    rocket = Rocket(
        props=RocketProps(
            length=4.0,
            mass=mass,
            center_of_mass_offset=RocketVec3(2.0, 0.0, 0.0),
            inertia_diag=(50.0, 50.0, 10.0),
            thrust_application_offset=RocketVec3(0.0, 0.0, 0.0),
        ),
        state=RocketState(
            position=WorldVec3(0.0, start_alt, 0.0),
            velocity=WorldVec3(0.0, 0.0, 0.0),
            orientation=get_upright_orientation(),
            angular_velocity_body=RocketVec3(0.0, 0.0, 0.0),
            thrust_body=RocketVec3(weight, 0.0, 0.0),
        ),
    )

    env = Environment(gravity=WorldVec3(0.0, -gravity, 0.0))
    kernel = Kernel(rocket=rocket, environment=env)

    if noisy:
        sensors = Sensors(
            kernel,
            gps_errors={"y": GaussianNoise(std=0.5, seed=42)},
            velocity_errors={"y": GaussianNoise(std=0.2, seed=43)},
        )
    else:
        sensors = Sensors(kernel)

    throttle = ThrottleActuator(rocket)
    actuators = Actuators(throttle=throttle)

    # For critical damping: Kd = 2 * sqrt(Kp * mass)
    # With Kp=100, mass=100: Kd = 2 * sqrt(10000) = 200
    # Natural frequency ω_n = sqrt(Kp/m) = 1 rad/s, settling time ~4s
    controller = PIDHoverController(
        throttle=throttle,
        target_alt=target_alt,
        weight=weight,
        Kp=100.0,
        Kd=200.0,
        max_thrust=2000.0,
    )

    scenario = Scenario(
        kernel=kernel,
        controller=controller,
        sensors=sensors,
        actuators=actuators,
        dt=dt,
    )

    return scenario, kernel


class TestPIDHoverController:
    """Tests for PIDHoverController."""

    def test_converges_to_target_clean_sensors(self):
        """With clean sensors, controller converges to target altitude."""
        scenario, kernel = make_hover_scenario(
            start_alt=5.0,
            target_alt=10.0,
            noisy=False,
        )

        # Run for 5 seconds
        for _ in range(250):  # 250 * 0.02 = 5s
            scenario.step()

        final_alt = kernel.rocket.state.position.y
        assert final_alt == pytest.approx(10.0, abs=0.3)

    def test_converges_to_target_noisy_sensors(self):
        """With noisy sensors, controller still converges (larger tolerance)."""
        scenario, kernel = make_hover_scenario(
            start_alt=5.0,
            target_alt=10.0,
            noisy=True,
        )

        # Run for 5 seconds
        for _ in range(250):
            scenario.step()

        final_alt = kernel.rocket.state.position.y
        # Larger tolerance due to noise
        assert final_alt == pytest.approx(10.0, abs=1.0)

    def test_holds_altitude_when_at_target(self):
        """Starting at target, controller maintains altitude."""
        scenario, kernel = make_hover_scenario(
            start_alt=10.0,
            target_alt=10.0,
            noisy=False,
        )

        # Run for 3 seconds
        for _ in range(150):
            scenario.step()

        final_alt = kernel.rocket.state.position.y
        assert final_alt == pytest.approx(10.0, abs=0.05)

    def test_descends_when_above_target(self):
        """Starting above target, controller descends."""
        scenario, kernel = make_hover_scenario(
            start_alt=15.0,
            target_alt=10.0,
            noisy=False,
        )

        # Run for 5 seconds
        for _ in range(250):
            scenario.step()

        final_alt = kernel.rocket.state.position.y
        assert final_alt == pytest.approx(10.0, abs=0.3)

    def test_velocity_settles_to_zero(self):
        """Controller damps velocity to near zero at target."""
        scenario, kernel = make_hover_scenario(
            start_alt=5.0,
            target_alt=10.0,
            noisy=False,
        )

        # Run for 5 seconds
        for _ in range(250):
            scenario.step()

        final_vy = kernel.rocket.state.velocity.y
        assert abs(final_vy) < 0.2  # Nearly stationary
