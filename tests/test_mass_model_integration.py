"""
Integration tests: MassModel with Kernel.

Verifies that fuel burn during thrusting actually changes the rocket's
mass/props, and that rockets without a mass_model behave identically
to the existing behavior (regression).
"""

import numpy as np

from simulacra.noumena.kernel import Kernel
from simulacra.noumena.mass_model import FuelComponent, MassComponent, MassModel
from simulacra.noumena.rocket import Rocket, RocketProps, RocketState, get_upright_orientation
from simulacra.noumena.world import Environment
from simulacra.physics import RocketVec3, WorldVec3


def make_rocket_with_mass_model(thrust: float = 100.0) -> tuple[Rocket, Environment]:
    """Rocket with a simple mass model: engine + fuel."""
    model = MassModel(
        components=[
            MassComponent("engine", 10.0, RocketVec3(0.0, 0.0, 0.0)),
            FuelComponent(
                "fuel", 20.0, RocketVec3(2.0, 0.0, 0.0),
                full_mass=20.0, burn_rate=5.0,
            ),
        ]
    )
    props = model.to_rocket_props(
        length=4.0,
        thrust_application_offset=RocketVec3(0.0, 0.0, 0.0),
    )
    state = RocketState(
        position=WorldVec3(0.0, 100.0, 0.0),
        velocity=WorldVec3(0.0, 0.0, 0.0),
        orientation=get_upright_orientation(),
        angular_velocity_body=RocketVec3(0.0, 0.0, 0.0),
        thrust_body=RocketVec3(thrust, 0.0, 0.0),
    )
    rocket = Rocket(props=props, state=state, mass_model=model)
    env = Environment()
    return rocket, env


def make_rocket_without_mass_model(
    mass: float = 30.0, thrust: float = 100.0
) -> tuple[Rocket, Environment]:
    """Rocket without a mass model (existing behavior)."""
    props = RocketProps(
        length=4.0,
        mass=mass,
        center_of_mass_offset=RocketVec3(1.0, 0.0, 0.0),
        inertia_diag=(1.0, 1.0, 1.0),
        thrust_application_offset=RocketVec3(0.0, 0.0, 0.0),
    )
    state = RocketState(
        position=WorldVec3(0.0, 100.0, 0.0),
        velocity=WorldVec3(0.0, 0.0, 0.0),
        orientation=get_upright_orientation(),
        angular_velocity_body=RocketVec3(0.0, 0.0, 0.0),
        thrust_body=RocketVec3(thrust, 0.0, 0.0),
    )
    rocket = Rocket(props=props, state=state)
    env = Environment()
    return rocket, env


def test_mass_decreases_when_thrusting():
    rocket, env = make_rocket_with_mass_model(thrust=100.0)
    initial_mass = rocket.props.mass

    kernel = Kernel(rocket, env)
    kernel.step(0.1)

    # 5 kg/s * 0.1 s = 0.5 kg burned
    assert rocket.props.mass < initial_mass
    np.testing.assert_allclose(
        rocket.props.mass, initial_mass - 0.5, atol=1e-12
    )


def test_com_shifts_when_thrusting():
    rocket, env = make_rocket_with_mass_model(thrust=100.0)
    initial_com_x = rocket.props.center_of_mass_offset.x

    kernel = Kernel(rocket, env)
    for _ in range(10):
        kernel.step(0.1)

    # Fuel is at x=2, engine at x=0. As fuel depletes, CoM shifts toward engine.
    assert rocket.props.center_of_mass_offset.x < initial_com_x


def test_no_fuel_burn_when_not_thrusting():
    rocket, env = make_rocket_with_mass_model(thrust=0.0)
    initial_mass = rocket.props.mass

    kernel = Kernel(rocket, env)
    kernel.step(0.1)

    # No thrust -> no fuel burn -> mass unchanged
    assert rocket.props.mass == initial_mass


def test_constant_mass_model_mass_stays_constant():
    """Regression: rockets with constant mass_model keep constant mass."""
    rocket, env = make_rocket_without_mass_model(mass=30.0, thrust=100.0)

    kernel = Kernel(rocket, env)
    for _ in range(100):
        kernel.step(0.01)

    assert rocket.props.mass == 30.0


def test_default_mass_model_is_constant():
    """Rocket without explicit mass_model gets a constant MassModel."""
    rocket, _ = make_rocket_without_mass_model()
    assert isinstance(rocket.mass_model, MassModel)
    assert rocket.mass_model.fuel_capacity == 0.0
